from functools import partial

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.common.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
from lerobot.common.policies.pretrained import PreTrainedPolicy
from PIL import Image
from scipy.fft import idct
from termcolor import cprint
from torch import Tensor, nn
from transformers import AutoProcessor, AutoTokenizer, PaliGemmaForConditionalGeneration
from transformers.cache_utils import HybridCache, StaticCache
from transformers.models.auto import CONFIG_MAPPING

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

PRECISION = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def get_paligemma_config(torch_dtype, **kwargs):
    return CONFIG_MAPPING["paligemma"](
        transformers_version="4.48.1",
        _vocab_size=257152,
        bos_token_id=2,
        eos_token_id=1,
        hidden_size=2048,
        image_token_index=257152,
        model_type="paligemma",
        pad_token_id=0,
        projection_dim=2048,
        text_config={
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 2048,
            "intermediate_size": 16384,
            "model_type": "gemma",
            "num_attention_heads": 8,
            "num_hidden_layers": 18,
            "num_image_tokens": 256,
            "num_key_value_heads": 1,
            "torch_dtype": torch_dtype,
            "vocab_size": 257152,
            "_attn_implementation": "eager",
        },
        vision_config={
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "model_type": "siglip_vision_model",
            "num_attention_heads": 16,
            "num_hidden_layers": 27,
            "num_image_tokens": 256,
            "patch_size": 14,
            "projection_dim": 2048,
            "projector_hidden_act": "gelu_pytorch_tanh",
            "torch_dtype": torch_dtype,
            "vision_use_head": False,
        },
        **kwargs,
    )


class PI0FASTPolicy(PreTrainedPolicy):
    config_class = PI0FASTConfig
    name = "torch_pi0fast"

    def __init__(
        self,
        config: PI0FASTConfig,
        tokenizer_path: str = "google/paligemma-3b-pt-224",
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        self.config = config
        self.language_tokenizer = AutoProcessor.from_pretrained(tokenizer_path)
        self.model = PI0FAST(config)
        self.reset()

    def reset(self):
        return None

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        """
        Observation: {
            "image": {
                "base_0_rgb": (*b, c, h, w),  # uint8 [0, 255]
                ...
            },
            "state": float32 [*b, s],
            "prompt": List[str],

            "lang_tokens": float32 [*b, l],
            "lang_masks": float32 [*b, l],
        }
        """
        self.eval()

        actions = self.model.generate_actions(observation)
        return actions

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        loss_dict = self.model.forward(batch)
        return loss_dict["loss"], loss_dict


def block_causal_update_causal_mask(
    attention_mask,
    token_type_ids=None,
    past_key_values=None,
    cache_position=None,
    input_tensor=None,
    attn_implementation: str = "eager",
    dtype: torch.dtype = "float32",
):
    """
    Update the causal mask during training and generation. It can be customized to different attention masks.
    """
    if attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None
    using_static_cache = isinstance(past_key_values, StaticCache)
    min_dtype = torch.finfo(dtype).min

    if input_tensor is None:
        input_tensor = attention_mask

    inputs_lead_dim, sequence_length = input_tensor.shape[:2]

    if using_static_cache or isinstance(past_key_values, HybridCache):
        target_length = past_key_values.get_max_cache_shape()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else cache_position[0] + sequence_length + 1
        )

    # Handle precomputed attention masks
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask

    # Causal mask initialization
    causal_mask = torch.full(
        (sequence_length, target_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=cache_position.device,
    )

    # Standard causal masking (triu ensures tokens can only attend to past)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)

        # Apply block causal mask
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(causal_mask.device).bool()
            cumsum = torch.cumsum(token_type_ids, dim=1)
            block_causal_mask = cumsum[:, None, :] <= cumsum[:, :, None]

            # Combine causal_mask with block-wise attention mask
            causal_mask = torch.where(block_causal_mask, 0.0, causal_mask)
            causal_mask = causal_mask[:, None, :, :]
        else:
            # Apply past cache position constraint
            causal_mask *= torch.arange(
                target_length, device=cache_position.device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(
                inputs_lead_dim, 1, -1, -1
            )
    else:
        # Apply past cache position constraint
        causal_mask *= torch.arange(
            target_length, device=cache_position.device
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)

    if attention_mask is not None:
        causal_mask = (
            causal_mask.clone()
        )  # Copy to contiguous memory for in-place edits
        mask_length = attention_mask.shape[-1]

        # Apply padding mask
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
            :, None, None, :
        ].to(causal_mask.device)
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[
            :, :, :, :mask_length
        ].masked_fill(padding_mask, min_dtype)

    return causal_mask


def prepare_inputs_for_generation(
    # self,
    input_ids,
    past_key_values=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    pixel_values=None,
    attention_mask=None,
    token_type_ids=None,
    use_cache=True,
    num_logits_to_keep=None,
    labels=None,
    self=None,
    **kwargs,
):
    # create block causal attention
    if cache_position[0] > 0 and input_ids.shape[1] > 0:
        input_tensor = input_ids[:, -1:]
        new_positions = (
            torch.ones(
                (position_ids.shape[0], input_ids.shape[1]),
                dtype=position_ids.dtype,
                device=position_ids.device,
            ).cumsum(-1)
            + position_ids[:, -1:]
        )
        position_ids = torch.cat([position_ids, new_positions], dim=-1)
    else:
        input_tensor = inputs_embeds
    attention_mask = block_causal_update_causal_mask(
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        cache_position=cache_position,
        input_tensor=input_tensor,
        token_type_ids=token_type_ids,
        dtype=self.dtype,
        attn_implementation=self.config.text_config._attn_implementation,
    )
    # Overwritten -- custom `position_ids` and `pixel_values` handling
    model_inputs = self.language_model.prepare_inputs_for_generation(
        input_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=use_cache,
        num_logits_to_keep=num_logits_to_keep,
        token_type_ids=token_type_ids,
        **kwargs,
    )

    # Position_ids in Paligemma are 1-indexed
    if model_inputs.get("position_ids") is not None:
        model_inputs["position_ids"] += 1
    # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
    # Otherwise we need pixel values to be passed to model. NOTE: use_cache=False needs pixel_values always
    if cache_position[0] == 0:
        model_inputs["pixel_values"] = pixel_values
    is_training = token_type_ids is not None and labels is not None
    if cache_position[0] == 0 and isinstance(past_key_values, HybridCache):
        input_tensor = inputs_embeds if inputs_embeds is not None else input_ids
        causal_mask = self._update_causal_mask(
            attention_mask,
            token_type_ids,
            past_key_values,
            cache_position,
            input_tensor,
            is_training,
        )
        model_inputs["attention_mask"] = causal_mask

    return model_inputs


class PI0FAST(nn.Module):
    def __init__(self, config: PI0FASTConfig):
        super().__init__()
        self.config = config

        fast_tokenizer_path = "physical-intelligence/fast"
        pi0_paligemma_path = "google/paligemma-3b-pt-224"
        self.paligemma_tokenizer = AutoTokenizer.from_pretrained(pi0_paligemma_path)
        self.processor = AutoProcessor.from_pretrained(pi0_paligemma_path)
        self.fast_tokenizer = AutoProcessor.from_pretrained(
            fast_tokenizer_path, trust_remote_code=True
        )
        self.fast_skip_tokens = self.config.fast_skip_tokens
        self.max_input_seq_len = self.config.max_input_seq_len
        self.action_horizon = self.config.chunk_size
        self.action_dim = self.config.max_action_dim
        precision = config.precision
        torch_precision = PRECISION.get(precision, torch.float32)

        self.pad_token_id = (
            self.paligemma_tokenizer.pad_token_id
            if hasattr(self.paligemma_tokenizer, "pad_token_id")
            else self.paligemma_tokenizer.eos_token_id
        )

        paligemma_config = get_paligemma_config(torch_dtype=torch_precision)
        self.pi0_paligemma = PaliGemmaForConditionalGeneration(config=paligemma_config)

        self.pi0_paligemma.prepare_inputs_for_generation = partial(
            prepare_inputs_for_generation, self=self.pi0_paligemma
        )
        # change important stuff in bf16
        params_to_change_dtype = [
            "language_model",
            "vision_tower",
            "multi_modal",
        ]
        for name, param in self.pi0_paligemma.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch_precision)
        self.set_requires_grad()
        self.image_keys = self.config.image_features.keys()
        self.ignore_index = self.pi0_paligemma.config.ignore_index
        self.padding_side = self.config.padding_side

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.pi0_paligemma.vision_tower.eval()
            for params in self.pi0_paligemma.vision_tower.parameters():
                params.requires_grad = False
        # To avoid unused params issue with distributed training
        if self.config.freeze_lm_head:
            for name, params in self.pi0_paligemma.named_parameters():
                if "embed_tokens" in name:  # lm heads and embedding layer are tied
                    params.requires_grad = False

    def embed_tokens(self, tokens: torch.Tensor):
        return self.pi0_paligemma.language_model.model.embed_tokens(tokens)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.pi0_paligemma.prepare_inputs_for_generation(*args, **kwargs)

    def prepare_images(self, observation: dict[str, Tensor]):
        """Use zeros for unpresented views without padding masks."""
        dtype = observation["state"].dtype
        bsize = observation["state"].shape[0]
        images, img_masks = [], []
        for key in IMAGE_KEYS:
            if key in observation["image"]:
                # resize, pad, and normalize
                img = observation["image"][key]
                img = img.to(dtype) / 127.5 - 1.0
                img = resize_with_pad(
                    img, *self.config.resize_imgs_with_padding, pad_value=-1.0
                )
                images.append(img)
                img_masks.append(
                    torch.ones((bsize,), dtype=torch.bool, device=img.device)
                )
            else:
                img = torch.full_like(img, fill_value=-1.0)
                images.append(img)
                img_masks.append(
                    torch.ones((bsize,), dtype=torch.bool, device=img.device)
                )
        images = torch.stack(images, dim=1)  # (*b, n, c, h, w)
        img_masks = torch.stack(img_masks, dim=1)  # (*b, n)
        return images, img_masks

    def _act_tokens_to_paligemma_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.paligemma_tokenizer.vocab_size - 1 - self.fast_skip_tokens - tokens
        return out

    def fast_tokenizer_wrapper(self, actions_norm):
        actions_norm = actions_norm.to(torch.float32)
        batch_tokens = self.fast_tokenizer(actions_norm)
        fast_out = self.processor.tokenizer.pad(
            {"input_ids": batch_tokens}, return_tensors="pt"
        )
        return fast_out

    def create_token_type_ids(
        self, padded_mask: torch.Tensor, prefix_len: int
    ) -> torch.Tensor:
        token_type_ids = torch.zeros_like(padded_mask, dtype=torch.bool)
        # Compute cumulative sum mask
        cumsum_mask = (padded_mask != 0).cumsum(dim=1)
        # Suffix block (everything after prefix_len)
        suffix_mask = cumsum_mask > prefix_len
        token_type_ids = suffix_mask
        return token_type_ids

    def create_input_tokens(self, state, lang_text, actions=None):
        bsize = state.shape[0]
        device = state.device

        # Note that `state` is expected to be normalized to [-1, 1] range.
        bins = torch.linspace(-1, 1, 256 + 1, device=device)[:-1]
        discretized = torch.bucketize(state, bins) - 1

        prefix_texts = []
        for txt, disc in zip(lang_text, discretized, strict=False):
            cleaned = txt.lower().strip().replace("_", " ")
            state_str = " ".join(str(val.item()) for val in disc)
            prefix_texts.append(f"Task: {cleaned}, State: {state_str};\n")

        # tokenizer automatically adds <bos> token
        prefix_out = self.paligemma_tokenizer(
            prefix_texts,
            add_special_tokens=True,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )
        prefix_ids = prefix_out["input_ids"].to(device)
        prefix_mask = prefix_out["attention_mask"].to(device)
        prefix_lens = prefix_mask.sum(dim=1)[:, None].cpu()

        if actions is not None:
            # see https://github.com/Physical-Intelligence/openpi/blob/0992224b1cf89d0fe282ac381596c1048b766adc/src/openpi/models/tokenizer.py#L39
            # JAX OpenPI does not:
            # 1. normalize action before passing to fast tokenizer
            # 2. pad action to 32dim
            # 3. replace action token with 0 to pad tokens
            # And it does:
            # 1. add "|" after action tokens

            fast_out = self.fast_tokenizer_wrapper(actions.cpu())
            act_ids = fast_out["input_ids"]
            act_mask = fast_out["attention_mask"].to(device)
            act_ids = self._act_tokens_to_paligemma_tokens(act_ids).to(device)
            act_ids[torch.where(1 - act_mask)] = self.paligemma_tokenizer.pad_token_id
            bos = self.paligemma_tokenizer(
                "Action: ", add_special_tokens=False, return_tensors="pt"
            )
            eos = self.paligemma_tokenizer(
                "|<eos>", add_special_tokens=False, return_tensors="pt"
            )
            bos_token = bos["input_ids"].expand(act_ids.shape[0], -1).to(device)
            bos_mask = bos["attention_mask"].expand(act_ids.shape[0], -1).to(device)
            eos_token = eos["input_ids"].expand(act_ids.shape[0], -1).to(device)
            eos_mask = eos["attention_mask"].expand(act_ids.shape[0], -1).to(device)
            act_ids = torch.cat([bos_token, act_ids, eos_token], dim=1)
            act_mask = torch.cat([bos_mask, act_mask, eos_mask], dim=1)
            act_mask = act_mask.to(device)
        else:
            act_ids = torch.empty(bsize, 0, dtype=torch.long, device=device)
            act_mask = torch.empty(bsize, 0, dtype=torch.long, device=device)

        final_ids = torch.cat([prefix_ids, act_ids], dim=1)
        final_mask = torch.cat([prefix_mask, act_mask], dim=1)
        batch_inputs = {
            "input_ids": final_ids.tolist(),
            "attention_mask": final_mask.tolist(),
        }

        # Use tokenizer pad function
        padded_output = self.paligemma_tokenizer.pad(
            batch_inputs, padding="longest", max_length=180, return_tensors="pt"
        )
        padded_mask = padded_output["attention_mask"]

        # define tensor of padding lengths
        att_mask = (padded_mask != 0).cumsum(dim=1) > prefix_lens

        token_type_ids = self.create_token_type_ids(
            padded_mask=padded_mask, prefix_len=prefix_lens
        )

        padded_output["padded_mask"] = padded_output.pop("attention_mask")
        padded_output["attention_mask"] = att_mask
        # loss is computed not on prefix, and not on padding
        padded_output["loss_mask"] = att_mask & padded_output["padded_mask"]
        padded_output["token_type_ids"] = token_type_ids
        return padded_output

    def shift_padding_side(
        self,
        tokens: torch.Tensor,
        ar_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        targets: torch.Tensor,
        token_type_ids: torch.Tensor,
        padding_side: str = "right",
    ) -> tuple[torch.Tensor]:
        if padding_side not in ["right", "left"]:
            return tokens, ar_mask, padding_mask, loss_mask, targets, token_type_ids

        new_tokens = torch.empty_like(tokens)
        new_ar_masks = torch.empty_like(ar_mask)
        new_padding_mask = torch.empty_like(padding_mask)
        new_loss_mask = torch.empty_like(loss_mask)
        new_targets = torch.empty_like(targets)
        new_token_type_ids = torch.empty_like(token_type_ids)
        batch_size = tokens.shape[0]
        for i in range(batch_size):
            padding_indices = torch.where(padding_mask[i] == 0)[0]
            non_padding_indices = torch.where(padding_mask[i] == 1)[0]
            if padding_side == "left":
                new_indices = torch.cat((padding_indices, non_padding_indices), dim=0)
            else:
                new_indices = torch.cat((non_padding_indices, padding_indices), dim=0)
            new_tokens[i] = tokens[i].index_select(0, new_indices)
            new_ar_masks[i] = ar_mask[i].index_select(0, new_indices)
            new_padding_mask[i] = padding_mask[i].index_select(0, new_indices)
            new_loss_mask[i] = loss_mask[i].index_select(0, new_indices)
            new_targets[i] = targets[i].index_select(0, new_indices)
            new_token_type_ids[i] = token_type_ids[i].index_select(0, new_indices)

        return (
            new_tokens,
            new_ar_masks,
            new_padding_mask,
            new_loss_mask,
            new_targets,
            new_token_type_ids,
        )

    def forward(self, batch: dict[str, Tensor]):
        device = batch["state"].device
        images, img_masks = self.prepare_images(batch)
        padded_outs = self.create_input_tokens(
            state=batch["state"],
            lang_text=batch["prompt"],
            actions=batch["action"],
        )
        embs, pad_masks, _, targets, loss_mask, token_type_ids = self.embed_inputs(
            images,
            img_masks,
            padded_outs["input_ids"],
            padded_outs["padded_mask"],
            padded_outs["attention_mask"],
            padded_outs["loss_mask"],
            padded_outs["token_type_ids"],
            padding_side=self.padding_side,
        )
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        token_type_ids = token_type_ids.to(dtype=torch.int64)
        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + embs.shape[1], device=embs.device
        )
        pad_masks = block_causal_update_causal_mask(
            attention_mask=pad_masks,
            past_key_values=None,
            cache_position=cache_position,
            input_tensor=embs,
            token_type_ids=token_type_ids,
            dtype=self.pi0_paligemma.dtype,
            attn_implementation=self.pi0_paligemma.config.text_config._attn_implementation,
        )
        outputs = self.pi0_paligemma.forward(
            input_ids=None,
            token_type_ids=None,
            attention_mask=pad_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=embs,
            use_cache=False,
            labels=None,
        )

        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Shift left for next-step prediction
        logits = logits[:, 588:-1, :]
        targets = targets[:, 588 + 1 :].to(device)  # Shift targets
        loss_mask = loss_mask[:, 588 + 1 :].to(device)  # Ensure correct shape

        # Compute per-token loss
        token_loss = loss_fct(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        # Apply loss mask
        token_loss = token_loss * loss_mask.reshape(-1)

        # Compute final loss
        loss = token_loss.sum() / torch.clamp(loss_mask.sum(), min=1)

        # accuracy
        with torch.no_grad():
            acc = (logits.argmax(-1) == targets)[loss_mask].float().mean()

        # Return loss dictionary
        loss_dict = {"ce_loss": loss.item(), "loss": loss, "acc": acc.item()}
        return loss_dict

    def decode_actions_with_fast(
        self,
        tokens: list[list[int]],
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
        relaxed_decoding: bool = True,
    ) -> np.array:
        """
        Adapt original decoding in FAST to always return actions instead of zeros.
        """
        self.time_horizon = (
            time_horizon
            or self.fast_tokenizer.time_horizon
            or self.fast_tokenizer.called_time_horizon
        )
        self.action_dim = (
            action_dim
            or self.fast_tokenizer.action_dim
            or self.fast_tokenizer.called_action_dim
        )

        # Cache the time horizon and action dimension for the next call
        self.called_time_horizon = self.time_horizon
        self.called_action_dim = self.action_dim

        assert self.time_horizon is not None and self.action_dim is not None, (
            "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."
        )

        decoded_actions = []
        for token in tokens:
            try:
                decoded_tokens = self.fast_tokenizer.bpe_tokenizer.decode(token)
                decoded_dct_coeff = (
                    np.array(list(map(ord, decoded_tokens)))
                    + self.fast_tokenizer.min_token
                )
                if relaxed_decoding:
                    # Expected sequence length
                    expected_seq_len = self.time_horizon * self.action_dim
                    diff = expected_seq_len - decoded_dct_coeff.shape[0]
                    # Apply truncation if too long
                    if diff < 0:
                        decoded_dct_coeff = decoded_dct_coeff[
                            :expected_seq_len
                        ]  # Truncate on the right
                        cprint(
                            f"Relaxed decoding: expected sequence length {expected_seq_len}, got {decoded_dct_coeff.shape[0]}. ",
                            "yellow",
                        )
                    # Apply padding if too short
                    elif diff > 0:
                        decoded_dct_coeff = np.pad(
                            decoded_dct_coeff,
                            (0, diff),
                            mode="constant",
                            constant_values=0,
                        )
                        cprint(
                            f"Relaxed decoding: expected sequence length {expected_seq_len}, got {decoded_dct_coeff.shape[0]}. ",
                            "yellow",
                        )

                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, self.action_dim)
                assert decoded_dct_coeff.shape == (
                    self.time_horizon,
                    self.action_dim,
                ), (
                    f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({self.time_horizon}, {self.action_dim})"
                )
            except Exception as e:
                print(f"Error decoding tokens: {e}")
                print(f"Tokens: {token}")
                decoded_dct_coeff = np.zeros((self.time_horizon, self.action_dim))
            decoded_actions.append(
                idct(
                    decoded_dct_coeff / self.fast_tokenizer.scale, axis=0, norm="ortho"
                )
            )
        return np.stack(decoded_actions)

    def extract_actions(
        self, tokens: torch.Tensor, action_horizon: int, action_dim: int
    ) -> torch.Tensor:
        """
        Extracts actions from predicted output tokens using the FAST model.

        Args:
            tokens (torch.Tensor): The input tensor of tokenized outputs.
            action_horizon (int): The number of timesteps for actions.
            action_dim (int): The dimensionality of each action.

        Returns:
            torch.Tensor: The extracted actions as a tensor of shape (action_horizon, action_dim).
        """
        # Decode predicted output tokens
        decoded_tokens = self.paligemma_tokenizer.batch_decode(
            tokens, skip_special_tokens=True
        )
        cleaned_tokens = [
            tokens_sequence.replace("Action:", "")
            .replace(":", "")
            .strip()
            .split("|")[0]
            .strip()
            for tokens_sequence in decoded_tokens
        ]
        raw_action_tokens = [
            self.processor.tokenizer.encode(
                sample_tokens, return_tensors="pt", padding=False
            )
            for sample_tokens in cleaned_tokens
        ]  # something like this should be robust #looks good
        action_tokens = [
            self._act_tokens_to_paligemma_tokens(raw_action_token)
            for raw_action_token in raw_action_tokens
        ]
        # returns the tensor of decoded actions per sample in a list
        decoded_actions = [
            torch.tensor(
                self.decode_actions_with_fast(
                    tok.tolist(),
                    time_horizon=action_horizon,
                    action_dim=action_dim,
                    relaxed_decoding=self.config.relaxed_action_decoding,
                ),
                device=tokens.device,
            ).squeeze(0)
            for tok in action_tokens
        ]

        return torch.stack(
            decoded_actions,
            dim=0,
        )

    def generate_actions(self, batch: dict[str, Tensor]):
        # normalze, resize, pad, and stack images
        images, img_masks = self.prepare_images(batch)

        # create input tokens from state and prompt
        padded_outs = self.create_input_tokens(
            state=batch["state"], lang_text=batch["prompt"], actions=None
        )

        # embed inputs
        tokens = padded_outs["input_ids"]
        pad_mask = padded_outs["padded_mask"]
        ar_mask = padded_outs["attention_mask"]
        loss_mask = padded_outs["loss_mask"]
        token_type_ids = padded_outs["token_type_ids"]
        embs, pad_masks, att_masks2, targets, loss_mask, token_type_ids = (
            self.embed_inputs(
                images,
                img_masks,
                tokens,
                pad_mask,
                ar_mask,
                loss_mask,
                token_type_ids,
                padding_side="left",
            )
        )

        # generate actions
        token_type_ids = token_type_ids.to(dtype=torch.int64)
        prefix_position_ids = torch.cumsum(pad_masks, dim=1) - 1
        output_tokens = self.pi0_paligemma.generate(
            input_ids=None,
            attention_mask=pad_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=embs,
            use_cache=self.config.use_cache,
            max_new_tokens=self.config.max_decoding_steps,
            do_sample=False,
            num_beams=1,
            token_type_ids=token_type_ids,
        )

        # decode actions from output tokens
        actions = self.extract_actions(
            output_tokens, self.action_horizon, self.action_dim
        )
        return actions

    def embed_image(self, image: torch.Tensor):
        # Handle different transformers versions
        if hasattr(self.pi0_paligemma, "get_image_features"):
            return self.pi0_paligemma.get_image_features(image)
        else:
            return self.pi0_paligemma.model.get_image_features(image)

    def embed_inputs(
        self,
        images,
        img_masks,
        tokens,
        pad_mask,
        ar_mask,
        loss_mask,
        token_type_ids,
        padding_side: str = "right",
    ):
        bsize = images.shape[0]
        device = images.device

        # embed image
        images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        img_emb = self.embed_image(images)
        img_emb = einops.rearrange(img_emb, "(b n) l d -> b (n l) d", b=bsize)
        num_img_embs = img_emb.shape[1]
        img_masks = einops.repeat(img_masks, "b n -> b (n l)", l=num_img_embs // 3)
        img_tgt_tokens = (
            torch.ones_like(img_masks, dtype=torch.long) * self.pad_token_id
        )
        img_loss_mask = torch.zeros_like(img_masks, dtype=torch.long)

        # embed language and state
        tokens_emb = self.embed_tokens(tokens.to(device))
        num_tokens_embs = tokens_emb.shape[1]

        embs = torch.cat([img_emb, tokens_emb], dim=1)
        pad_masks = torch.empty(
            (bsize, num_img_embs + num_tokens_embs), device=device, dtype=torch.bool
        )
        att_masks = torch.zeros(
            (bsize, num_img_embs + num_tokens_embs), device=device, dtype=torch.bool
        )
        loss_masks = torch.empty(
            (bsize, num_img_embs + num_tokens_embs), device=device, dtype=torch.bool
        )

        pad_masks[:, :num_img_embs] = img_masks
        pad_masks[:, num_img_embs:] = pad_mask
        att_masks[:, num_img_embs:] = ar_mask
        loss_masks[:, :num_img_embs] = img_loss_mask
        loss_masks[:, num_img_embs:] = loss_mask

        targets = torch.cat([img_tgt_tokens.to(device), tokens.to(device)], dim=1)
        token_type_ids = torch.cat(
            [img_loss_mask.to(device), token_type_ids.to(device)], dim=1
        )

        # Shift pad tokens to the left (.generate()) or right (.train())
        embs, att_masks, pad_masks, loss_masks, targets, token_type_ids = (
            self.shift_padding_side(
                embs,
                att_masks,
                pad_masks,
                loss_masks,
                targets,
                token_type_ids,
                padding_side=padding_side,
            )
        )

        targets = torch.where(targets == self.pad_token_id, self.ignore_index, targets)
        return embs, pad_masks, att_masks, targets, loss_masks, token_type_ids


def resize_with_pad(img, width, height, pad_value=0, interpolate_like_pi=True):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    if interpolate_like_pi:
        img = (img * 255.0).to(dtype=torch.uint8)
        img = img.permute(0, 2, 3, 1)
        original_device = img.device
        img = img.to(device="cpu").numpy()
        imgs = []
        for sub_img in img:
            sub_img = Image.fromarray(sub_img)
            resized_img = sub_img.resize((resized_width, resized_height), resample=2)
            resized_img = torch.from_numpy(np.array(resized_img))
            imgs.append(resized_img)
        img = torch.stack(imgs, dim=0)
        img = img.permute(0, 3, 1, 2)
        resized_img = img.to(device=original_device, dtype=torch.float32) / 255.0
    else:
        resized_img = F.interpolate(
            img,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img
