import einops
import numpy as np
import torch
import torch.nn.functional as F
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pretrained import PreTrainedPolicy
from torch import Tensor, nn
from transformers import AutoTokenizer

from .paligemma_with_expert import PaliGemmaWithExpertConfig, PaliGemmaWithExpertModel
from .utils import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    resize_with_pad,
    sample_beta,
)

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


class PI0Policy(PreTrainedPolicy):
    """Wrapper class around PI0FlowMatching model to train and run inference within LeRobot."""

    config_class = PI0Config
    name = "torch_pi0"

    def __init__(
        self,
        config: PI0Config,
        tokenizer_path: str = "/home/dzb/pretrained/paligemma3b",
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        self.config = config
        self.language_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = PI0FlowMatching(config)
        self.reset()

    def reset(self):
        return None

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad
    def select_action(
        self, observation: dict[str, Tensor], noise: Tensor | None = None
    ):
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

        images, img_masks = self.prepare_images(observation)
        state = self.prepare_state(observation)
        lang_tokens, lang_masks = self.prepare_language(observation)
        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )
        return actions

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass to compute the loss

        batch: {
            "image": {
                "base_0_rgb": (*b, c, h, w),  # uint8 [0, 255]
                ...
            },
            "state": float32 [*b, s],
            "lang_tokens": float32 [*b, l],
            "lang_masks": float32 [*b, l],
            "action": float32 [*b, ha, da]
        }
        """
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions, action_dim = self.prepare_action(batch)
        noise = batch.get("noise", None)
        time = batch.get("time", None)

        loss_dict = {}
        losses = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )

        actions_is_pad = batch.get("action_is_pad", None)
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, :action_dim]
        loss_dict["losses"] = losses.clone()

        # For backward pass
        loss = losses.mean()
        # For logging
        loss_dict["l2_loss"] = loss.item()

        return loss, loss_dict

    def prepare_images(self, observation: dict[str, Tensor]):
        """Normalize, resize, and pad images and stack them into a tensor.

        Args:
            observation (dict[str, Tensor])

        Returns:
            images (torch.Tensor): (*b, n, c, h, w) images in range [-1.0, 1.0]
            img_masks (torch.Tensor): (*b, n) masks for images, True if image is present, False if missing
        """
        dtype = observation["state"].dtype
        bsize = observation["state"].shape[0]
        images, img_masks = [], []
        present_img_keys = [key for key in IMAGE_KEYS if key in observation["image"]]
        missing_img_keys = [key for key in IMAGE_KEYS if key not in present_img_keys]

        for key in present_img_keys:
            # resize, pad, and normalize
            img = observation["image"][key]
            img = img.to(dtype) / 127.5 - 1.0
            img = resize_with_pad(
                img, *self.config.resize_imgs_with_padding, pad_value=-1.0
            )
            images.append(img)
            img_masks.append(torch.ones((bsize,), dtype=torch.bool, device=img.device))

        for key in missing_img_keys:
            # zero padding
            img = torch.full_like(img, fill_value=-1.0)
            images.append(img)
            img_masks.append(torch.zeros((bsize,), dtype=torch.bool, device=img.device))

        images = torch.stack(images, dim=1)  # (*b, n, c, h, w)
        img_masks = torch.stack(img_masks, dim=1)  # (*b, n)

        return images, img_masks

    def prepare_state(self, observation: dict[str, Tensor]):
        """Pad the state to the maximum state dimension.

        Args:
            observation (dict[str, Tensor])

        Returns:
            state (torch.Tensor): (*b, max_state_dim) padded state tensor
        """
        state = observation["state"]
        state = F.pad(state, (0, self.config.max_state_dim - state.shape[1]))
        return state

    def prepare_action(self, observation: dict[str, Tensor]):
        """Pad the action to the maximum action dimension.

        Args:
            observation (dict[str, Tensor])

        Returns:
            action (torch.Tensor): (*b, n, max_action_dim) padded action tensor
            action_dim (int): the actual dimension of the action before padding
        """
        action = observation["action"]
        action_dim = action.shape[-1]
        action = F.pad(action, (0, self.config.max_action_dim - action_dim))
        return action, action_dim

    def prepare_language(self, observation: dict[str, Tensor]):
        """If `prompt` is provided, modify it to PaliGemma format and tokenize it.
        If `lang_tokens` and `lang_masks` are provided, use them directly.

        PaliGemma expects prefix prompts to be formatted as:
        <images> .... <images> <bos> prompt <sep>, where <sep> uses `\\n`.
        So here we format the prompt to start with `<bos>` and end with `\\n`.
        Later, we will concatenate the images and language tokens into a single sequence.

        Args:
            observation (dict[str, Tensor])

        Returns:
            lang_tokens (torch.Tensor): (*b, l) language tokens
            lang_masks (torch.Tensor): (*b, l) masks for language tokens, True if token is present, False if missing
        """
        lang_tokens = observation.get("lang_tokens", None)
        lang_masks = observation.get("lang_masks", None)
        prompt = observation.get("prompt", None)

        # either provide `prompt` or (`lang_tokens`, `lang_masks`)
        if prompt is None and (lang_tokens is None or lang_masks is None):
            raise ValueError(
                "Either 'prompt' or ('lang_tokens', 'lang_masks') must be provided in the observation."
            )

        device = observation["state"].device
        if prompt is not None and (lang_tokens is None or lang_masks is None):
            prompt = [p if p.startswith("<bos>") else f"<bos>{p}" for p in prompt]
            prompt = [p if p.endswith("\n") else f"{p}\n" for p in prompt]
            tokenized_prompt = self.language_tokenizer.__call__(
                prompt,
                padding="max_length",
                padding_side="right",
                max_length=self.config.tokenizer_max_length,
                return_tensors="pt",
            )
            lang_tokens = tokenized_prompt["input_ids"].to(device=device)
            lang_masks = tokenized_prompt["attention_mask"].to(
                device=device, dtype=torch.bool
            )
        else:
            lang_tokens = observation["lang_tokens"].to(device=device)
            lang_masks = observation["lang_masks"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks


class PI0FlowMatching(nn.Module):
    """
    π0: A Vision-Language-Action Flow Model for General Robot Control

    [Paper](https://www.physicalintelligence.company/download/pi0.pdf)
    [Jax code](https://github.com/Physical-Intelligence/openpi)

    Designed by Physical Intelligence. Ported from Jax by Hugging Face.
    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │Gemma │        │
    │  ┌──────────►│Expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x 10  │        │
    │ │         │  └▲──▲──┘        │
    │ │PaliGemma│   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘

    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # paligemma with action expert
        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_with_export_config
        )

        # projection layers
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(
            self.config.max_action_dim, self.config.proj_width
        )
        self.action_out_proj = nn.Linear(
            self.config.proj_width, self.config.max_action_dim
        )

        self.action_time_mlp_in = nn.Linear(
            self.config.proj_width * 2, self.config.proj_width
        )
        self.action_time_mlp_out = nn.Linear(
            self.config.proj_width, self.config.proj_width
        )

        self.set_requires_grad()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.

        Args:
            images (torch.Tensor):    float (*b, n, c, h, w) images in range [-1.0, 1.0]
            img_masks (torch.Tensor):  bool (*b, n) masks for images
            lang_tokens (torch.Tensor): int (*b, l) language tokens
            lang_masks (torch.Tensor): bool (*b, l) masks for language tokens
        """
        bsize = images.shape[0]
        device = images.device
        dtype = images.dtype

        # embed image
        images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        img_emb = self.paligemma_with_expert.embed_image(images)
        num_patch = img_emb.shape[1]
        img_emb = einops.rearrange(img_emb, "(b n) l d -> b (n l) d", b=bsize)
        img_emb = img_emb.to(dtype=dtype) * (img_emb.shape[-1] ** 0.5)
        num_img_embs = img_emb.shape[1]
        img_masks = einops.repeat(img_masks, "b n -> b (n l)", l=num_patch)

        # embed language
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        num_lang_embs = lang_emb.shape[1]
        lang_emb = lang_emb.to(dtype=dtype) * np.sqrt(lang_emb.shape[-1])

        # assemble embeddings
        embs = torch.cat([img_emb, lang_emb], dim=1)
        pad_masks = torch.cat([img_masks, lang_masks], dim=1)

        # PaliGemma uses bidirectional attention for prefix tokens,
        # so we set 1D `att_masks` to zeros.
        # (see `make_att_2d_masks` to understand why zeros means bidirection)
        att_masks = torch.zeros(
            (bsize, num_img_embs + num_lang_embs), device=device, dtype=torch.bool
        )
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing.

        Args:
            state (torch.Tensor):         float32 (*b, s) robot state
            noisy_actions (torch.Tensor): float32 (*b, n, m) noisy actions
            timestep (torch.Tensor):      float32 (*b,) timestep in [0, 1] range
        """
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        # embed state
        state_emb = self.state_proj(state)

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.config.proj_width,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = einops.repeat(time_emb, "b d -> b n d", n=action_emb.shape[1])
        action_time_emb = torch.cat([action_emb, time_emb], dim=-1)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        action_time_dim = action_time_emb.shape[1]

        # Add to input tokens
        embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
        pad_masks = torch.ones(
            (bsize, action_time_dim + 1), device=device, dtype=torch.bool
        )

        # Set attention masks for suffix tokens so that prefix tokens cannot attend to suffix tokens.
        # And state token cannot attend action tokens.
        # Action tokens use a bidirectional attention.
        att_masks = torch.zeros(
            (bsize, action_time_dim + 1), device=device, dtype=torch.bool
        )
        att_masks[:, :2] = True

        return embs, pad_masks, att_masks

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
    ) -> Tensor:
        bsize = state.shape[0]
        dtype = state.dtype
        device = state.device

        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            actions_shape = (
                bsize,
                self.config.n_action_steps,
                self.config.max_action_dim,
            )
            noise = torch.randn(actions_shape, device=device, dtype=dtype)

        if time is None:
            time = self.sample_time(bsize, device).to(dtype)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, time
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            actions_shape = (
                bsize,
                self.config.n_action_steps,
                self.config.max_action_dim,
            )
            noise = torch.randn(actions_shape, device=device, dtype=dtype)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=dtype, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            v_t = self.predict_velocity(
                state, prefix_pad_masks, past_key_values, x_t, expanded_time
            )

            # Euler step
            x_t += dt * v_t
            time += dt

        return x_t

    def predict_velocity(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        """predict velocity at time t using the suffix model."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, timestep
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
