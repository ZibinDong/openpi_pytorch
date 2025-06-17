from dataclasses import dataclass

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.configs.policies import PreTrainedConfig

"""
delete list

- n_action_steps: int = 5
- normalization_mapping: dict[str, NormalizationMode]
- max_state_dim: int = 32  # pi0fast does not need this
- empty_cameras: int = 0
- adapt_to_pi_aloha: bool = False
- use_delta_joint_actions_aloha: bool = False
- __post_init__, device check
"""


@PreTrainedConfig.register_subclass("torch_pi0fast")
@dataclass
class TorchPI0FASTConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 10

    # Shorter state and action vectors will be padded
    max_action_dim: int = 7  # 7

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (224, 224)
    interpolate_like_pi: bool = False  # ? dont know what this does

    # Tokenizer
    tokenizer_max_length: int = 48

    # Projector
    proj_width: int = 1024

    # Decoding
    max_decoding_steps: int = 256
    fast_skip_tokens: int = (
        128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens
    )
    max_input_seq_len: int = 256  # 512

    # Utils
    use_cache: bool = True

    # Frozen parameters
    freeze_vision_encoder: bool = True
    freeze_lm_head: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    checkpoint_path: str = None

    padding_side: str = "right"

    precision: str = "bfloat16"
    grad_clip_norm: float = 1

    # Allows padding/truncation of generated action tokens during detokenization to ensure decoding.
    # In the original version, tensors of 0s were generated if shapes didn't match for stable decoding.
    relaxed_action_decoding: bool = True

    def __post_init__(self):
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
