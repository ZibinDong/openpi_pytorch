import lightning as L
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from peft import LoraConfig, TaskType, get_peft_model
from termcolor import cprint
from torchvision.transforms.v2 import Resize

from pi0 import PI0FASTPolicy
from utils.normalizers import Normalizer
from utils.server import PolicyServer


def to_device_dtype(d, device, dtype):
    for key, value in d.items():
        if isinstance(value, dict):
            to_device_dtype(value, device, dtype)
        elif isinstance(value, torch.Tensor):
            if key not in ["action_is_pad"]:
                d[key] = value.to(device=device, dtype=dtype)
            else:
                d[key] = value.to(device=device)
        else:
            pass
    return d


class LightningTrainingWrapper(L.LightningModule):
    def __init__(self, config, ckpt_path):
        super().__init__()
        # load model in `configure_model` to accelerate model loading
        self.policy = None
        self.config = config
        self.ckpt_path = ckpt_path

    def configure_model(self):
        if self.policy is None:
            policy = PI0FASTPolicy.from_pretrained(self.ckpt_path, config=self.config)
            # add lora to pi0_paligemma model
            model = policy.model.pi0_paligemma
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                task_type=TaskType.CAUSAL_LM,
                target_modules="all-linear",
            )
            policy.model.pi0_paligemma = get_peft_model(model, peft_config)
            self.policy = policy

    def forward(self, batch):
        return self.policy(batch)[0]

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.policy(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("acc", loss_dict["acc"], prog_bar=True)
        return loss


class SO100Policy:
    def __init__(
        self,
        ckpt_path: str,
        pi0fast_ckpt_path: str,
        repo_ids: list[str] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype

        # load policy
        cprint("Loading SO100 Policy...", "yellow")
        config = PreTrainedConfig.from_pretrained(pi0fast_ckpt_path)
        config.max_action_dim = 6  # set action dimension to 6
        config.chunk_size = 50  # set action chunk length to 50
        training_policy = LightningTrainingWrapper(config, pi0fast_ckpt_path)
        training_policy.configure_model()
        training_policy.load_state_dict(
            torch.load(ckpt_path, map_location=device)["state_dict"]
        )
        self.policy = training_policy.policy.to(device=device, dtype=dtype).eval()
        cprint("SO100 Policy loaded successfully!", "green")

        cprint("Prepareing norm stats...", "yellow")
        dataset = MultiLeRobotDataset(repo_ids)
        self.normalizer = Normalizer(
            norm_stats=dataset.stats,
            norm_type={
                "observation.images.base": "identity",
                "observation.images.wrist": "identity",
                "observation.state": "minmax",
                "action": "minmax",
            },
        )
        print(self.normalizer.norm_stats)
        cprint("Norm stats prepared successfully!", "green")

        self.resize = Resize((224, 224))

        cprint("Ready to use SO100 Policy!", "green")

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        """
        obs: {
            "base": uint8 (H, W, C),
            "wrist": uint8 (H, W, C),
            "state": float32 (state_dim,),
            "prompt": str
        }
        """
        obs = self.normalizer.normalize(
            {
                "observation.images.base": obs["base"],
                "observation.images.wrist": obs["wrist"],
                "observation.state": obs["state"],
                "prompt": obs["prompt"],
            }
        )

        base_image = torch.tensor(
            obs["observation.images.base"], dtype=torch.uint8, device=self.device
        )
        wrist_image = torch.tensor(
            obs["observation.images.wrist"], dtype=torch.uint8, device=self.device
        )
        base_image = base_image.permute(2, 0, 1)[None]
        wrist_image = wrist_image.permute(2, 0, 1)[None]
        base_image = self.resize(base_image)
        wrist_image = self.resize(wrist_image)
        state = torch.tensor(
            obs["observation.state"], dtype=self.dtype, device=self.device
        )[None]
        prompt = obs["prompt"]
        action = self.policy.select_action(
            {
                "image": {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": wrist_image,
                },
                "state": state,
                "prompt": prompt,
            }
        )
        action = action[:, :, :6]
        action = action.float().cpu().numpy()
        action = self.normalizer.unnormalize({"action": action})["action"]
        return action

    def __call__(self, obs: np.ndarray):
        return self.act(obs)


if __name__ == "__main__":
    policy = SO100Policy(
        ckpt_path="/mnt/20T/dzb/pi0fast_so100_checkpoints/epoch=39-step=29760.ckpt",
        pi0fast_ckpt_path="/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_fast_base_pytorch",
        repo_ids=[f"ZibinDong/so100_play_screwdriver_0{i + 1}" for i in range(6)],
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    server = PolicyServer(policy, host="0.0.0.0", port=12346)
    server.run()
