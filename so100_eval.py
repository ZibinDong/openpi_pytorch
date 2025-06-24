import numpy as np
import pytorch_lightning as L
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig

from termcolor import cprint
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Resize

from pi0.modeling_pi0 import PI0Policy
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


class PI0SO100Dataset(Dataset):
    def __init__(
        self,
        repo_id="ZibinDong/so100_grab_screwdriver",
        root="/mnt/20T/datasets/bridgev2/lerobot/ZibinDong/so100_grab_screwdriver",
    ):
        image_transforms = Resize((224, 224))
        delta_timestamps = {
            "observation.images.base": [0],
            "observation.images.wrist": [0],
            "observation.state": [0],
            "action": [i / 30 for i in range(50)],
        }
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
        )

        self.normalizer = Normalizer(
            norm_stats=self.dataset.meta.stats,
            norm_type={
                "observation.images.base": "identity",
                "observation.images.wrist": "identity",
                "observation.state": "meanstd",
                "action": "meanstd",
            },
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        normalized_item = self.normalizer.normalize(item)
        base_image = (normalized_item["observation.images.base"] * 255).to(torch.uint8)
        wrist_image = (normalized_item["observation.images.wrist"] * 255).to(
            torch.uint8
        )
        return {
            "image": {"base_0_rgb": base_image, "left_wrist_0_rgb": wrist_image},
            "state": normalized_item["observation.state"][0],
            "action": normalized_item["action"],
            "action_is_pad": normalized_item["action_is_pad"],
            "prompt": item["task"],
        }


class LightningTrainingWrapper(L.LightningModule):
    def __init__(self, policy: PI0Policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch):
        return self.policy(batch)[0]

    def training_step(self, batch, batch_idx):
        loss = self.policy(batch)[0]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.policy.get_optim_params(), lr=5e-5, weight_decay=1e-2
        )
        return optimizer


class SO100Policy:
    def __init__(self, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype

        # load policy
        cprint("Loading SO100 Policy...", "yellow")
        config = PreTrainedConfig.from_pretrained(
            "/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch"
        )
        config.device = "cpu"
        config.freeze_vision_encoder = True
        config.train_expert_only = True
        config.train_state_proj = True
        policy = PI0Policy(config)
        training_policy = LightningTrainingWrapper(policy)
        training_policy.load_state_dict(
            torch.load(
                "/mnt/20T/dzb/pi0_so100_checkpoints/epoch=47-step=20812.ckpt",
                map_location="cpu",
            )["state_dict"]
        )
        self.policy = policy.to(device=device, dtype=dtype).eval()
        cprint("SO100 Policy loaded successfully!", "green")

        cprint("Prepareing norm stats...", "yellow")
        dataset = LeRobotDataset(
            repo_id="ZibinDong/so100_grab_screwdriver",
            root="/mnt/20T/datasets/bridgev2/lerobot/ZibinDong/so100_grab_screwdriver",
        )
        self.normalizer = Normalizer(
            norm_stats=dataset.meta.stats,
            norm_type={
                "observation.images.base": "identity",
                "observation.images.wrist": "identity",
                "observation.state": "meanstd",
                "action": "std",
            },
        )
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
        obs = self.normalizer.normalize({
            "observation.images.base": obs["base"],
            "observation.images.wrist": obs["wrist"],
            "observation.state": obs["state"],
            "prompt": obs["prompt"],
        })

        base_image = torch.tensor(obs["observation.images.base"], dtype=torch.uint8, device=self.device)
        wrist_image = torch.tensor(obs["observation.images.wrist"], dtype=torch.uint8, device=self.device)
        base_image = base_image.permute(2, 0, 1)[None]
        wrist_image = wrist_image.permute(2, 0, 1)[None]
        base_image = self.resize(base_image)
        wrist_image = self.resize(wrist_image)
        state = torch.tensor(obs["observation.state"], dtype=self.dtype, device=self.device)[None]
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
        state = state.float().cpu().numpy()
        state_action = self.normalizer.unnormalize({"observation.state": state, "action": action})
        state = state_action["observation.state"]
        action = state_action["action"]
        action = action + state
        return action

    def __call__(self, obs: np.ndarray):
        return self.act(obs)


if __name__ == "__main__":
    policy = SO100Policy(dtype=torch.bfloat16)
    server = PolicyServer(policy, host="0.0.0.0", port=12346)
    server.run()
