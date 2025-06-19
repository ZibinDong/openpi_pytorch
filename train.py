import pytorch_lightning as L
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import ColorJitter, Compose, RandomCrop, Resize

from pi0.new_modeling_pi0 import PI0Policy
from utils.normalizers import Normalizer
from utils.schedulers import CosineDecaySchedule


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


class PI0BridgeDataset(Dataset):
    def __init__(
        self,
        repo_id="ZibinDong/bridgedatav2_val",
        root="/mnt/20T/datasets/bridgev2/lerobot/ZibinDong/bridgedatav2_val",
    ):
        image_transforms = Compose(
            [RandomCrop(243), Resize(224), ColorJitter(0.3, 0.4, 0.5)]
        )
        delta_timestamps = {
            "image": [0],
            "state": [0],
            "action": [i / 5 for i in range(50)],
        }
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
        )

        self.normalizer = Normalizer(
            norm_stats=self.dataset.meta.stats,
            norm_type={"image": "identity", "state": "meanstd", "action": "meanstd"},
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        normalized_item = self.normalizer.normalize(item)
        image = (normalized_item["image"] * 255).to(torch.uint8)
        return {
            "image": {"base_0_rgb": image},
            "state": normalized_item["state"][0],
            "action": normalized_item["action"],
            "action_is_pad": normalized_item["action_is_pad"],
            "prompt": item["task"],
        }


class LightningTrainingWrapper(L.LightningModule):
    def __init__(self, policy: PI0Policy):
        super().__init__()
        self.policy = policy

        self.lr_scheduler = CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        )

    def forward(self, batch):
        return self.policy(batch)[0]

    def training_step(self, batch, batch_idx):
        loss = self.policy(batch)[0]
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.policy.get_optim_params(), lr=5e-5, weight_decay=1e-2
        )
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: self.lr_scheduler(step))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


config = PreTrainedConfig.from_pretrained(
    "/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch"
)
config.device = "cpu"
config.freeze_vision_encoder = True
config.train_expert_only = True
config.train_state_proj = True
policy = PI0Policy(config)
training_policy = LightningTrainingWrapper(policy)

dataset = PI0BridgeDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

trainer = L.Trainer(
    devices=[0, 1, 2, 3],
    strategy="ddp_find_unused_parameters_true",
    max_epochs=1,
    enable_progress_bar=True,
    gradient_clip_val=1.0,
    precision="bf16-true",
    accumulate_grad_batches=2,
)

trainer.fit(training_policy, dataloader)
