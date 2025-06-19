import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies import PI0Config
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import ColorJitter, Compose, RandomCrop, Resize

from pi0.new_modeling_pi0 import PI0Policy
from utils.normalizers import Normalizer


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


device = "cuda:1"
policy = PI0Policy.from_pretrained(
    "/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch"
)
for params in policy.model.paligemma_with_expert.paligemma.parameters():
    params.requires_grad = False
policy = policy.to(device, torch.bfloat16)
dataset = PI0BridgeDataset()

dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
batch = next(iter(dataloader))
batch = to_device_dtype(batch, device, torch.bfloat16)

loss = policy(batch)
