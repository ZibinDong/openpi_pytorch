import lightning as L
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Resize

from pi0.modeling_pi0 import PI0Policy
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
                "action": "std",
            },
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item["action"] = item["action"] - item["observation.state"]
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
    def __init__(self, config):
        super().__init__()
        self.policy = None
        self.config = config

    def configure_model(self):
        if self.policy is None:
            self.config.device = self.device
            self.policy = PI0Policy(self.config)

    def forward(self, batch):
        return self.policy(batch)[0]

    def training_step(self, batch, batch_idx):
        loss = self.policy(batch)[0]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.policy.get_optim_params(), lr=5e-5, weight_decay=1e-2, eps=1e-6
        )
        scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=100,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


dataset = PI0SO100Dataset()
dataloader = DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=4, persistent_workers=True
)

callback = ModelCheckpoint(
    dirpath="/mnt/20T/dzb/pi0_so100_checkpoints",
    filename="{epoch}-{step}",
    save_top_k=-1,
    every_n_epochs=4,
)

trainer = L.Trainer(
    accelerator="cuda",
    devices=4,
    strategy="ddp_find_unused_parameters_true",
    max_epochs=50,
    enable_progress_bar=True,
    gradient_clip_val=1.0,
    precision="bf16-mixed",
    accumulate_grad_batches=4,
    callbacks=[callback],
)

with trainer.init_module():
    config = PreTrainedConfig.from_pretrained(
        "/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch"
    )
    config.device = "cpu"
    config.freeze_vision_encoder = True
    config.train_expert_only = True
    config.train_state_proj = True
    training_policy = LightningTrainingWrapper(config)


trainer.fit(training_policy, dataloader)
# trainer.fit(
#     training_policy,
#     dataloader,
#     ckpt_path="/mnt/20T/dzb/pi0_so100_checkpoints/epoch=39-step=17236.ckpt",
# )

# training_policy.configure_model()
# training_policy.load_state_dict(
#     torch.load(
#         "/mnt/20T/dzb/pi0_so100_checkpoints/epoch=47-step=20812.ckpt",
#         map_location="cpu",
#     )["state_dict"]
# )

# device = "cuda:1"
# batch = next(iter(dataloader))
# batch = to_device_dtype(batch, "cuda:1", torch.bfloat16)
# policy = training_policy.policy.to("cuda:1", torch.bfloat16).eval()
# action = policy.select_action(batch)
# action = action[:, :, :6]
