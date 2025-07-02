import lightning as L
import torch
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Resize

from pi0 import PI0FASTPolicy
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
    def __init__(self, repo_ids):
        image_transforms = Resize((224, 224))

        # [i / 25 for i in range(50)] represents action chunks in 50 steps at 25 FPS.
        # The timestamps are set to 0 for the images and state, as we only use current obs.
        delta_timestamps = {
            "observation.images.base": [0],
            "observation.images.wrist": [0],
            "observation.state": [0],
            "action": [i / 25 for i in range(50)],
        }
        self.dataset = MultiLeRobotDataset(
            repo_ids=repo_ids,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
        )
        self.normalizer = Normalizer(
            norm_stats=self.dataset.stats,
            norm_type={
                "observation.images.base": "identity",
                "observation.images.wrist": "identity",
                "observation.state": "minmax",
                "action": "minmax",
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


dataset = PI0SO100Dataset(
    [f"ZibinDong/so100_play_screwdriver_0{i + 1}" for i in range(6)]
)
dataloader = DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=2, persistent_workers=True
)

callback = ModelCheckpoint(
    dirpath="/mnt/20T/dzb/pi0fast_so100_checkpoints",  # where you want to save the checkpoints
    filename="{epoch}-{step}",
    save_top_k=-1,  # save all checkpoints
    every_n_epochs=4,  # save all checkpoints
)

trainer = L.Trainer(
    accelerator="cuda",
    devices=4,
    strategy="ddp",
    max_epochs=50,
    enable_progress_bar=True,
    gradient_clip_val=1.0,
    precision="bf16-mixed",
    accumulate_grad_batches=4,
    callbacks=[callback],
)

with trainer.init_module():
    ckpt_path = (
        "/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_fast_base_pytorch"
    )
    config = PreTrainedConfig.from_pretrained(ckpt_path)
    config.max_action_dim = 6  # set action dimension to 6
    config.chunk_size = 50  # set action chunk length to 50
    training_policy = LightningTrainingWrapper(config, ckpt_path)

trainer.fit(training_policy, dataloader)
