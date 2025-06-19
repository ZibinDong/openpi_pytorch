import torch

from pi0 import PI0FASTPolicy, PI0Policy

PATH_TO_PI_MODEL = (
    "/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch"
)
PATH_TO_PI_FAST_MODEL = (
    "/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_fast_libero_pytorch"
)
model_type = "pi0fast"  # or "pi0fast"

# load model
if model_type == "pi0":
    policy = PI0Policy.from_pretrained(PATH_TO_PI_MODEL)
else:
    policy = PI0FASTPolicy.from_pretrained(PATH_TO_PI_FAST_MODEL)

# create pseudo observation
# check the comment in `PI0Policy.select_action` for the expected observation format
# let's assume we have the following observation
device = policy.config.device
observation = {
    "image": {
        "base_0_rgb": torch.randint(
            0, 256, (1, 3, 224, 224), dtype=torch.uint8, device=device
        ),
        # "left_wrist_0_rgb": ...,   Suppose we don't have this view
        # "right_wrist_0_rgb": ...,  Suppose we don't have this view
    },
    "state": torch.randn(1, 8, device=device) * 0.2,
    "prompt": ["do something"],
}

# select action
# let's assume the `action_dim` is 7
action = policy.select_action(observation)[0, :, :7]
print(action)
