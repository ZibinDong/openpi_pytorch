# Simplified LeRobot's Pytorch PI0 & PI0-fast Implementation

[The LeRobot team](https://github.com/huggingface/lerobot/tree/main) has made a substantial contribution to the community through their diligent efforts in converting the PI0 and PI0-fast VLA models to PyTorch. This was an impressive undertaking. However, the original release included only limited usage instructions and examples, making it challenging for users to get the models running correctly by simply following the provided guidance.

This repository addresses those issues by introducing numerous fixes and removing redundant code and functionalities. Furthermore, it now includes comprehensive usage documentation, enabling users to seamlessly deploy official [OpenPI](https://github.com/Physical-Intelligence/openpi/tree/main?tab=readme-ov-file) checkpoints and fine-tune their own models with ease.

## 1. Installation

If you only need to use the VLA models, you'll just need to install [LeRobot](https://github.com/huggingface/lerobot/tree/main) and [PyTorch](https://pytorch.org/). If you plan to run Libero's test scripts (not necessary for VLA), you'll also need to install [CleanDiffuser's Libero support](https://github.com/CleanDiffuserTeam/CleanDiffuser/tree/lightning/cleandiffuser/env/libero).

---

## 2. Usage

### 2.1. Converting OpenPI Checkpoints

You can directly use the checkpoints LeRobot has uploaded to [HuggingFace](https://huggingface.co/lerobot/pi0):

```python
from pi0 import Pi0Policy
policy = Pi0Policy.from_pretrained("lerobot/pi0")
```

LeRobot has only uploaded the `pi0_base` model. However, OpenPI provides a [**list of checkpoints**](https://github.com/Physical-Intelligence/openpi?tab=readme-ov-file#model-checkpoints) for inference or fine-tuning, so I highly recommend using the conversion script to **flexibly obtain various OpenPI checkpoints**.

First, you'll need to install [OpenPI](https://github.com/Physical-Intelligence/openpi/tree/main?tab=readme-ov-file) and download an official JAX checkpoint. Let's take `pi0_libero` as an example:

```python
from openpi.shared import download
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
```

This will store the downloaded checkpoint in `"/home/username/.cache/openpi/openpi-assets/checkpoints/pi0_libero"` if you're using Ubuntu. Then, you can run the conversion script by simply providing the JAX checkpoint path and the desired PyTorch checkpoint path:

```bash
python convert_pi0_to_hf_lerobot.py \
    --checkpoint_dir /home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_libero/params \
    --output_path /home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch
```

**Note:** After completing this step, **do not delete the JAX checkpoint**. This folder contains crucial `norm_stats` parameters, which are essential if you plan to use the model for inference.

### 2.2. Try Inference Code

Please see `1_e2e_inference.py`.