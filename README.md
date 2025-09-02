# Simplified LeRobot's Pytorch PI0 & PI0-fast Implementation

[The LeRobot team](https://github.com/huggingface/lerobot/tree/main) has made a substantial contribution to the community through their diligent efforts in converting the PI0 and PI0-fast VLA models to PyTorch. This was an impressive undertaking. However, the original release included only limited usage instructions and examples, making it challenging for users to get the models running correctly by simply following the provided guidance.

This repository addresses those issues by introducing numerous fixes and removing redundant code and functionalities. Furthermore, it now includes comprehensive usage documentation, enabling users to seamlessly deploy official [OpenPI](https://github.com/Physical-Intelligence/openpi/tree/main?tab=readme-ov-file) checkpoints and fine-tune their own models with ease.

## 1. Installation

If you only need to use the VLA models, you'll just need to install [LeRobot](https://github.com/huggingface/lerobot/tree/main) and [PyTorch](https://pytorch.org/). If you plan to run Libero's test scripts (not necessary for VLA), you'll also need to install [CleanDiffuser's Libero support](https://github.com/CleanDiffuserTeam/CleanDiffuser/tree/lightning/cleandiffuser/env/libero).

---

### 1.1 You need to create TWO environment.
- The first one is for downloading JAX model and converting it to pytorch one
- The second one is for traning, evaluating the model using pytorch

Now, let's begin the installation. As it's very complex, you need to do it step by step.

### 1.2 Installing the first envrionment

**1) Create a virtual envrionment**

You can use conda or mamba to create an environment
```
mamba create -n pi0_jax python=3.11
conda activate pi0_jax
```

Than, install the uv on it.
```
mamba install uv
```

**2) Install the Openpi**

Firstly, download it 
```
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
```

Than, use `uv` to install the package
```
GIT_LFS_SKIP_SMUDGE=1 uv pip install . --system
```

Now, the first environment is successfully installed.


### 1.3 Installing the second envrionment

**1) Create a virtual envrionment**

You can use conda or mamba to create an environment
```
mamba create -n pi0_torch python=3.10
conda activate pi0_torch
```
Than, install the `uv`, `ipython` and `ffmpeg` on it.
```
mamba install ffmpeg=7.1.1 uv ipython
```

**2) Install the basis package**

Firstly, use `uv lock` to solve the environment.

Secondly, you need to generate the `requirements.txt`.
```
uv export --format requirements-txt > requirements.txt
```

Finally, install all the package
```
uv pip install --system -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match
```

> This environment contains all the basic package used by `CleanDiffuser`, `lerobot`, `LIBERO` and `openpi_pytorch`.


**3) Install `Cleandiffuser`**

Firstly, download it by `git`. Be careful, the `lightning` branch is the only branch we can used.

```
git clone -b lightning git@github.com:CleanDiffuserTeam/CleanDiffuser.git
```

Then, go into the `Cleandiffuser` folder
```
cd Cleandiffuser
```

Then, edit the `pyproject.toml`, delete all the dependences in the file
```
dependencies = [

]
```

> We have already installed it in the Step2, delete it and make sure the `pip` will not change the version of basis package.

Now, install it
```
pip install  .    
```


**4) Install `lerobot`**

Firstly, download it

```
git clone https://github.com/huggingface/lerobot.git 
```

Then, go into the folder and edit the `pyproject.toml`, and delete all the dependences.

Then, install it
```
pip install  .    
```


**5) Install `LIBERO`**

Firstly, download it

```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git 
```

Then, go into the folder and edit the `requirements.txt`, and delete all the lines.

Then, install it
```
pip install  .    
```

Now, all the packages have been successfully installed.





## 2. Usage

### 2.1. Converting OpenPI Checkpoints

> In this step, you need to use `pi0_jax` environment

You can directly use the checkpoints LeRobot has uploaded to [HuggingFace](https://huggingface.co/lerobot/pi0):

```python
from pi0 import Pi0Policy
policy = Pi0Policy.from_pretrained("lerobot/pi0")
```

LeRobot has only uploaded the `pi0_base` model. However, OpenPI provides a [**list of checkpoints**](https://github.com/Physical-Intelligence/openpi?tab=readme-ov-file#model-checkpoints) for inference or fine-tuning, so I highly recommend using the conversion script to **flexibly obtain various OpenPI checkpoints**.

First, you'll need to install [OpenPI](https://github.com/Physical-Intelligence/openpi/tree/main?tab=readme-ov-file) and download an official JAX checkpoint. Let's take `pi0_libero` as an example:

```python
from openpi.shared import download
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero", anon=True)
```

This will store the downloaded checkpoint in `"/home/username/.cache/openpi/openpi-assets/checkpoints/pi0_libero"` if you're using Ubuntu. Then, you can run the conversion script by simply providing the JAX checkpoint path and the desired PyTorch checkpoint path:

```bash
python convert_pi0_to_hf_lerobot.py \
    --checkpoint_dir /home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_libero/params \
    --output_path /home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch
```

**Note:** After completing this step, **do not delete the JAX checkpoint**. This folder contains crucial `norm_stats` parameters, which are essential if you plan to use the model for inference.

### 2.2. Try Inference Code

> In this step, you need to use `pi0_torch` environment

Please see `1_e2e_inference.py`.