<h1 align="center"> PhysHSI: Towards a Real-World Generalizable and Natural Humanoid-Scene Interaction System </h1>

<p align="center">
  <a href='https://why618188.github.io/' target='_blank'>Huayi Wang*</a>,
  <a href='https://zwt006.github.io/' target='_blank'>Wentao Zhang*</a>,
  <a href='https://ingrid789.github.io/IngridYu/' target='_blank'>Runyi Yu*</a>,
  <a href="https://taohuang13.github.io/">Tao Huang</a>,
  <a href="https://renjunli99.github.io/">Junli Ren</a>,
  <a href="https://trap-1.github.io/">Feiyu Jia</a>,
  <a href="https://openreview.net/profile?id=%7EZiRui_Wang4">Zirui Wang</a>,
  <br>
  <a href="https://why618188.github.io/physhsi/">Xiaojie Niu</a>,
  <a href="https://xiao-chen.tech/">Xiao Chen</a>,
  <a href="https://jiahe-chen.cn/">Jiahe Chen</a>,
  <a href="https://cqf.io/">Qifeng Chen<sup>&dagger;</sup></a>,
  <a href="https://wangjingbo1219.github.io/">Jingbo Wang<sup>&dagger;</sup></a>,
  <a href='https://oceanpang.github.io/' target='_blank'>Jiangmiao Pang<sup>&dagger;</sup></a>
  <br>
  *Equal Contributions&nbsp;&nbsp;&nbsp;&nbsp;<sup>&dagger;</sup>Corresponding Authors
  <br>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2502.08378"><img src="https://img.shields.io/badge/arXiv-2502.08378-brown" alt="arXiv"></a>
  <a href="https://youtu.be/dTj6FjoQ5u0"><img src="https://img.shields.io/badge/Youtube-🎬-yellow" alt="YouTube"></a>
  <a href="https://why618188.github.io/physhsi/"><img src="https://img.shields.io/badge/Website-%F0%9F%9A%80-green" alt="Website"></a>
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="License: CC BY-NC-SA 4.0"></a>
</p>

<p align="center">
  <img src="teaser.png" alt="Project teaser" width="100%">
</p>

## 🛠️ Installation

1. Create a conda environment and install PyTorch:

```bash
conda create -n physhsi python=3.8
conda activate physhsi
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

2. Install Isaac Gym:
- Download and install Isaac Gym Preview 4 from [NVIDIA Developer](https://developer.nvidia.com/isaac-gym).
- Navigate to its Python folder and install.
  ```bash
  cd isaacgym/python && pip install -e .
  ```

3. Clone this repository.
```bash
git clone https://github.com/InternRobotics/PhysHSI.git
cd PhysHSI
```

4. Install PhysHSI.
```bash
cd rsl_rl && pip install -e .
cd ../legged_gym && pip install -e .
```

5. Install additional dependencies.
```bash
cd .. && pip install -r requirements.txt
```

## 🕹️ Run PhysHSI

PhysHSI supports six tasks for the Unitree G1 humanoid robot: **CarryBox, SitDown, LieDown, StandUp, StyleLoco-Dinosaur,** and **StyleLoco-Highknee**.

### Motion Visualization

Reference motion data for each task can be found in the [motion data folder](legged_gym/resources/dataset/). To visualize reference motion data, run:

```bash
cd legged_gym
python legged_gym/scripts/play.py --task [task_name] --play_dataset
```

Here, `[task_name]` can be one of `[carrybox, liedown, sitdown, standup, styleloco_dinosaur, styleloco_highknee]`.

### Play with Pre-trained Checkpoints

Pre-trained checkpoints for each task are available in the [checkpoint folder](legged_gym/resources/ckpt/). To play a task using a checkpoint, run:

```bash
python legged_gym/scripts/play.py --task [task_name] --resume_path resources/ckpt/[task_name].pt
```

For example, to play the CarryBox task:
```bash
python legged_gym/scripts/play.py --task carrybox --resume_path resources/ckpt/carrybox.pt
```

> ⚠️ Note:
> 
> During the first 1–2 episodes of `play.py`, you may observe slight interpenetration between the robot, the object, or the platform.
> 
> This issue only occurs in the initial episodes and does not affect training or subsequent performance.

## 🤖 Train PhysHSI

### CarryBox

CarryBox is a challenging long-horizon task. We train it in two steps:

1. **Initial training:** Use a relatively small AMP coefficient and relaxed termination conditions for easier learning. Run approximately 20k steps:
    ```bash
    python legged_gym/scripts/train.py --task carrybox --headless 
    ```

2. **Refined training**: To better align with the data, manually increase the AMP coefficient and continue training for about 30k steps:
    ```bash
    python legged_gym/scripts/train.py --task carrybox_resume --resume --resume_path [ckpt_path] --headless
    ```
    Here, `[ckpt_path]` refers to the checkpoint from the first 20k-step training stage.

To play the final trained checkpoint:
```bash
python legged_gym/scripts/play.py --task carrybox --resume_path [ckpt_path]
```

### Other Tasks

For the remaining five tasks, you can directly train them using:
```bash
python legged_gym/scripts/train.py --task [task_name] --headless
```
Here, `[task_name]` can be one of `[liedown, sitdown, standup, styleloco_dinosaur, styleloco_highknee]`.

To play the final trained checkpoint for any task:
```bash
python legged_gym/scripts/play.py --task [task_name] --resume_path [ckpt_path]
```

> By default, PhysHSI uses **TensorBoard** for logging training metrics.  
> 
> If you prefer to use **Weights & Biases (wandb)**, please enable it in the corresponding `[task_name]_config.py` file and set the appropriate `wandb_entity` for your account.

## 👏 Acknowledgements

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [Legged_gym](https://github.com/leggedrobotics/rsl_rl) and [HIMLoco](https://github.com/OpenRobotLab/HIMLoco): The foundation environments for training and running codes.
- [RSL_RL](https://github.com/leggedrobotics/rsl_rl): Reinforcement learning algorithm implementation.
- [AMP for Hardware](https://github.com/escontra/AMP_for_hardware) and [TokenHSI](https://github.com/liangpan99/TokenHSI): References for AMP and RSI implementations.
- [AMASS](https://amass.is.tue.mpg.de/), [SAMP](https://samp.is.tue.mpg.de/) and [100STYLE](https://www.ianxmason.com/100style/): Reference dataset construction.

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@article{wang2025physhsi,
  title   = {PhysHSI: Towards a Real-World Generalizable and Natural Humanoid-Scene Interaction System},
  author  = {Wang, Huayi and Zhang, Wentao and Yu, Runyi and Huang, Tao and Ren, Junli and Jia, Feiyu and Wang, Zirui and Niu, Xiaojie and Chen, Xiao and Chen, Jiahe and Chen, Qifeng and Wang, Jingbo and Pang, Jiangmiao},
  journal = {arXiv preprint arXiv:xxxx.xxxxx},
  year    = {2025},
}
```

## 📄 License

The PhysHSI code is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0 International License</a> <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>.
Commercial use is not allowed without explicit authorization.
