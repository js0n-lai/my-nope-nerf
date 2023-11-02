# Depth-Guided Neural Scene Reconstruction for Autonomous Vehicles
**[Data](https://drive.google.com/drive/folders/1JZ6EH5a1oL-0YeOxjlfeybh-einl_1On?usp=sharing) | [Pretrained Models](https://drive.google.com/drive/folders/1FnjRGaPmpPhLi-wNK3sWmvlDSJtWYUST?usp=sharing) | [PDF PENDING]**

Jason Lai, supervised by Viorela Ila

University of Sydney


## Table of Contents
- [Overview](#Overview)
- [Installation](#Installation)
- [Data](#Data)
- [Usage](#Usage)
- [Acknowledgement](#Acknowledgement)

## Overview
This is the codebase for my undergraduate honours thesis. We show depth-supervision (even with sparse priors) can improve the scene appearance and geometry constructed by NeRFs. We used [Virtual KITTI](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/) as a proxy for data collected by autonomous vehicles. Our implementation is built on top of [NoPe-NeRF: Optimising Neural Radiance Field with No Pose Prior](https://github.com/ActiveVisionLab/nope-nerf/).

## Installation

```
git clone https://github.com/ActiveVisionLab/nope-nerf.git
cd nope-nerf
conda env create -f environment.yaml
conda activate nope-nerf
```

## Data and Preprocessing
1. [Virtual KITTI Subset](https://drive.google.com/drive/folders/1JZ6EH5a1oL-0YeOxjlfeybh-einl_1On?usp=sharing):
Our preprocessed Virtual KITTI data contains the 3 scenes shown in the paper. Each scene contains images, monocular depth estimations from DPT and poses. There are two variants for each scene - one with Virtual KITTI poses and another with COLMAP poses. Numbering of each scene follows the Appendix in the pdf. You can download and unzip it to the `data` directory.

2. If you want to use your own Virtual KITTI sequence, visit [Virtual KITTI](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/) to acquire the dataset and extract all archives to the same root. You can then use the provided `get_kittivirtual.py` script to acquire the desired sequence and generate config files in `configs/V_KITTI`. Usage can be determined by running
```
python get_kittivirtual.py --help
```
All config settings inherit from `configs/default.yaml` and `configs/preprocess.yaml` if not explicitly given in the generated `YAML` files.

3. Monocular depth map generation: you can first download the pretrained DPT model from [this link](https://drive.google.com/file/d/1dgcJEYYw1F8qirXhZxgNK8dWWz_8gZBD/view?usp=sharing) provided by [Vision Transformers for Dense Prediction](https://github.com/isl-org/DPT) to `DPT` directory, then run
```
python preprocess/dpt_depth.py configs/V_KITTI/<preprocess_X.yaml>
```
This generates monocular depth maps under `dpt/` in the scene directory. Make sure the `cfg['dataloading']['path']` and `cfg['dataloading']['scene']` in `configs/preprocess.yaml` are set to your own image sequence. To use DPT depths for depth supervision, make sure `cfg['dataloading']['with_depth']` is `False` and the distortion parameters are enabled by setting `cfg['distortion']['learn_shift']`, `cfg['distortion']['learn_scale']` to `True`. Use the opposite settings if you wish to use Virtual KITTI depths for depth supervision instead.

## Training

1. Train a new model from scratch:

```
python train.py configs/V_KITTI/<X>.yaml
```
where you can replace `<X>` with other config files.

A good sanity check to perform is whether the poses were loaded correctly. By default, this is set to `False` in `configs/default.yaml` under `cfg['dataloading']['show_pose_only']`. Setting this to `True` allows the preprocessed poses to be visualised before terminating the program. Take particular note of the camera orientations - we intentionally use asymmetric frustums to remove any ambiguity (see `utils_poses/vis_cam_traj.py` for more details). The individual preprocessing steps can also be visualised by setting `visualise=True` in the calls to `self.make_c2ws_from_llff()`.

You can monitor the training process on <http://localhost:6006> using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
tensorboard --logdir ./out --port 6006
```
Or for multiple runs:
```
tensorboard --logdir_spec <name1>:out/V_KITTI/<X1>/logs,<name2>:out/V_KITTI/<X2>/logs, ...
```

For available training options, please take a look at `configs/default.yaml` or the provided Virtual KITTI examples.

2. [Pretrained Models](https://drive.google.com/drive/folders/1FnjRGaPmpPhLi-wNK3sWmvlDSJtWYUST?usp=sharing)
Our pretrained models are numbered according to the sample Virtual KITTI config files. Simply download the zip and extract the contents into the `out/` directory.

## Evaluation
1. Evaluate image quality and depth:
```
python evaluation/eval.py configs/V_KITTI/<X>.yaml
```
To evaluate depths, we compute the estimated ray termination range in model space, then convert this to a z-depth. The z-depths are then rescaled to the metric system by undoing the depth preprocessing.

2. Evaluate poses:
```
python evaluation/eval_poses.py configs/V_KITTI/<X>.yaml --vis
```
This will provide `Open3D` visualisations. To use `Matplotlib`, set `use_matplotlib` to `True` in `evaluation/eval_poses.py`. For evaluation, we undo the pose preprocessing to recover the original metric scale. We align the refined poses with Virtual KITTI poses by applying a rigid transformation that aligns the initial poses in each trajectory. We then propagate this transform to the rest of the refined trajectory.

## More Visualisations
Novel view synthesis - render new frames
```
python vis/render.py configs/V_KITTI/<X>.yaml
```
The config parameters under `cfg['extract_images']` can also be modified to change the output resolution and type of camera trajectory to render novel views along.

Pose visualisation (estimated trajectory only)
```
python vis/vis_poses.py configs/V_KITTI/<X>.yaml
```

## Acknowledgement
We thank Wenjing Bian et. al for their excellent open-source implementation of [NoPe-NeRF](https://github.com/ActiveVisionLab/nope-nerf/) from which we based much of our work on.
We also thank them for their insights about technical issues we faced early on. 
