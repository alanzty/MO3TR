# MO3TR
This the official implementation of Looking Beyond Two Frames: End-to-End Multi-Object Tracking Using Spatial and Temporal Transformers. 
The code is built on top of MMTRACK and MMDET. Please refer to these two repo for setup. 
Paper Link:https://arxiv.org/abs/2103.14829
## Getting Started
Install the main packages:

 - torch==1.9.1
 - mmcv==1.4.1
 - mmdet==2.19.1
 - mmtrack==0.8.0

Then use the following command to setup MO3TR:
```
bash setup
```
### Training
```
python run/train_nf.py
```
### Tracking
```
python run/track_nf.py
```
You can download the model weight used for MOT17 test here:
https://drive.google.com/file/d/1d-8A2dRNyMLb4kCYa7DMHCDvuiSMxc5K/view?usp=sharing
