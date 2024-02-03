# Stronger-CenterNet
This project is the official implementation of the paper 'Scale-balanced real-time object detection with varying input-image resolution' on IEEE Transactions on Circuits and Systems for Video Technology.
## Main libs
```
mmcv==0.2.10
torch==1.1.0
torchvision==0.3.0
cuda==10.1
```
## Install
```
cd stronger-centernet
python setup.py develop
```
## Training

download the COCO2017 dataset and set the folders as follows:
```
stronger-centerNet
----data
--------coco
------------annotations
----------------instances_train2017.json
----------------instances_val2017.json
------------train2017
----------------000000169766.jpg
----------------...
------------val2017
----------------000000581781.jpg
----------------...
```
training with 8 GPUs:
```
./tools/dist_train.sh ./configs/stronger_centernet/stronger_centernet_resnet18_10x_8GPU.py 8
```
## Evaluation
```
./tools/dist_test.sh ./configs/stronger_centernet/stronger_centernet_resnet18_10x_8GPU.py ./work_dirs/stronger_centernet_resnet18_10x/stronger_centernet_resnet18_10x_c4518ea4.pth 1
```
## Main results
The FPS is measured on V100 GPU with batchsize = 1, float32 mode. ^+ means model with AFFM and DDH.
backbone  | training size  | test size | AP@[0.5,...,0.95] | FPS 
 ---- | ----- | ------  | ----- | -----
ResNet-18  | 768x768 | 768x768 | 39.7 |90.5
ResNet-18^+  | 768x768 | 768x768 | 41.0 |55
ResNet-50  | 768x768 | 768x768 | 43.2 |46
ResNet-50^+  | 768x768 | 768x768 | 44.5 |38
Darknet-53  | 768x768 | 768x768 | 44.7 |42.6
Darknet-53^+  | 768x768 | 768x768 | 45.6 |35.8
## Acknowledgement
This project is mainly implemented based on [ttfnet](https://github.com/ZJULearning/ttfnet), [mmdetection](https://github.com/open-mmlab/mmdetection), [CenterNet](https://github.com/xingyizhou/CenterNet), etc. Many Thanks for these repos.
## Citations
If you use our work in your researches, please cite our paper as follow:
```
@article{yan2022scale,
  title={Scale-balanced real-time object detection with varying input-image resolution},
  author={Yan, Longbin and Qin, Yunxiao and Chen, Jie},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={33},
  number={1},
  pages={242--256},
  year={2022},
  publisher={IEEE}
}
```
