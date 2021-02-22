# YOLOv3 in PyTorch
A quite minimal implementation of YOLOv3 in PyTorch spanning only around 800 lines of code related to YOLOv3 (not counting plot image helper functions etc). The repository has support for training and evaluation and complete with helper functions for inference. There is currently pretrained weights for Pascal-VOC with MS COCO coming up. 

## Installation

### Clone and install requirements
```bash
$ git clone https://github.com/aladdinpersson/Machine-Learning-Collection
$ cd ML/Pytorch/object_detection/YOLOv3/
$ pip install requirements.txt
```
### Download pretrained weights on Pascal-VOC
Available on Kaggle: coming soon

### Download Pascal-VOC dataset
Download the preprocessed dataset from [this](www.kaggle.com/aladdinpersson/pascalvoc-yolo-works-with-albumentations). Just unzip this in the main directory.

### Download MS-COCO dataset
Download the preprocessed dataset from [this](www.kaggle.com/aladdinpersson/mscoco-yolo-works-with-albumentations). Just unzip this in the main directory.

### Training
Edit the config.py file to match the setup you want to use. Then run train.py

### Results
| Model                   | mAP @ 50 IoU |
| ----------------------- |:-----------------:|
| YOLOv3 (Pascal VOC) 	  | 78.2              |
| YOLOv3 (MS-COCO)        | Not done yet      |

The model was evaluated with confidence 0.2 and IOU threshold 0.45 using NMS.

## YOLOv3 paper 
The implementation is based on the following paper:

### An Incremental Improvement 
by Joseph Redmon, Ali Farhadi

#### Abstract
We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that’s pretty swell. It’s a little bigger than last time but more accurate. It’s still fast though, don’t worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared to 57.5 AP50 in 198 ms by RetinaNet, similar performance but 3.8× faster. As always, all the code is online at https://pjreddie.com/yolo/.

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```