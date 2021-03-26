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
Available on Kaggle: [link](https://www.kaggle.com/dataset/1cf520aba05e023f2f80099ef497a8f3668516c39e6f673531e3e47407c46694)

### Download Pascal VOC dataset
Download the preprocessed dataset from [link](https://www.kaggle.com/aladdinpersson/pascal-voc-dataset-used-in-yolov3-video). Just unzip this in the main directory.

### Download MS COCO dataset
Download the preprocessed dataset from [link](https://www.kaggle.com/dataset/79abcc2659dc745fddfba1864438afb2fac3fabaa5f37daa8a51e36466db101e). Just unzip this in the main directory.

### Training
Edit the config.py file to match the setup you want to use. Then run train.py

### Results
| Model                   | mAP @ 50 IoU |
| ----------------------- |:-----------------:|
| YOLOv3 (Pascal VOC) 	  | 78.2              |
| YOLOv3 (MS-COCO)        | Will probably train on this at some point      |

The model was evaluated with confidence 0.2 and IOU threshold 0.45 using NMS.

### Things I'm unsure of
From my understanding YOLOv3 labeled targets to include an anchor on each of the three different scales. This leads to a problem where we will have multiple 
predictions of the same object and I think the idea is that we rely more on NMS. The probability of an object in loss function should correspond to the IOU 
with the ground truth box, this should also alleviate with multiple bounding boxes prediction for each ground truth (since obj score is lower). When loading the 
original weights for YOLOv3 I good mAP results but the object score, no object score seems to be a bit different because the accuracy on those aren't great.
This suggests there's something different with the original implementation, but not sure what it is exactly. The original YOLOv3 paper also used  BCE loss 
for class labels since some datasets are multi-label, however I thought it was more natural to use CrossEntropy because both Pascal and COCO just have a single label. 

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
