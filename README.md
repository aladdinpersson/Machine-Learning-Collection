<p align="center"><img width="100%" src="ML/others/logo/ml_logo.svg" /></p>

--------------------------------------------------------------------------------


[![Build Status](https://travis-ci.com/aladdinpersson/Machine-Learning-Collection.svg?branch=master)](https://travis-ci.com/aladdinpersson/Machine-Learning-Collection) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[logo]: https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/others/logo/ylogo1.png

# Machine Learning Collection
In this repository you will find tutorials and projects related to Machine Learning. I try to make the code as clear as possible, and the goal is be to used as a learning resource and a way to lookup problems to solve specific problems. For most I have also done video explanations on YouTube to make it easier to follow the code. If you got any questions add an issue (or comment on YouTube, I read and answer to most), and if you would like to add an algorithm/improve something please do make a PR! This repository is contribution friendly :smiley:

## Table Of Contents
- [Machine Learning Algorithms](#machine-learning)
- [PyTorch Tutorials](#pytorch-tutorials)
	- [Basics](#basics)
	- [More Advanced](#more-advanced)
    - [Object Detection](#Object-Detection)
	- [Generative Adversarial Networks](#Generative-Adversarial-Networks)
	- [Architectures](#architectures)
- [TensorFlow Tutorials](#tensorflow-tutorials)
	- [Beginner Course](#beginner-course)
	- [Architectures](#CNN-Architectures)

## Machine Learning
* [![Youtube Link][logo]](https://youtu.be/pCCUnoes1Po) &nbsp; [Linear Regression](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/algorithms/linearregression/linear_regression_gradient_descent.py) **- With Gradient Descent** :white_check_mark: 
* [![Youtube Link][logo]](https://youtu.be/DQ6xfe75CDk) &nbsp; [Linear Regression](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/algorithms/linearregression/linear_regression_normal_equation.py) **- With Normal Equation** :white_check_mark:
* [![Youtube Link][logo]](https://youtu.be/x1ez9vi611I) &nbsp; [Logistic Regression](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/algorithms/logisticregression/logistic_regression.py)
* [![Youtube Link][logo]](https://youtu.be/3trW5Lig7BU) &nbsp; [Naive Bayes](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/algorithms/naivebayes/naivebayes.py) **- Gaussian Naive Bayes**
* [![Youtube Link][logo]](https://youtu.be/QzAaRuDskyc) &nbsp; [K-nearest neighbors](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/algorithms/knn/knn.py)
* [![Youtube Link][logo]](https://youtu.be/W4fSRHeafMo) &nbsp; [K-means clustering](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/algorithms/kmeans/kmeansclustering.py) 
* [![Youtube Link][logo]](https://youtu.be/gBTtR0bs-1k) &nbsp; [Support Vector Machine](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/algorithms/svm/svm.py) **- Using CVXOPT**
* [![Youtube Link][logo]](https://youtu.be/NJvojeoTnNM) &nbsp; [Neural Network](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/algorithms/neuralnetwork/NN.py)
* [Decision Tree](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/algorithms/decisiontree/decision_tree.py) **- Decision Tree by Philip Andreadis**

## PyTorch Tutorials
If you have any specific video suggestion please make a comment on YouTube :)

### Basics
* [![Youtube Link][logo]](https://youtu.be/x9JiIFvlUwk) &nbsp; [Tensor Basics](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_tensorbasics.py)
* [![Youtube Link][logo]](https://youtu.be/Jy4wM2X21u0) &nbsp; [Feedforward Neural Network](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/804c45e83b27c59defb12f0ea5117de30fe25289/ML/Pytorch/Basics/pytorch_simple_fullynet.py#L26-L35)
* [![Youtube Link][logo]](https://youtu.be/wnK3uWv_WkU) &nbsp; [Convolutional Neural Network](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/157a5f458f272a513eb6b4a19d6613aec32dc21c/ML/Pytorch/Basics/pytorch_simple_CNN.py#L25-L41)
* [![Youtube Link][logo]](https://youtu.be/Gl2WXLIMvKA) &nbsp; [Recurrent Neural Network](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_rnn_gru_lstm.py)
* [![Youtube Link][logo]](https://youtu.be/jGst43P-TJA) &nbsp; [Bidirectional Recurrent Neural Network](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_bidirectional_lstm.py)
* [![Youtube Link][logo]](https://youtu.be/g6kQl_EFn84) &nbsp; [Loading and saving model](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/804c45e83b27c59defb12f0ea5117de30fe25289/ML/Pytorch/Basics/pytorch_loadsave.py#L26-L34)
* [![Youtube Link][logo]](https://youtu.be/ZoZHd0Zm3RY) &nbsp; [Custom Dataset (Images)](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/aba36b89b438ca8f608a186f4d61d1b60c7f24e0/ML/Pytorch/Basics/custom_dataset/custom_dataset.py#L12-L29)
* [![Youtube Link][logo]](https://youtu.be/9sHcLvVXsns) &nbsp; [Custom Dataset (Text)](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset_txt/loader_customtext.py)
* [![Youtube Link][logo]](https://youtu.be/qaDe0qQZ5AQ) &nbsp; [Transfer Learning and finetuning](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/804c45e83b27c59defb12f0ea5117de30fe25289/ML/Pytorch/Basics/pytorch_pretrain_finetune.py#L33-L54)
* [![Youtube Link][logo]](https://youtu.be/Zvd276j9sZ8) &nbsp; [Transforms & Data Augmentation](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/804c45e83b27c59defb12f0ea5117de30fe25289/ML/Pytorch/Basics/pytorch_transforms.py#L56-L72)
* [![Youtube Link][logo]](https://youtu.be/P31hB37g4Ak) &nbsp; [Learning Rate Scheduler](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/804c45e83b27c59defb12f0ea5117de30fe25289/ML/Pytorch/Basics/pytorch_lr_ratescheduler.py#L45-L78) 
* [![Youtube Link][logo]](https://youtu.be/xWQ-p_o0Uik) &nbsp; [Initialization of weights](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/804c45e83b27c59defb12f0ea5117de30fe25289/ML/Pytorch/Basics/pytorch_init_weights.py#L35-L49)
* [![Youtube Link][logo]](https://youtu.be/RLqsxWaQdHE) &nbsp; [TensorBoard Example](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/79f2e1928906f3cccbae6c024f3f79fd05262cd1/ML/Pytorch/Basics/pytorch_tensorboard_.py#L72-L120)
* [![Youtube Link][logo]](https://youtu.be/y6IEcEBRZks) &nbsp; [Calculate Mean and STD of Images](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/55637e6afbb8cc8be6a63e04bbc899704f862911/ML/Pytorch/Basics/pytorch_std_mean.py#L41-L53)

### More Advanced
* [![Youtube Link][logo]](https://youtu.be/WujVlF_6h5A) &nbsp; [Text Generating LSTM](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Projects/text_generation_babynames/generating_names.py)
* [![Youtube Link][logo]](https://youtu.be/y2BaTt1fxJU) &nbsp; [Image Captioning](https://github.com/AladdinPerzon/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/image_captioning)
* [![Youtube Link][logo]](https://youtu.be/imX4kSKDY7s) &nbsp; [Neural Style Transfer](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/neuralstyle/nst.py)
* [![Youtube Link][logo]](https://www.youtube.com/playlist?list=PLhhyoLH6IjfzxdlsLrclcCTsS8kIcfWJb) &nbsp; [Torchtext [1]](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/torchtext/torchtext_tutorial1.py) [Torchtext [2]](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/torchtext/torchtext_tutorial2.py) [Torchtext [3]](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/torchtext/torchtext_tutorial3.py)
* [![Youtube Link][logo]](https://youtu.be/EoGUlvhRYpk) &nbsp; [Seq2Seq](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/Seq2Seq/seq2seq.py) **- Sequence to Sequence (LSTM)**
* [![Youtube Link][logo]](https://youtu.be/sQUqQddQtB4) &nbsp; [Seq2Seq + Attention](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/Seq2Seq_attention/seq2seq_attention.py) **- Sequence to Sequence with Attention (LSTM)**
* [![Youtube Link][logo]](https://youtu.be/M6adRGJe5cQ) &nbsp; [Seq2Seq Transformers](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/seq2seq_transformer/seq2seq_transformer.py) **- Sequence to Sequence with Transformers**
* [![Youtube Link][logo]](https://youtu.be/U0s0f995w14) &nbsp; [Transformers from scratch](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py) **- Attention Is All You Need**

### Object Detection
* [![Youtube Link][logo]](https://youtu.be/XXYG5ZWtjj0) &nbsp; [Intersection over Union](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/metrics/iou.py) 
* [![Youtube Link][logo]](https://youtu.be/YDkjWEN8jNA) &nbsp; [Non-Max Suppression](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/metrics/nms.py)
* [![Youtube Link][logo]](https://youtu.be/FppOzcDvaDI) &nbsp; [Mean Average Precision](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/metrics/mean_avg_precision.py)
* [![Youtube Link][logo]](https://youtu.be/n9_XyCGr-MI) &nbsp; [YOLO from scratch](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO)

### Generative Adversarial Networks
* [![Youtube Link][logo]](https://youtu.be/OljTVUVzPpM) &nbsp; [Simple FC GAN](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/1.%20SimpleGAN/fc_gan.py)
* [![Youtube Link][logo]](https://youtu.be/IZtv9s_Wx9I) &nbsp; [DCGAN](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/2.%20DCGAN)
* [![Youtube Link][logo]](https://youtu.be/pG0QZ7OddX4) &nbsp; [WGAN](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/3.%20WGAN)
* [![Youtube Link][logo]](https://youtu.be/pG0QZ7OddX4) &nbsp; [WGAN-GP](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/4.%20WGAN-GP)

### Architectures
* [![Youtube Link][logo]](https://youtu.be/fcOW-Zyb5Bo) &nbsp; [LeNet5](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/79f2e1928906f3cccbae6c024f3f79fd05262cd1/ML/Pytorch/CNN_architectures/lenet5_pytorch.py#L15-L35) **- CNN architecture**
* [![Youtube Link][logo]](https://youtu.be/ACmuBbuXn20) &nbsp; [VGG](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/79f2e1928906f3cccbae6c024f3f79fd05262cd1/ML/Pytorch/CNN_architectures/pytorch_vgg_implementation.py#L16-L62) **- CNN architecture**
* [![Youtube Link][logo]](https://youtu.be/uQc4Fs7yx5I) &nbsp; [Inception v1](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_inceptionet.py) **- CNN architecture**
* [![Youtube Link][logo]](https://youtu.be/DkNIBBBvcPs) &nbsp; [ResNet](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_resnet.py) **- CNN architecture**
 
## TensorFlow Tutorials
If you have any specific video suggestion please make a comment on YouTube :)

### Beginner Course
* [![Youtube Link][logo]](https://youtu.be/5Ym-dOS9ssA) &nbsp; Tutorial 1 - Installation, Video Only
* [![Youtube Link][logo]](https://youtu.be/HPjBY1H-U4U) &nbsp; [Tutorial 2 - Tensor Basics](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial2-tensorbasics.py)
* [![Youtube Link][logo]](https://youtu.be/pAhPiF3yiXI) &nbsp; [Tutorial 3 - Neural Network](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial3-neuralnetwork.py)
* [![Youtube Link][logo]](https://youtu.be/WAciKiDP2bo) &nbsp; [Tutorial 4 - Convolutional Neural Network](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial4-convnet.py)
* [![Youtube Link][logo]](https://youtu.be/kJSUq1PLmWg) &nbsp; [Tutorial 5 - Regularization](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial5-regularization.py)
* [![Youtube Link][logo]](https://youtu.be/WAciKiDP2bo) &nbsp; [Tutorial 6 - RNN, GRU, LSTM](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial6-rnn-gru-lstm.py)
* [![Youtube Link][logo]](https://youtu.be/kJSUq1PLmWg) &nbsp; [Tutorial 7 - Functional API](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial7-indepth-functional.py)
* [![Youtube Link][logo]](https://youtu.be/cKMJDkWSDnY) &nbsp; [Tutorial 9 - Custom Layers](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial9-custom-layers.py)
* [![Youtube Link][logo]](https://youtu.be/idus3KO6Wic) &nbsp; [Tutorial 10 - Saving and Loading Models](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial10-save-model.py)
* [![Youtube Link][logo]](https://youtu.be/WJZoywOG1cs) &nbsp; [Tutorial 11 - Transfer Learning](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial11-transfer-learning.py)
* [![Youtube Link][logo]](https://youtu.be/YrMy-BAqk8k) &nbsp; [Tutorial 12 - TensorFlow Datasets](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial12-tensorflowdatasets.py)
* [![Youtube Link][logo]](https://youtu.be/8wwfVV7ixyY) &nbsp; [Tutorial 13 - Data Augmentation](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial13-data-augmentation.py)
* [![Youtube Link][logo]](https://youtu.be/WUzLJZCKNu4) &nbsp; [Tutorial 14 - Callbacks](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial14-callbacks.py)
* [![Youtube Link][logo]](https://youtu.be/S6tLSI8bjGs) &nbsp; [Tutorial 15 - Custom model.fit](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial15-customizing-modelfit.py)
* [![Youtube Link][logo]](https://youtu.be/_u7AVsxANes) &nbsp; [Tutorial 16 - Custom Loops](https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial16-customloops.py)
* [![Youtube Link][logo]](https://youtu.be/k7KfYXXrOj0) &nbsp; [Tutorial 17 - TensorBoard](https://github.com/AladdinPerzon/Machine-Learning-Collection/tree/master/ML/TensorFlow/Basics/tutorial17-tensorboard)
* [![Youtube Link][logo]](https://youtu.be/q7ZuZ8ZOErE) &nbsp; [Tutorial 18 - Custom Dataset Images](https://github.com/AladdinPerzon/Machine-Learning-Collection/tree/master/ML/TensorFlow/Basics/tutorial18-customdata-images)
* [![Youtube Link][logo]](https://youtu.be/NoKvCREx36Q) &nbsp; [Tutorial 19 - Custom Dataset Text](https://github.com/AladdinPerzon/Machine-Learning-Collection/tree/master/ML/TensorFlow/Basics/tutorial19-customdata-text)
* [![Youtube Link][logo]](https://youtu.be/ea5Z1smiR3U) &nbsp; [Tutorial 20 - Classifying Skin Cancer](https://github.com/AladdinPerzon/Machine-Learning-Collection/tree/master/ML/TensorFlow/Basics/tutorial20-classify-cancer-beginner-project-example) **- Beginner Project Example**
 
### CNN Architectures
* [LeNet](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/TensorFlow/CNN_architectures/LeNet5)
* [AlexNet](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/TensorFlow/CNN_architectures/AlexNet)
* [VGG](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/TensorFlow/CNN_architectures/VGGNet)
* [GoogLeNet](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/TensorFlow/CNN_architectures/GoogLeNet)
* [ResNet](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/TensorFlow/CNN_architectures/ResNet)
 