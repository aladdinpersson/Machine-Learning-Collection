# Exploring the MNIST dataset with PyTorch

The goal of this small project of mine is to learn different models and then try and see what kind of test accuracies we can get on the MNIST dataset. I checked some popular models (LeNet, VGG, Inception net, ResNet) and likely I will try more out in the future as I learn more network architectures. I used an exponential learning rate decay and data augmentation, in the beginning I was just using every data augmentation other people were using but I learned that using RandomHorizontalFlip when learning to recognize digits might not be so useful (heh). I also used a lambda/weight decay of pretty standard 5e-4. My thinking during training was first that I split into a validationset of about 10000 examples and made sure that it was getting high accuracies on validationset with current hyperparameters. After making sure that it wasn't just overfitting the training set, I changed so that the model used all of the training examples (60000) and then when finished training to about ~99.9% training accuracy I tested on the test set.

## Accuracy
| Model |  Number of epochs  | Training set acc. | Test set acc. |
| ----------------- | ----------- | ----------------- | ----------- |
| [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) | 150 | 99.69%      | 99.12%  |
| [VGG13](https://arxiv.org/abs/1409.1556)              | 100 |  99.95%      |  99.67%   |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 100 |  99.92%      |  99.68%   |
| [GoogLeNet](https://arxiv.org/abs/1409.4842)          | 100 |  99.90%      |  99.71%   |
| [ResNet101](https://arxiv.org/abs/1512.03385)          | 100 | 99.90%      |  99.68%  |

TODO: MobileNet, ResNext, SqueezeNet, .., ?

### Comments and things to improve
I believe LeNet has more potential as it's not really overfitting the training set that well and needs more epochs. I believe that in the original paper by LeCun et. al. (1998) showed that they achieved about 99.1% test accuracy which is similar to my results but we also need to remember the limitations that were back then. I do think training it for a bit longer to make it ~99.8-99.9% on training set would get it up to perhaps 99.2-99.3% test accuracy if we're lucky. So far the other models I think have performed quite well and is close, at least from my understanding, to current state of the art. If you would like to really maximize accuracy you would train an ensemble of models and then average their predictions to achieve better accuracy but I've not done that here as I don't think it's that interesting. This was mostly to learn different network architectures and to then check if they work as intended. If you find anything that I can improve or any mistakes, please tell me what and I'll do my best to fix it!

### How to run
```bash
usage: train.py [-h] [--resume PATH] [--lr LR] [--weight-decay R]
                [--momentum R] [--epochs N] [--batch-size N]
                [--log-interval N] [--seed S] [--number-workers S]
                [--init-padding S] [--create-validationset] [--save-model]

PyTorch MNIST

optional arguments:
  --resume PATH Saved model. (ex: PATH = checkpoint/mnist_LeNet.pth.tar)
  --batch-size N (ex: --batch-size 64), default is 128.
  --epochs N  (ex: --epochs 10) default is 100.
  --lr LR learning rate (ex: --lr 0.01), default is 0.001.
  --momentum M SGD w momentum (ex: --momentum 0.5), default is 0.9.
  --seed S random seed (ex: --seed 3), default is 1.
  --log-interval N print accuracy ever N mini-batches, ex (--log-interval 50), default 240.
  --init-padding S Initial padding on images (ex: --init-padding 5), default is 2 to make 28x28 into 32x32.
  --create-validation to create validationset
  --save-model to save weights
  --weight-decay R What weight decay you want (ex: --weight-decay 1e-4), default 1e-5.
  --number-workers S How many num workers you want in PyTorch (ex --number-workers 2), default is 0.


Example of a run is:
python train.py --save-model --resume checkpoint/mnist_LeNet.pth.tar --weight-decay 1e-5 --number-workers 2
```
