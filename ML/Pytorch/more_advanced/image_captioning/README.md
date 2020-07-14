### Image Captioning

Download the dataset used: https://www.kaggle.com/dataset/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb
Then set images folder, captions.txt inside a folder Flickr8k.

train.py: For training the network

model.py: creating the encoderCNN, decoderRNN and hooking them togethor

get_loader.py: Loading the data, creating vocabulary

utils.py: Load model, save model, printing few test cases downloaded online