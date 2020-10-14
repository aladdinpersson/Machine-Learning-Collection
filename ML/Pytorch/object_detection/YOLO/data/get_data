#!/usr/bin/env bash

## DOWNLOAD from JOSEPHS WEBSITE (SLOWER DOWNLOAD)                                 
#wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
#wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
#wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar    
                                                              
## OR DOWNLOAD FROM HERE (FASTER DOWNLOAD)                                          
# VOC2007 DATASET                                                              
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.ta
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar # 

# VOC2012 DATASET                                                              
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.ta

# Extract tar files
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar

# Need voc_label.py to clean up data from xml files
wget https://pjreddie.com/media/files/voc_label.py

# Run python file to clean data from xml files
python voc_label.py

# Get train by using train+val from 2007 and 2012
# Then we only test on 2007 test set
# Unclear from paper what they actually just as a dev set
cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
cp 2007_test.txt test.txt

# Move txt files we won't be using to clean up a little bit
mkdir old_txt_files
mv 2007* 2012* old_txt_files/

python generate_csv.py

mkdir data
mkdir data/images
mkdir data/labels

cp VOCdevkit/*.jpg data/images/
cp VOCdevkit/VOC2007/labels/*.txt data/labels/
cp VOCdevkit/VOC2012/labels/*.txt data/labels/

mkdir data                                                                              
mkdir data/images                                                                       
mkdir data/labels                                                                       
                                                                                        
mv VOCdevkit/VOC2007/JPEGImages/*.jpg data/images/                                      
mv VOCdevkit/VOC2012/JPEGImages/*.jpg data/images/                                      
mv VOCdevkit/VOC2007/labels/*.txt data/labels/                                          
mv VOCdevkit/VOC2012/labels/*.txt data/labels/ 

# We don't need VOCdevkit folder anymore, can remove
# in order to save some space 
rm -rf VOCdevkit/

mv test.txt old_txt_files/
mv train.txt old_txt_files/
