"""
Splits the dataset into training, testing, and validation set. 
Default train : test : val = 8 : 1 : 1

After execution there will be four documents in Imagesets folder
"""

import os
import random

trainval_percent = 0.9
train_percent = 0.9
imageFolderPath = 'data/images'
txtsavepath = 'data/ImageSets'
total_images = os.listdir(imageFolderPath)

num = len(total_images)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('data/ImageSets/trainval.txt', 'w')
ftest = open('data/ImageSets/test.txt', 'w')
ftrain = open('data/ImageSets/train.txt', 'w')
fval = open('data/ImageSets/val.txt', 'w')

for i in list:
    name = total_images[i][:-4] + '\n'
    #print(name)
    
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
   
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()