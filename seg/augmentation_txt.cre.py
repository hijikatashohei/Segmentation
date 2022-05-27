import os
import glob
import random
import numpy as np
from sklearn.model_selection import train_test_split
import math
import time
import codecs
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from PIL import Image
import albumentations as A
import cv2
import sys

gra_n = "20000"
brightness = 0


path_img = "data/VOCdevkit/VOC2012/JPEGImages/"
path_ano = "data/VOCdevkit/VOC2012/SegmentationClass/"
f_list = list()
f_list_last = list()
# f_list_ano = list()
# f_list_ano_last = list()

#明暗＿データ増強#

# Declare an augmentation pipeline


transform_1 = A.Compose([
    # A.RandomCrop(width=256, height=256),
    # A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.2,-0.2),contrast_limit=(0.0, 0.0),p=1),
    # A.ShiftScaleRotate(shift_limit=(0,0),scale_limit=(0,0),rotate_limit=(0,0),shift_limit_x=(0.04,0.04),shift_limit_y=(0,0),p=1)
])

transform_2 = A.Compose([
    # A.RandomCrop(width=256, height=256),
    # A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.4,-0.4),contrast_limit=(0.0, 0.0),p=1),
    # A.ShiftScaleRotate(shift_limit=(0,0),scale_limit=(0,0),rotate_limit=(0,0),shift_limit_x=(0.04,0.04),shift_limit_y=(0,0),p=1)
])

transform_3 = A.Compose([
    # A.RandomCrop(width=256, height=256),
    # A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(0.15,0.15),contrast_limit=(0.0, 0.0),p=1),
    # A.ShiftScaleRotate(shift_limit=(0,0),scale_limit=(0,0),rotate_limit=(0,0),shift_limit_x=(0.04,0.04),shift_limit_y=(0,0),p=1)
])

transform_4 = A.Compose([
    # A.RandomCrop(width=256, height=256),
    # A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(0.3,0.3),contrast_limit=(0.2, 0.0),p=1),
    # A.ShiftScaleRotate(shift_limit=(0,0),scale_limit=(0,0),rotate_limit=(0,0),shift_limit_x=(0.04,0.04),shift_limit_y=(0,0),p=1)
])

transform_5 = A.Compose([
    # A.RandomCrop(width=256, height=256),
    # A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.5,-0.5),contrast_limit=(0.0, 0.0),p=1),
    # A.ShiftScaleRotate(shift_limit=(0,0),scale_limit=(0,0),rotate_limit=(0,0),shift_limit_x=(0.04,0.04),shift_limit_y=(0,0),p=1)
])

# transformed = transform(im)
# transformed_image = transformed["image"]



name_list = glob.glob(os.path.join(path_img, '*'))
name_list_ano = glob.glob(os.path.join(path_ano, '*'))
# print(name_list)
num = len(name_list_ano)-1


for p in name_list:
    name = os.path.splitext(os.path.basename(p))[0]
    # print(name)
    f_list_last.append(name)
    
f_list_last.sort(key=int)
# print(f_list_last)


# for p in name_list_ano:
#     name = os.path.splitext(os.path.basename(p))[0]
#     # print(name)
#     f_list_ano_last.append(name)
    
# f_list_ano_last.sort(key=int)
# print(f_list_ano_last)


for i in range(num):
    im = cv2.imread(path_img+f_list_last[i]+".jpg", cv2.IMREAD_GRAYSCALE)
    im_np = np.array(im)

    transformed_1 = transform_1(image=im_np)
    transformed_image_1 = transformed_1["image"]

    transformed_2 = transform_2(image=im_np)
    transformed_image_2 = transformed_2["image"]

    transformed_3 = transform_3(image=im_np)
    transformed_image_3 = transformed_3["image"]

    transformed_4 = transform_4(image=im_np)
    transformed_image_4 = transformed_4["image"]

    transformed_5 = transform_5(image=im_np)
    transformed_image_5 = transformed_5["image"]

    im_ano = Image.open(path_ano+f_list_last[i]+".png")
    # print("aaaaaaa")
    # copy_im_ano = im_ano.copy()
    # print("bbbbbbb")
    cv2.imwrite(path_img+str(i+201)+'.jpg', transformed_image_1)
    cv2.imwrite(path_img+str(i+201+num)+'.jpg', transformed_image_2)
    cv2.imwrite(path_img+str(i+201+num+num)+'.jpg', transformed_image_3)
    cv2.imwrite(path_img+str(i+201+num+num+num)+'.jpg', transformed_image_4)
    cv2.imwrite(path_img+str(i+201+num+num+num+num)+'.jpg', transformed_image_5)

    im_ano.save(path_ano+str(i+201)+'.png')
    im_ano.save(path_ano+str(i+201+num)+'.png')
    im_ano.save(path_ano+str(i+201+num+num)+'.png')
    im_ano.save(path_ano+str(i+201+num+num+num)+'.png')
    im_ano.save(path_ano+str(i+201+num+num+num+num)+'.png')


    # name = os.path.splitext(os.path.basename(path_img+str(i+201)+'.jpg'))[0]
    # f_list.append(name+"\n")

#パスリスト更新#
# name_list = glob.glob(os.path.join(path_img, '*'))


# for p in name_list:
#     name = os.path.splitext(os.path.basename(p))[0]
#     print(name)
#     f_list.append(name+"\n")



f_list_tx = list()
name_list_tx = glob.glob(os.path.join(path_ano, '*'))

for p in name_list_tx:
    name = os.path.splitext(os.path.basename(p))[0]
    # print(name)
    f_list_tx.append(name+"\n")

os.remove('data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt')
os.remove('data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')

f_1 = open('data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt', 'a')  #a：上書きor新規作成#

train_size = 0.8
test_size = 0.1
train, val = train_test_split(
    f_list_tx, train_size=train_size, shuffle=True)




f_1.writelines(train)

f_2 = open('data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt', 'a')
f_2.writelines(val)
