import random
import math
import time
import os
import cv2
import codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from PIL import Image
import albumentations as A

import sys

gra_n = "20000"
brightness = 0

# Declare an augmentation pipeline
transform = A.Compose([
    # A.RandomCrop(width=256, height=256),
    # A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.3,-0.3),contrast_limit=(0.0, 0.0),p=0.5),
    # A.ShiftScaleRotate(shift_limit=(0,0),scale_limit=(0,0),rotate_limit=(0,0),shift_limit_x=(0.04,0.04),shift_limit_y=(0,0),p=1)
])


# transformed = transform(im)
transformed_image = transformed["image"]


for i in range(201):
    im_ano = cv2.imread("data/VOCdevkit/VOC2012/JPEGImages"+str(i)+'.png')
    im = cv2.imread("data/VOCdevkit/VOC2012/SegmentationClass"+str(i)+'.jpg')
    transformed = transform(im)
    transformed_image = transformed["image"]
    copy_im_ano = im_ano.copy()
    cv2.imwrite("data/VOCdevkit/VOC2012/JPEGImages"+str(i+201)+'.png', transformed_image)
    cv2.imwrite("data/VOCdevkit/VOC2012/SegmentationClass"+str(i+201)+'.png', copy_im_ano)













# if __name__ == "__main__":
image = cv2.imread("{}.jpg".format(gra_n))
# colimage = Image.open("{}.jpg".format(gra_n))
# print(type(image))
# print(type(colimage))
# picture = np.array(Image.open("08440.jpg"))
# picture = np.array(cv2.imread("08440.jpg"))
picture1 = np.array(image)
# picture2 = np.array(colimage)
# print(type(picture1))
# print(type(picture2))

# print(image.shape)
# print(picture1.shape)
# print(picture2.shape)

# cv2.imwrite('data/monoim.jpg', picture1)
# cv2.imwrite('data/colorim.jpg', picture2)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(8, 5))
# Augment an image
# transformed1 = transform(image=picture1)
# transformed_image1 = transformed1["image"]

# plt.imshow(transformed_image1)

transformed = transform(image=picture1)
transformed_image = transformed["image"]

# transformed = transform(image=picture1)
transformed_image2 = transformed["image"]
# print(type(transformed_image))
# print(transformed_image.shape)

# cv2.imwrite('{}_{}_x.jpg'.format(gra_n, brightness), transformed_image)

plt.imshow(transformed_image)
# plt.imshow(image,cmap='Greys_r')
plt.show()
#画像の読み込み
# im = Image.open("08440.jpg")

#画像をarrayに変換
# im_list = np.asarray(im)
#貼り付け
# plt.imshow(im_list)
#表示
# plt.show()
    # delay = 40 # [ms]
    # csv = "dataset_list/dataset_list_init.csv"

    # input_list, anno_list = make_multi_list(csv, delay)
    # # input_list, input2_list, anno_list = make_multi_list(csv, delay)

    # dataset = Dataset(input_list, anno_list)
    # # dataset = Dataset(input_list, input2_list, anno_list)

    # print(3)
    # memory_view()
    # # print("dataset  : ", len(dataset))
    # # print("input_list", input_list)
    # # print("anno_list", anno_list)

    # train_size = 0.8
    # test_size = 0.1

    # train_dataset, dataset = train_test_split(
    #     dataset, train_size=train_size, shuffle=False)

    # val_dataset, test_dataset = train_test_split(
    #     dataset, test_size=round(test_size / (1 - train_size), 2), shuffle=False)

    # # print("train_dataset", type(train_dataset))
    # # print("train_dataset[0]", type(train_dataset[0]))
    # # print(train_dataset)

    # print("train_split : ", len(train_dataset))
    # print("data_split  : ", len(dataset))
    # print("val_split   : ", len(val_dataset))
    # print("test_split  : ", len(test_dataset))
    # memory_view()

    # random.seed(0)
    # random.shuffle(train_dataset)

    # dataset_dict = {'train': train_dataset,
    #                     'val': val_dataset, 'test': test_dataset}
    # print(4)

    # net = Net()
    # print(5)

    # batch_size = 64

    # for phase in ["train", "val"]:
    #     print("phase", phase)

    #     for i, dataset  in enumerate(dataset_dict[phase]):
    #         print(i)
    #         print(dataset)

    # num_epochs = 8
    # train_model(net, dataset_dict, batch_size=batch_size, num_epochs=num_epochs, dl_box=True)
