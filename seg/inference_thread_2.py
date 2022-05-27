from PIL import Image
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt

import torch

from utils.dataloader import make_datapath_list, DataTransform

import cv2
import os

import numpy as np
from sklearn.model_selection import train_test_split
import time
import threading
import shutil
# import torch.nn.utils.prune as prune
# from numba import jit



# ファイルパスリスト作成
rootpath = "./data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
    rootpath=rootpath)

# 後ほどアノテーション画像のみを使用する

from utils.pspnet import PSPNet

# def infe():

net = PSPNet(n_classes=2)

net.eval()

# 学習済みパラメータをロード
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark=True
state_dict = torch.load("./weights/pspnet18_30_2.pth",
                        map_location={'cuda:0': 'cpu'})
# net = torch.nn.DataParallel(net)
net.load_state_dict(state_dict, strict=False)
net.to(device)

# module_name = []
# for i in net.state_dict().keys():
#     if "weight" in i:
#         i = i.replace(".weight", "")
#         module_name.append(i)
# # print(module_name)

# for j in module_name:
#     module = eval("net."+j)

#     # prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)
#     prune.random_unstructured(module, name="weight", amount=0.5)
#     module._forward_pre_hooks
#     # print(list(module.named_buffers()))

print('ネットワーク設定完了：学習済みの重みをロードしました')

# print(net.state_dict().keys())
# print(net.feature_dilated_res_1.block6.cb_3.conv.weight)

path = "./predata"
pathlist = []
files_pathlist = os.listdir(path)



# 2. 前処理クラスの作成
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
# transform = DataTransform(
#     input_size=475, color_mean=color_mean, color_std=color_std)

transform = DataTransform(
    input_size=475)

# img_width = 640
# img_height = 512

# @jit
def img_pro(y, i):
    # t1 = time.time()
    # 5. PSPNetの出力から最大クラスを求め、カラーパレット形式にし、画像サイズを元に戻す
    y = y[0].cpu().detach().numpy()  # y：torch.Size([1, 21, 475, 475])
    y = np.argmax(y, axis=0)
    anno_class_img = Image.fromarray(np.uint8(y), mode="P")
    # anno_class_img = cv2.imwrite(np.uint8(y))
    anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
    anno_class_img.putpalette(p_palette)
    # print(time.time()-t)

    # 6. 画像を透過させて重ねる
    # trans_img = Image.new('RGBA', anno_class_img.size, (0, 0, 0, 0))
    # anno_class_img = anno_class_img.convert('RGBA')  # カラーパレット形式をRGBAに変換
    # img = cv2.imread(image_file_path)   #imgをcv2で開く
    # # ocv_img = np.array(img)
    # ocv_anno_class_img = np.asarray(anno_class_img) #pillow→cv2
    # trans_img = np.asarray(trans_img)

    # img_bool = cv2.resize(img_bool,size)
    # ocv_anno_class_img.flags.writeable = True
    # ocv_anno_class_img[np.where((ocv_anno_class_img == [0, 0, 0,255]).all(axis=-1))] = [0, 0, 0, 0]

    # print(time.time()-t)
    # for x in range(img_width):
    #     for y in range(img_height):
    #         # 推論結果画像のピクセルデータを取得
    #         pixel = anno_class_img.getpixel((x, y))
    #         r, g, b, a = pixel

    #         # (0, 0, 0)の背景ならそのままにして透過させる
    #         if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
    #             continue
    #         else:
    #             # それ以外の色は用意した画像にピクセルを書き込む
    #             trans_img.putpixel((x, y), (r, g, b, 2000))
                # 150は透過度の大きさを指定している
    # blended = cv2.addWeighted(src1=img1,alpha=1,src2=img2,beta=0.3,gamma=0)
    # print(time.time()-t)
    # img = Image.open(image_file_path)   # [高さ][幅][色RGB]
    # result = Image.alpha_composite(img.convert('RGBA'), trans_img)
    # print(img_bool.shape)
    # print(img.shape)
    # print(trans_img.shape)
    # print(trans_img.shape[1::-1])
    
    # dst = cv2.addWeighted(img_bool, 0.1, trans_img, 0.9, 0)
    
    # im_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # result = Image.alpha_composite(img.convert('L'), trans_img)
    # result = result.convert('L')
    # result = blended.convert('L')
    # plt.imshow(result)
    # plt.show()

    # result.save("seg_data/"+files_pathlist[k]+"/"+pathlist_f[i])
    # cv2.imwrite("seg_data/"+files_pathlist[k]+"/"+pathlist_f[i], img_bool)
    # cv2.imwrite("seg_data/"+files_pathlist[k]+"/"+pathlist_f[i], im_gray)
    anno_class_img = anno_class_img.convert('RGBA')  # カラーパレット形式をRGBAに変換
    img = cv2.imread(image_file_path)   #imgをcv2で開く
    ocv_anno_class_img = np.asarray(anno_class_img) #pillow→cv2
    
    color_lower = np.array([0, 0, 0, 255])                 # 抽出する色の下限(BGR形式)
    color_upper = np.array([0, 0, 0, 255])                 # 抽出する色の上限(BGR形式)
    img_mask = cv2.inRange(ocv_anno_class_img, color_lower, color_upper)    # 範囲からマスク画像を作成
    img_bool = cv2.bitwise_not(ocv_anno_class_img, ocv_anno_class_img, mask=img_mask)      # 元画像とマスク画像の演算(背景を白くする)
    img_bool = cv2.cvtColor(img_bool, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.bitwise_and(img_bool, img)
    cv2.imwrite("seg_data/"+files_pathlist[k]+"/"+pathlist_f[i], dst)
    # print(time.time()-t1)

# @jit
def infe(i):
    # t2 = time.time()
    # 1. 元画像の表示
    image_file_path = "./predata/"+files_pathlist[k]+"/"+pathlist_f[i+2]
    img = Image.open(image_file_path)   # [高さ][幅][色RGB]
    img_width, img_height = img.size
    # print(img_width)
    # print(img_height)
    # img = Image.close
    # print(time.time()-t2)

    # 3. 前処理
    # 適当なアノテーション画像を用意し、さらにカラーパレットの情報を抜き出す
    anno_file_path = val_anno_list[0]
    anno_class_img = Image.open(anno_file_path)   # [高さ][幅]
    p_palette = anno_class_img.getpalette()
    phase = "val"
    img, anno_class_img = transform(phase, img, anno_class_img)

    # print(time.time()-t2)
    # 4. PSPNetで推論する
    x = img.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 475, 475])
    x = x.to(device)
    # print(time.time()-t2)
    outputs = net(x)
    y = outputs[0]  # AuxLoss側は無視 yのサイズはtorch.Size([1, 21, 475, 475])
    # # 5. PSPNetの出力から最大クラスを求め、カラーパレット形式にし、画像サイズを元に戻す
    # y = y[0].cpu().detach().numpy()  # y：torch.Size([1, 21, 475, 475])
    # print(time.time()-t2)

def loop1():
    for i in range(0,len(pathlist_f),2):
        t=time.time()


        thread1 = threading.Thread(target=img_pro(y,i))
        thread2 = threading.Thread(target=infe(i))
        thread1.start()
        thread2.start()
        print(time.time()-t)
def loop2():
    for i in range(1,len(pathlist_f),2):
        t1=time.time()


        thread3 = threading.Thread(target=img_pro(y2,i))
        thread4 = threading.Thread(target=infe(i))
        thread3.start()
        thread4.start()
        print(time.time()-t1)

for k in range(len(files_pathlist)):
    pathlist = "./predata/"+files_pathlist[k]
    pathlist_f = os.listdir(pathlist)

    pathlist_dir = "./seg_data/"+files_pathlist[k]

    if (os.path.exists(pathlist_dir)):
        shutil.rmtree(pathlist_dir)
    else:
        os.mkdir(pathlist_dir)


    # 1. 元画像の表示
    image_file_path = "./predata/"+files_pathlist[k]+"/"+pathlist_f[0]
    img = Image.open(image_file_path)   # [高さ][幅][色RGB]
    img_width, img_height = img.size
    # print(img_width)
    # print(img_height)
    # img = Image.close


    # 3. 前処理
    # 適当なアノテーション画像を用意し、さらにカラーパレットの情報を抜き出す
    anno_file_path = val_anno_list[0]
    anno_class_img = Image.open(anno_file_path)   # [高さ][幅]
    p_palette = anno_class_img.getpalette()
    phase = "val"
    img, anno_class_img = transform(phase, img, anno_class_img)


    # 4. PSPNetで推論する
    # print(img.shape)
    x = img.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 475, 475])
    # print(x.shape)
    x = x.to(device)
    outputs = net(x)
    y = outputs[0]

    # print(y.shape)
    # print(y)
    # # 5. PSPNetの出力から最大クラスを求め、カラーパレット形式にし、画像サイズを元に戻す
    # y = y[0].cpu().detach().numpy()  # y：torch.Size([1, 21, 475, 475])

        # 1. 元画像の表示
    image_file_path2 = "./predata/"+files_pathlist[k]+"/"+pathlist_f[1]
    img2 = Image.open(image_file_path2)   # [高さ][幅][色RGB]
    img_width2, img_height2 = img2.size
    # print(img_width)
    # print(img_height)
    # img = Image.close


    # 3. 前処理
    # 適当なアノテーション画像を用意し、さらにカラーパレットの情報を抜き出す
    anno_file_path = val_anno_list[0]
    anno_class_img = Image.open(anno_file_path)   # [高さ][幅]
    p_palette = anno_class_img.getpalette()
    phase = "val"
    img2, anno_class_img = transform(phase, img2, anno_class_img)


    # 4. PSPNetで推論する
    # print(img.shape)
    x2 = img2.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 475, 475])
    # print(x.shape)
    x2 = x2.to(device)
    outputs2 = net(x2)
    y2 = outputs2[0]

    thread5 = threading.Thread(target=loop1)
    thread6 = threading.Thread(target=loop2)
    thread5.start()
    thread6.start()
    
    
        # thread1.join()
        # thread1.join()
        # thread2.join()
       
        # print(time.time()-t)
        
        # print(time.time()-t)


       


print("finish!")