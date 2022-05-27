import os
import urllib.request
import zipfile
import tarfile

# フォルダ「data」が存在しない場合は作成する
data_dir = "./data/"   #パスを指定#
if not os.path.exists(data_dir):    #指定したパスが無ければ＝指定したフォルダが無い#
    os.mkdir(data_dir)

 # フォルダ「weights」が存在しない場合は作成する
weights_dir = "./weights/"
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)

# VOC2012のデータセットをここからダウンロードします
# 時間がかかります（約15分）
url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
target_path = os.path.join(data_dir, "VOCtrainval_11-May-2012.tar") 

if not os.path.exists(target_path):
    urllib.request.urlretrieve(url, target_path)  #作成したパスに読み込んだurlのデータを保存#
    
    tar = tarfile.TarFile(target_path)  # tarファイルを読み込み
    tar.extractall(data_dir)  # tarを解凍
    tar.close()  # tarファイルをクローズ

# フォルダ「weights」にネットワークの初期値として使用する「pspnet50_ADE20K.pth」を
# 筆者のGoogle Driveから手動でダウンロードする
    
# https://drive.google.com/open?id=12eN6SpnawYuQmD1k9VgVW3QSgPR6hICc