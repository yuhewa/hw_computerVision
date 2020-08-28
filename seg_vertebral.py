import os
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
# 如果不自己寫dataload的話, 可以用下面這個預設的loader
# import torchvision.datasets as dset
# dset.ImageFolder()

# 若要做變換(也就是augmentation), 需用到torchvision.transform.compose()函數
# 可以 import torchvision.transform as trasform

# segmentation模組
# import segmentation_models_pytorch as smp
# from segmentation_models_pytorch.encoders import get_preprocessing_fn



# 取得檔案路徑後, 在其後加上欲讀取檔案名稱
# dir_path = os.path.dirname(__file__)
# 發現新的方法 getcwd, 一樣是用來取的路徑, 與上面那個有一點差異
root_dir = os.getcwd() 
train_file = os.path.join(root_dir, "vertebral","f01")
val_file = os.path.join(root_dir, "vertebral","f02")


# 先定義一個Dataset的子類
class imageDataset(Dataset):
    def __init__(self, file_dir, transform = None):
        self.file_dir =  file_dir
        self.transform = transform

        self.img_dir = os.path.join(file_dir, 'image')
        self.label_dir = os.path.join(file_dir, 'label')

        self.filenames = os.listdir(self.img_dir)

        # print(self.file_dir)
        # print(self.img_dir)
        # print(self.filenames)
        
 # override getitem和len這兩個方法
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.filenames[index])
        img = cv2.imread(img_path)
        label_path = os.path.join(self.label_dir, self.filenames[index])
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # print(img_path)
        # print(label_path)
        return img, label

    def __len__(self):
        return len(self.filenames)


dataset = imageDataset(train_file)
train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
# 若想控制取出量, 調整batch_size就好, 他會一個batch取出一張
# 若有20張, batch_size設10. 則有2個batch, 就會取出2張

for img, label in train_loader:
    cv2.imshow('vertebral', np.uint8(label.numpy()[0])) #不確定為什麼要使用.numpy()才能顯示
    key = cv2.waitKey(1000) # 1000ms = 1s
    if key == 27: #27代表ESC
        break








# model設定
# aux_params=dict(
#     pooling='max',             # one of 'avg', 'max'
#     dropout=0.5,               # dropout ratio, default is None
#     activation='relu',      # activation function, default is None
#     classes=2,                 # define number of output labels
# )
# model = smp.Unet('resnet18', classes=2, encoder_weights='imagenet', aux_params=aux_params)








