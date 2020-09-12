import os

from cv2 import cv2
from skimage import io


import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# 如果不自己寫dataload的話, 可以用下面這個預設的loader
# import torchvision.datasets as dset
# dset.ImageFolder()

# 方便建構模型與使用pre-trained model
import segmentation_models_pytorch as smp
# 很好用的工具, 可以跑進度條
from tqdm import tqdm
# 用來算IoU
from sklearn.metrics import confusion_matrix



# 先繼承Dataset, 並撰寫init、getitem、len 
# __init__用來存取檔案路徑位置
# __getitem__則將路徑與檔名串在一起並讀取圖檔, 回傳圖檔
class imageDataset(Dataset):
    def __init__(self, file_dir, classes = None, augmentation = None):
        self.file_dir =  file_dir
        self.img_dir = os.path.join(file_dir, 'image')
        self.label_dir = os.path.join(file_dir, 'label')
        self.filenames = os.listdir(self.img_dir)

        # 從classes取出字串, 轉小寫後取出其對應index
        self.class_values = [classes.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        
 # override getitem和len這兩個方法
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.filenames[index])
        # sk_image = io.imread(img_path)
        image = cv2.imread(img_path)

        label_path = os.path.join(self.label_dir, self.filenames[index])
        # sk_label = io.imread(label_path, as_gray=True)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        # labels = [(label == value) for value in self.class_values]
        # label = np.stack(labels, axis=-1).astype('float')

        # print('---------------------------------------------------')
        # print('in data, without augmentation:')
        # print('skimage.io.imread')
        # print(sk_image.shape,'     ', sk_label.shape)
        # print('cv2.ioread')
        # print(image.shape,'     ', label.shape)
        # print('---------------------------------------------------')

        print('---------------------------------------------------')
        print('in dataset, before augmentation:')
        print(image.shape,'     ', label.shape)
        print('---------------------------------------------------')

        # augmentation
        if self.augmentation != None:
            image = self.augmentation(np.uint8(image))
            label = self.augmentation(np.uint8(label))

        print('---------------------------------------------------')
        print('in dataset, after augmentation:')
        print(image.shape,'     ', label.shape)
        print('---------------------------------------------------')

        return image, label

    def __len__(self):
        return len(self.filenames)


CLASSES = ['zxc']

# start

# dir_path = os.path.dirname(__file__)
# 發現新的方法 getcwd, 一樣是用來取得當前目錄, 與上面那個有一點差異
root_dir = os.getcwd()
train_dir = os.path.join(root_dir, "vertebral","f01")
val_dir = os.path.join(root_dir, "vertebral","f02")

# augmentation
my_transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.Resize((1216,512)),
    # transforms.RandomRotation(degrees=10),
    #transforms.RandomCrop(500,500),
    transforms.ToTensor()
])

# 讀取train和validation的資料
train_dataset = imageDataset(train_dir, augmentation = my_transform, classes=CLASSES)
# train_dataset = imageDataset(train_dir, classes=CLASSES)
val_dataset = imageDataset(val_dir, classes=CLASSES)

TrainingLoader = DataLoader(train_dataset, batch_size=1, shuffle=True)
ValidationLoader = DataLoader(val_dataset, batch_size=1, shuffle=False)
# 若想控制取出量, 調整batch_size就好

Epoch = 10
# main program
if __name__ =='__main__':

    # 從loader出來後, 會在最前面的維度加一維batch
    # for i,(data,target) in tqdm(enumerate(TrainingLoader),total=len(TrainingLoader)):
    #     print()                               # when without augmentation
    #     print(data.shape,'    ',target.shape) # batch height width  ,  batch height width 1

    for img, label in TrainingLoader:
        print('---------------------------------------------------')
        print('in dataloader, after augmentation:')
        print(img.shape,'     ', label.shape)
        print('---------------------------------------------------')
        break