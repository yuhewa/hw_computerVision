import os

#兩種讀取方式的shape會不一樣
from skimage import io
from cv2 import cv2 

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def multishow(img, label):
    fig=plt.figure(figsize=(8, 12))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    fig.add_subplot(1, 2, 2)
    plt.imshow(label)
    plt.show()



root_dir = os.getcwd() 
train_dir = os.path.join(root_dir, "vertebral","f01")
val_dir = os.path.join(root_dir, "vertebral","f02")


# 先定義一個Dataset的子類
class imageDataset(Dataset):
    def __init__(self, file_dir, classes = None, augmentation = None, preprocessing = None):
        self.file_dir =  file_dir
        self.img_dir = os.path.join(file_dir, 'image')
        self.label_dir = os.path.join(file_dir, 'label')
        self.filenames = os.listdir(self.img_dir)

        # 從classes取出字串, 轉小寫後取出其對應index
        self.class_values = [classes.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        # print(self.file_dir)
        # print(self.img_dir)
        # print(self.filenames)
                
 # override getitem和len這兩個方法
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.filenames[index])
        img = cv2.imread(img_path)
        sk_img = io.imread(img_path)

        label_path = os.path.join(self.label_dir, self.filenames[index])
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) # label shape: 1200 500
        sk_label = io.imread(img_path, as_gray=True)

        print('--------------------in dataset-------------------------------')
        print('cv2.imread:')
        print(img.shape,'     ', label.shape)
        print('---------------------------------------------------')
        print('skimage.io.imread:')
        print(sk_img.shape,'     ', sk_label.shape)
        print('---------------------------------------------------')
        # labels = [(label == value) for value in self.class_values]  # labels shape 3 1200 500
        # label = np.stack(labels, axis=-1).astype('float') # label shape: 1200 500 3

        # # augmentation
        # if self.augmentation:
        #     sample = self.augmentation(image=img, label=label)
        #     img, label = sample['image'], sample['label']
        
        # # preprocessing
        # if self.preprocessing:
        #     sample = self.preprocessing(image=img, label=label)
        #     img, label = sample['image'], sample['label']

        return img, label

    def __len__(self):
        return len(self.filenames)

CLASSES = 'zxc'

train_dataset = imageDataset(
    train_dir,
    # augmentation = get_training_augmentation(),
    classes=CLASSES
)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)


for img, label in train_loader:
    # print('--------------------after dataloader-------------------------------')
    # print('cv2.imread:')
    # print(img.shape,'     ', label.shape)
    # print('---------------------------------------------------')
    # print('skimage.io.imread:')
    # print(img.shape,'     ', label.shape)
    # print('---------------------------------------------------')
    break




# pytorch內建的transform, 可用來做augmentation
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomHorizontalFlip(p = 0.5),
#     transforms.ColorJitter(brightness=0.5)
#     transforms.ToTensor()
# ])

# 儲存圖片
# img_num = 21
# for i in range(aug_num)
#     save_imge('aug'+str(img_num)+'.png')
#     img_num += 1