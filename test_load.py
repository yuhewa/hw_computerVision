import os
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
import albumentations as albu

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

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
    def __init__(self, file_dir, augmentation = None, preprocessing = None):
        self.file_dir =  file_dir
        self.img_dir = os.path.join(file_dir, 'image')
        self.label_dir = os.path.join(file_dir, 'label')
        self.filenames = os.listdir(self.img_dir)

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        # print(self.file_dir)
        # print(self.img_dir)
        # print(self.filenames)
        
 # override getitem和len這兩個方法
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.filenames[index])
        img = cv2.imread(img_path)
        label_path = os.path.join(self.label_dir, self.filenames[index])
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # augmentation
        if self.augmentation:
            sample = self.augmentation(image=img, label=label)
            image, label = sample['image'], sample['label']
        
        # preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, label=label)
            image, label = sample['image'], sample['label']

        return img, label

    def __len__(self):
        return len(self.filenames)

# augmentation函式
def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        #0.9的機率取出OneOf中的其中一個, 各個抽中的機率皆為1/3( 因為1/(1+1+1) )
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)

# 之後會用於取得preprocessing
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')
def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = imageDataset(
    train_dir,
    augmentation = get_training_augmentation(),  
    preprocessing = get_preprocessing(preprocessing_fn),
)

train_loader = train_dataset

for img, label in train_loader:
    print(type(img))
    print(type(label))
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