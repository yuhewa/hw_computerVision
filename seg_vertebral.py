import os
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# 如果不自己寫dataload的話, 可以用下面這個預設的loader
# import torchvision.datasets as dset
# dset.ImageFolder()
import albumentations as albu

import segmentation_models_pytorch as smp


# 兩張圖片並排顯示
def multishow(img, label):
    fig=plt.figure(figsize=(8, 12))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    fig.add_subplot(1, 2, 2)
    plt.imshow(label)
    plt.show()

# augmentation函式
def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=1200, min_width=500, always_apply=True, border_mode=0),
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


# get_validation_augmentation和get_preprocessing還不是很懂怎麼運作的
def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(1200, 500)
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

# dir_path = os.path.dirname(__file__)
# 發現新的方法 getcwd, 一樣是用來取得當前目錄, 與上面那個有一點差異
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

        
 # override getitem和len這兩個方法
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.filenames[index])
        image = cv2.imread(img_path)
        label_path = os.path.join(self.label_dir, self.filenames[index])
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        
        labels = [(label == value) for value in self.class_values]
        label = np.stack(labels, axis=-1).astype('float')


        # augmentation和preprocessing不太懂是怎麼做的
        # augmentation
        if self.augmentation:
            sample = self.augmentation(image=image, label=label)
            image, label = sample['image'], sample['label']
        
        # preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, label=label)
            image, label = sample['image'], sample['label']

        return image, label

    def __len__(self):
        return len(self.filenames)








# train_dataset = imageDataset(train_dir,)
# train_loader = DataLoader(dataset, batch_size=10, shuffle=False)

# augmented_dataset = imageDataset(train_dir,augmentation=get_training_augmentation(),)
# augmented_loader = DataLoader(augmented_dataset, batch_size=10, shuffle=False) 

# 若想控制取出量, 調整batch_size就好, 他會一個batch取出一張
# 若有20張, batch_size設10. 則有2個batch, 就會取出2張





# 直接從dataset取出圖片
# image:train_dataset[0][0], label:train_dataset[0][1]
# multishow(np.uint8(train_dataset[0][0]), np.uint8(train_dataset[0][1]))

# 從loader取出圖片
# for img, label in augmented_loader:

    # # 兩張並排顯示, 不過無法控制停止
    # multishow(np.uint8(img.numpy()[0]), np.uint8(label.numpy()[0]))
    # break
    # 單張顯示
    # cv2.imshow('vertebral', np.uint8(label.numpy()[0])) #不確定為什麼要使用.numpy()[0]才能顯示
    # key = cv2.waitKey(1000) # 1000ms = 1s
    # if key == 27: #27代表ESC
    #     break

# 使用這個函數似乎可以依序取出全部圖片
# batch = next(iter(train_loader))
# img, label = batch


# model設定
# aux_params=dict(
#     pooling='max',             # one of 'avg', 'max'
#     dropout=0.5,               # dropout ratio, default is None
#     activation='sigmoid',      # activation function, default is None
#     classes=2,                 # define number of output labels
# )


# 設定model
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'
CLASSES = ['vertebral']


model = smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    activation=ACTIVATION,
    classes=1,
)


# 讀取train和validation的資料
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = imageDataset(
    train_dir,
    augmentation = get_training_augmentation(),  
    preprocessing = get_preprocessing(preprocessing_fn),
    classes=CLASSES
)

val_dataset = imageDataset(
    val_dir,
    # augmentation = get_validation_augmentation(),  
    preprocessing = get_preprocessing(preprocessing_fn),
    classes=CLASSES
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# 設定 loss 和 optimizer
loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

val_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)



# train model for 40 epochs

max_score = 0

for i in range(0, 10):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    val_logs = val_epoch.run(val_loader)
    
    # do something (save model, change lr, etc.)
    if max_score < val_logs['iou_score']:
        max_score = val_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

