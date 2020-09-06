import os
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as transforms
# 若不用albumentaion, 也可以用pytorch預設的transform做augmentation
import albumentations as albu
from torch.utils.data import Dataset, DataLoader
# 如果不自己寫dataload的話, 可以用下面這個預設的loader
# import torchvision.datasets as dset
# dset.ImageFolder()

# 方便建構模型與使用pre-trained model
import segmentation_models_pytorch as smp




# augmentation函式
# 使用albumentation的原因是因為其運算較快
# 將圖片做各種影像處理以擴增訓練資料, p為機率
def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        # albu.PadIfNeeded(min_height=1200, min_width=500, always_apply=True, border_mode=0),
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


# get_validation用於將圖片大小補齊
def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(1200, 500)
    ]
    return albu.Compose(test_transform)

# 若不transpose, 則會出現channel數無法match的error
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

# 用於獲得pre-train model的前處理, 以及對其做transpose
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


# 先繼承Dataset, 並撰寫init、getitem、len 
# __init__用來存取檔案路徑位置
# __getitem__則將路徑與檔名串在一起並讀取圖檔, 回傳圖檔
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


# 兩張圖片並排顯示
def multishow(img, label):
    fig=plt.figure(figsize=(8, 12))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    fig.add_subplot(1, 2, 2)
    plt.imshow(label)
    plt.show()

# 直接從dataset取出圖片
# image:train_dataset[0][0], label:train_dataset[0][1]
# multishow(np.uint8(train_dataset[0][0]), np.uint8(train_dataset[0][1]))

# 從loader取出圖片
# for img, label in augmented_loader:
    # method 1 兩張並排顯示, 不過無法控制停止
    # multishow(np.uint8(img.numpy()[0]), np.uint8(label.numpy()[0]))

    # method 2 單張顯示, 可停止
    # cv2.imshow('vertebral', np.uint8(label.numpy()[0])) #不確定為什麼要使用.numpy()[0]才能顯示
    # key = cv2.waitKey(1000) # 1000ms = 1s
    # if key == 27: #27代表ESC
    #     break

# 使用這個函數似乎可以依序取出全部圖片
# batch = next(iter(train_loader))
# img, label = batch




# 設定model
ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'
CLASSES = ['vertebral']

# 可以多設 dropout=0.5 避免overfitting 
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    activation=ACTIVATION,
    classes=1,
)



preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# 讀取train和validation的資料
train_dataset = imageDataset(
    train_dir,
    augmentation = get_training_augmentation(),  
    preprocessing = get_preprocessing(preprocessing_fn),
    classes=CLASSES
)

val_dataset = imageDataset(
    val_dir,
    augmentation = get_validation_augmentation(),  
    preprocessing = get_preprocessing(preprocessing_fn),
    classes=CLASSES
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
# 若想控制取出量, 調整batch_size就好, 他會一個batch取出一張
# 若有20張, batch_size設10. 則有2個batch, 就會取出2張


# 設定 loss 和 optimizer
# IoU為segmentation的準確度
loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]
# 將model的參數傳給optimizer才能更新model
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

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





max_score = 0
epoch = 40
# 開始訓練
for i in range(0, epoch):
    print('\nEpoch: {}'.format(i))
    # 紀錄訓練過程的分數與驗證分數
    train_logs = train_epoch.run(train_loader)
    val_logs = val_epoch.run(val_loader)
    
    # 若目前驗證分數大於過往紀錄, 則儲存model
    if max_score < val_logs['iou_score']:
        max_score = val_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')
    # 避免無法收斂, 在訓練一定ㄋ次數後會減少learning rate 
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

