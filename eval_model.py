import os
from cv2 import cv2 
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

from imageDataset import imageDataset
from PIL import Image 

# 圖片並排顯示
def multishow(img, label, pred):
    fig=plt.figure(figsize=(8, 5))
    fig.add_subplot(1, 3, 1)
    plt.imshow(img)
    fig.add_subplot(1, 3, 2)
    plt.imshow(1 - label, cmap ='gray') # 1 - label 反轉黑白
    fig.add_subplot(1, 3, 3)
    plt.imshow(1 - pred, cmap ='gray')
    plt.show()


root_dir = os.getcwd()
train_dir = os.path.join(root_dir, "vertebral","f01")
modelfile_path = os.path.join(root_dir,'best_model','test_model.pth')


ENCODER = 'resnet18'
ACTIVATION = 'sigmoid'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['vertebral']
model = smp.Unet(
    encoder_name=ENCODER, 
    activation=ACTIVATION,
    classes=1,
).to(device)
model.load_state_dict(torch.load(modelfile_path))


train_dataset = imageDataset(train_dir, classes=CLASSES)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

Threshold_val = 0.6
threshold = torch.tensor([Threshold_val])


for img, label in train_loader:
    img = img.to(device)
    pred = model(img)
    pred = (pred.cpu() > threshold.cpu()) # output為tensor
    pred = pred.cpu() # 要把他轉存到cpu, 不然會有bug
    pred = pred.numpy().squeeze()
    # 化為numpy,並squeeze, 使1 1 1216 512 變成 1216 512, 便可以轉成圖片
    # 擔心一個問題, 若batch size不是1的話, 似乎無法squeeze, 可能要記得如果要eval就只能設batch size=1


    img = img.squeeze().permute(1,2,0).cpu() # 將channel移到後面才能正確顯示
    label = label.squeeze().cpu()

    # pred_img = Image.fromarray(pred.astype(np.uint8), 'L')  # 將matix轉為圖片

    # multishow(np.uint8(img.numpy()[0]), np.uint8(label.numpy()[0]), np.uint8(label.numpy()[0]))
    # 原本可以運作的multishow, 因為圖片經過前處理, channel到最前面, 無法正確運作
    multishow(img.numpy(), label.numpy(), pred)

    break



