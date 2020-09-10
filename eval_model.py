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
    pred = (pred.cpu() > threshold.cpu())
    
    # multishow(np.uint8(img.numpy()[0]), np.uint8(label.numpy()[0]), np.uint8(pred.numpy()[0]))
    break



