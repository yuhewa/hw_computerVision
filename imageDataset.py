import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from cv2 import cv2

class imageDataset(Dataset):
    def __init__(self, file_dir, classes = None, augmentation = None):
        self.file_dir =  file_dir
        self.img_dir = os.path.join(file_dir, 'image')
        self.label_dir = os.path.join(file_dir, 'label')
        self.filenames = os.listdir(self.img_dir)

        # 從classes取出字串, 轉小寫後取出其對應index
        self.class_values = [classes.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((1216,512)), # 要使之為32的倍數, 網絡架構的關係
            transforms.ToTensor() # 原本是heigh, width, channel, 變成channel, heigh, width 就可以對應了
            ]) 
        
 # override getitem和len這兩個方法
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.filenames[index])
        image = cv2.imread(img_path)
        label_path = os.path.join(self.label_dir, self.filenames[index])
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        labels = [(label == value) for value in self.class_values]
        label = np.stack(labels, axis=-1).astype('float')

        # augmentation
        if self.augmentation != None:
            image = self.augmentation(np.uint8(image))
            label = self.augmentation(np.uint8(label))
        else:
            image = self.preprocessing(np.uint8(image))
            label = self.preprocessing(np.uint8(label))
 
        return image, label

    def __len__(self):
        return len(self.filenames)
