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
        image = cv2.imread(img_path)
        label_path = os.path.join(self.label_dir, self.filenames[index])
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        labels = [(label == value) for value in self.class_values]
        label = np.stack(labels, axis=-1).astype('float')

        # augmentation
        if self.augmentation != None:
            image = self.augmentation(np.uint8(image))
            label = self.augmentation(np.uint8(label))
 
        return image, label

    def __len__(self):
        return len(self.filenames)



# 設定model hyperparameters
ENCODER = 'resnet18'
ACTIVATION = 'sigmoid'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['vertebral']
# 可以多設 dropout=0.5 避免overfitting 
model = smp.Unet(
    encoder_name=ENCODER, 
    activation=ACTIVATION,
    classes=1,
).to(device)


optimizer = torch.optim.SGD(model.parameters(),lr=0.0001,momentum=0.9,weight_decay=5e-4)
Threshold_val = 0.6
criterion = torch.nn.BCEWithLogitsLoss()


# 參考網址 https://www.youtube.com/watch?v=AZr64OxshLo
# 可以了解IoU和Dice Coefficient兩者的意義

# IoU計算, 使用confusion matrix計算出判斷正確的pixel
# 用來當作練習, 實際跑很慢, 所以不用
def compute_iou(y_pred, y_true):
    # y_true和y_pred都是矩陣(mask), 只有1和0, flatten後變成一維向量
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    # 全稱 sklearn.metrics.confusion_matrix
    # 用confusion matrix => 後來可用來計算判斷正確的pixel數量(true positive)
    # 因為是是confusion matrix的輸入是布林矩陣, 故回傳為2x2矩陣 
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # compute mean iou
    # np.diag()根據輸入的向量維度, 輸出也會不同. 若一維, 則回傳對角項是輸入的矩陣, 若二維, 則回傳其對角的向量
    # 因current是confusion矩陣, 故回傳其對角值向量, 也就是判斷正確的pixel, 也就是兩mask的交集, 固定義為intersection
    intersection = np.diag(current)
    # confusion_matrix .sum(axis=1)和.sum(axis=0) => 0為垂直, 1為水平
    # confusion matrix
    #      pred 
    # true      0    1
    #      0    xx   xx
    #     
    #      1    xx   xx

    # .sum(axis=1)會得到ground truth中0和1的各自pixel數量
    # .sum(axis=0)則反之
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    
    # 減掉intersection是因為它重複計算了, 這樣可以算出沒交集和交集的總和
    union = ground_truth_set + predicted_set - intersection
    # 並得到IoU
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)


def dice_coeff(true_mask, pred_mask, non_seg_score=1.0):
    assert true_mask.shape == pred_mask.shape
    # 化為布林矩陣
    true_mask = np.asarray(true_mask).astype(np.bool)
    pred_mask = np.asarray(pred_mask).astype(np.bool)

    # 若兩者皆為0, 怎無法算出分數
    img_sum = true_mask.sum() + pred_mask.sum()
    if img_sum == 0:
        return non_seg_score

    # 計算dice coeff
    intersection = np.logical_and(true_mask, pred_mask)
    return 2. * intersection.sum() / img_sum


# training
# 傳入epoch, loader, net, optim開始訓練
def train(epoch,trainloader,Net,optimizer):
    Net.train()
    train_loss = 0.0
    correct_pixel = 0
    total_pixel = 0
 
    for i,(data,target) in tqdm(enumerate(trainloader),total=len(trainloader)):
        data,target = data.to(device) , target.to(device)     #(2,3,512,512)
        pred = Net(data)
        loss = criterion(pred.cuda().float(),target.cuda().float())
        # backward, 更新parameter
        optimizer.zero_grad() # 將累加的grad歸零
        loss.backward()
        optimizer.step()

        # loss.item()用意 => 因為tensor返回值為tensor維度, 加.item()才能取出其值
        # train_loss算是用來記錄loss的值
        train_loss+=loss.item()
        # 將 threshold 化成 tensor
        threshold = torch.tensor([Threshold_val])

        # .cpu()用意 => 轉存到CPU, 延伸:為何要用cpu算
        # pred是一個矩陣, 其值為判斷是否為目標分類的分數
        # 因此若是大於預設的分數值,也就是threshold, 便判斷是目標的分類, 低於就判斷不是
        # 因此會得到一個布林矩陣, pred mask
        pred = (pred.cpu() > threshold.cpu())

        # 計算 accuracy
        total_pixel += target.nelement()  # 計算所有pixel數量
        correct_pixel += pred.eq(target.cpu()).sum().item()  # pred 和 target pixel一致的數量(包含背景)
        train_acc = correct_pixel / total_pixel
        #mean_iou = compute_iou(y_pred=pred.cpu(), y_true=target.cpu())
        Dice = dice_coeff(true_mask=target.cpu(), pred_mask=pred.cpu())

        # 顯示訓練過程
        verbose_step = len(trainloader) // 10
        if i % (verbose_step + 1) == 0:
            print('')
            print('epoch:', epoch)
            print('loss_avg:', train_loss / (i + 1))  # 到第i個batch loss的平均
            #print('miou:', mean_iou)
            print('Dice loss:', 1 - Dice)  # dice loss的分數
            print('overall_acc:', train_acc)


# validation
def val(epoch,validationloader,Net):
    Net = Net.eval()
    train_loss = 0.0
    correct_pixel = 0
    total_pixel = 0
    # 因為驗證不需要改變其grad, 故在no_grad下進行, 就不會累計其grad
    # 過程和在 training 時計算一樣, 不過少了更新參數的過程
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(validationloader), total=len(validationloader)):
            data, target = data.to(device), target.to(device)
            pred = Net(data)  # (3,1,512,512)
            loss = criterion(pred.cuda().float(), target.cuda().float())
            train_loss += loss.item()
            threshold = torch.tensor([Threshold_val])
            pred = (pred.cpu() > threshold.cpu())
            total_pixel += target.nelement()  # 計算所有pixel數量
            correct_pixel += pred.eq(target.cpu()).sum().item()  # pred 和 target pixel一致的數量(包含背景)

            train_acc = correct_pixel / total_pixel
            # mean_iou = compute_iou(y_pred=pred.cpu(), y_true=target.cpu())
            Dice_Coeff = dice_coeff(true_mask=target.cpu(), pred_mask=pred.cpu())
    print('')
    print('val_loss_avg:', train_loss / (len(validationloader)))
    # print('val_miou:', mean_iou)
    print('val_DiceLoss:', 1 - Dice_Coeff )
    print('val_overall_acc:', train_acc)





# start

# dir_path = os.path.dirname(__file__)
# 發現新的方法 getcwd, 一樣是用來取得當前目錄, 與上面那個有一點差異
root_dir = os.getcwd() 
train_dir = os.path.join(root_dir, "vertebral","f01")
val_dir = os.path.join(root_dir, "vertebral","f02")

# augmentation
my_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((512,512)),
    transforms.RandomRotation(degrees=10),
    #transforms.RandomCrop(500,500),
    transforms.ToTensor()
])

# 讀取train和validation的資料
train_dataset = imageDataset(train_dir, augmentation = my_transform, classes=CLASSES)
val_dataset = imageDataset(val_dir, classes=CLASSES)
TrainingLoader = DataLoader(train_dataset, batch_size=2, shuffle=True)
ValidationLoader = DataLoader(val_dataset, batch_size=2, shuffle=False)
# 若想控制取出量, 調整batch_size就好

Epoch = 10
# main program
if __name__ =='__main__':
    print(len(TrainingLoader))
    for epoch in tqdm(range(Epoch),total=Epoch):
        print('epoch:',epoch)
        train(epoch=epoch,trainloader = TrainingLoader,Net = model,optimizer = optimizer)
        #if (epoch+1)%40 ==0:
            #learning_rate = learning_rate/10
        #val(epoch=epoch,validationloader=ValidationLoader,Net= Net)