# -*- coding: utf-8 -*-
# @Time : 2022/7/21 15:40
# @Author : wutianmin

import pandas as pd
import numpy as np
from astropy.io import fits
import os

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, label_binarize
from matplotlib import pyplot as plt
from scipy import interpolate
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import Conv1d, MaxPool1d, Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as Data
from sklearn import metrics
import seaborn as sns
from astropy.io import fits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


hdulist = fits.open(r'E:\output\train_data_10.fits')
# 3000个pixel
# 字节顺序与系统的本机字节顺序不匹配
flux = hdulist[0].data.byteswap().newbyteorder()
objid = hdulist[1].data['objid'].byteswap().newbyteorder()
label = hdulist[1].data['label'].byteswap().newbyteorder()
wavelength = np.linspace(3900,9000,3000)
#
for i in range(flux.shape[0]):
    flux[i,:] = (flux[i,:]-min(flux[i,:]))/(max(flux[i,:])-min(flux[i,:]))

# label = torch.nn.functional.one_hot(torch.tensor(label, dtype=torch.long))
#c = {0:'GALAXY',1:'QSO',2:'STAR'}
flux_tensor = torch.tensor(flux, dtype=torch.float32)
label_tensor = torch.tensor(label, dtype=torch.long)
comp_file = Data.TensorDataset(flux_tensor, label_tensor)


train_set, val_test_set = train_test_split(comp_file, test_size=0.3, random_state=30,shuffle=True)
val_set, test_set = train_test_split(val_test_set, test_size=0.3, random_state=30,shuffle=True)
train_data_size = len(train_set)
val_data_size = len(val_set)
print('训练数据集的长度为：{}'.format(train_data_size))
print('验证数据集的长度为：{}'.format(val_data_size))

BATCH_SIZE = 128

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

# CNN model
class cnnmodel(nn.Module):
    def __init__(self):
        super(cnnmodel, self).__init__()
        self.model1 = Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Flatten(),
            nn.Linear(128*373, 128),
            nn.Linear(128, 3),
            )

    def forward(self, x):
        x = self.model1(x)
        return x

cnnm = cnnmodel()
cnnm = cnnm.to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

learning_rate = 0.01# 学习率
optim = torch.optim.SGD(cnnm.parameters(), lr=learning_rate)
scheduler = StepLR(optim, step_size=30, gamma=0.1)

def train(EPOCHS, min_loss):

    writer = SummaryWriter(r'E:\aafile\pythonProject5\LLM_train\logstest2')
    count_loss = 0# 储存精度未降低的次数
    k = 0# 储存总训练次数
    train_lossd = []# 储存train_loss
    val_lossd = []# 储存val_loss
    for epoch in range(EPOCHS):

        train_loss = 0.0
        total_train_step = 0
        con_train = 0
        acc_train = 0.0
        # 训练步骤开始
        cnnm.train()
        for batch_idx, (spec, target) in enumerate(train_dataloader):
            spec = torch.reshape(spec, (-1, 1, 3000))
            spec = spec.to(device)
            target = target.to(device)
            outputs = cnnm(spec)
            optim.zero_grad()
            loss = loss_fn(outputs, target)
            loss.backward()
            optim.step()

            total_train_step = total_train_step + 1# 训练次数加一
            train_loss = train_loss + loss.item()# 总体训练损失

            con_train += len(target)
            _, predict_train = torch.max(outputs, 1)
            this_acc_train = (target == predict_train).sum().item()
            acc_train += this_acc_train

        train_loss = train_loss/total_train_step # 平均训练损失

        accuracy_train = acc_train / con_train

        # 验证步骤开始
        total_test_step = 0.0
        val_loss = 0.0
        con = 0
        acc = 0.0
        cnnm.eval()
        with torch.no_grad():  
            for batch_idx, (spec, target) in enumerate(val_dataloader):
                spec = torch.reshape(spec, (-1, 1, 3000))
                spec = spec.to(device)
                target = target.to(device)
                outputs = cnnm(spec)
                loss = loss_fn(outputs, target)

                val_loss = val_loss + loss.item()

                total_test_step += 1

                con += len(target)
                _, predict = torch.max(outputs, 1)
                this_acc = (target == predict).sum().item()

                acc += this_acc

        val_loss = val_loss / total_test_step
        accuracy = acc / con

        scheduler.step()
        # if epoch % 100 == 0:
        print("Epoch:{}/{}, Train Loss:{:.4f}, Train accuracy:{:.4f}"
                  "Val_Loss:{:.4f}, Val accuracy:{:.4f}".format(epoch, EPOCHS,
                                                          train_loss, accuracy_train,val_loss,accuracy))
        if val_loss < min_loss:
            min_loss = val_loss
            count_loss = 0
            torch.save(cnnm, r'E:\aafile\pythonProject5\LLM_train\model_best.pth')
            print('模型已保存')
        else:
            count_loss = count_loss + 1
        print(count_loss)
        if count_loss >=100:
            print('提前停止训练')
            break
        k = k + 1
        train_lossd.append(train_loss)
        val_lossd.append(val_loss)

        writer.add_scalars('loss', {'trainloss': train_loss, 'valloss': val_loss}, epoch)

    writer.close()

    xs=np.arange(k)
    plt.plot(xs, train_lossd, label='trainloss')
    plt.plot(xs, val_lossd, label='valloss')
    plt.legend()
    plt.savefig(r'E:\aafile\pythonProject5\LLM_train\loss_best.png')

def test(model_path):

    model = torch.load(model_path)
    model = model.to(device)
    y_true = []
    y_scores = []
    acc = 0.0
    with torch.no_grad():
        for i in range(len(test_set)):
            spec = test_set[i][0]
            spec = torch.reshape(spec, (-1, 1, 3000))
            target = test_set[i][1]

            spec = spec.to(device)
            target = target.to(device)

            outputs = model(spec)
            _, predict = torch.max(outputs, 1)
            this_acc = (target == predict).sum().item()
            acc += this_acc

            y_true.append(target.item())
            y_scores.extend(predict.cpu().numpy())
    accuracy = acc / len(test_set)
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    print(accuracy)
    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_scores,normalize='true'
                                    )
    label_name = ['GALAXY','QSO','STAR']
    plt.imshow(cm, cmap='Blues')
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    plt.savefig('confusionmatrix.png')


if __name__ == '__main__':
    min_loss = 1000000
    # 训练轮数
    EPOCHS = 1000  # 1000次
    train(EPOCHS, min_loss)
    model_path = r'E:\aafile\pythonProject5\LLM_train\model_best.pth'
    test(model_path)
