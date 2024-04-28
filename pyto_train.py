import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import numpy as np
from astropy.io import fits
import os
from sklearn.preprocessing import MinMaxScaler
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

#9:1训练集测试集
train_set, val_set = train_test_split(comp_file, test_size=0.1, shuffle=True)
train_data_size = len(train_set)
val_data_size = len(val_set)
print('训练数据集的长度为：{}'.format(train_data_size))
print('验证数据集的长度为：{}'.format(val_data_size))

BATCH_SIZE = 256

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)


# 构建一个简单的神经网络模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# 准备数据集（假设有一个二维特征）
X = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
y = torch.tensor([[0], [0], [1], [1]])

# 定义模型和损失函数
input_dim = 3000#X.shape[1]
model = SimpleClassifier(input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    model.train()  # 声明训练
    for batch_idx, (spec, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(spec)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    model.eval()
    # 模型评估
    with torch.no_grad():
        for batch_idx, (spec, target) in enumerate(val_dataloader):
            predicted = model(spec).round()
            accuracy = (predicted == target).sum().item() / len(y)
            print(f'Accuracy: {accuracy:.2f}')