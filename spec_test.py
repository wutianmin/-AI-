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

model1 = torch.load(r'E:\aafile\pythonProject5\LLM_train\best_spec_model2.pth')
model1 = model1.to(device)


hdulist = fits.open(r'E:\output\test_data.fits')
# 3000个pixel
# 字节顺序与系统的本机字节顺序不匹配
flux = hdulist[0].data.byteswap().newbyteorder()
objid = hdulist[1].data['objid'].byteswap().newbyteorder()
wavelength = np.linspace(3900,9000,3000)

for i in range(flux.shape[0]):
    flux[i,:] = (flux[i,:]-min(flux[i,:]))/(max(flux[i,:])-min(flux[i,:]))

flux_tensor = torch.tensor(flux, dtype=torch.float32)

outputsd = []

for i in range(flux.shape[0]):
    flux_tt = torch.reshape(flux_tensor[i], (-1, 1, 3000))
    flux_tt = flux_tt.to(device)  # 用GPU训练

    outputs = model1(flux_tt)

    _, predict = torch.max(outputs, 1)
    outputsd.append(predict.item())
    print(predict)

df_output = pd.concat((pd.DataFrame(objid), pd.DataFrame(outputsd)), axis=1)
df_output.columns=['objid','label']
df_output.to_csv(r'E:\aafile\pythonProject5\LLM_train\test_data_with_label.csv', index=False)