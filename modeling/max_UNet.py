import torch.nn as nn
import torch
from torch import autograd
from functools import partial
import torch.nn.functional as F
from torchvision import models
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),#padding=1保证在3*3卷积下分辨率不变
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class max_Unet(nn.Module):#输入的必须是四维的tensor：batch_size*classes*w*h
    def __init__(self, in_ch, out_ch):
        super(max_Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, out_ch, 1)
        self.x_ones = nn.Conv2d(1,1,(256,1))
        self.x_ones.weight = nn.Parameter(data=torch.ones((1,1,256,1)),requires_grad=True)#第一个参数表示输出通道数
        self.y_ones = nn.Conv2d(1,1,(1,256))
        self.y_ones.weight = nn.Parameter(data=torch.ones((1,1,1,256)), requires_grad=True)

    def forward(self, x):
        #print(x.shape)
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        #print(p1.shape)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        #print(p2.shape)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        #print(p3.shape)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        #print(p4.shape)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)#1*1*256*256
        # x_softmax = F.softmax(c10,dim=1)#每行最大值接近1，其余值接近0
        # y_softmax = F.softmax(c10, dim=0)#每列最大值接近1，其余值接近0
        # out = nn.Sigmoid()(c10)
        # x_map = x_softmax * out
        # y_map = y_softmax * out
        # out_x = self.x_ones(y_map)
        # out_y = self.y_ones(x_map)
        out = nn.Sigmoid()(c10)
        out_x = out.max(dim=2,keepdim=True)[0]
        out_y = out.max(dim=3, keepdim=True)[0]
        assert out_x.max() > 1 or out_x.min() < 0, "label error max{} min{}".format(out_x.max(), out_x.min())

        return out_x,out_y,out
# a = torch.ones(1,1,256,256).to(torch.device("cuda"))
# model = semi_Unet(1,1).to(torch.device("cuda"))
# b,c,d = model(a)
# print(b.size(),c.size(),d.size())
# a = np.array([[1,2],[1,2]])
# b= np.random.rand(2,1)
# print(torch.ones(1,3))
# print(a)
# x_ones = nn.Conv2d(1,1,(256,1))
# x_ones.weight = nn.Parameter(data=torch.ones((256,1)),requires_grad=True)
# print(x_ones.weight.size())
# a =torch.ones(1,1,256,256)
# b=a*a
# print(b.size())
# a = torch.ones(1,1,3,3)
# b = a.max(dim=2,keepdim=True)
# print(a.size())
# print(b[0])
# print(b[0].size())
# x=torch.randn(1,1,3,4)
# print(x.max(dim=3,keepdim=True)[0].size())