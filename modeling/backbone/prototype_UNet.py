import torch.nn as nn
import torch
from utils import *
from torch import autograd
from functools import partial
import torch.nn.functional as F
from torchvision import models
import numpy as np
device = torch.device("cuda")

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


class proto_Unet(nn.Module):#输入的必须是四维的tensor：batch_size*classes*w*h
    def __init__(self, in_ch, out_ch):
        super(proto_Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        self.x_ones = nn.Conv2d(1, 1, (256, 1))
        self.x_ones.weight = nn.Parameter(data=torch.ones((1, 1, 256, 1)), requires_grad=True)  # 第一个参数表示输出通道数
        self.y_ones = nn.Conv2d(1, 1, (1, 256))
        self.y_ones.weight = nn.Parameter(data=torch.ones((1, 1, 1, 256)), requires_grad=True)

    def forward(self, x,fore_mask,back_mask):
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
        c9 = self.conv9(merge9)#1*32*256*256

        merge_feature = torch.cat([c6,c7,c8,c9],dim=1)
        c10 = self.conv10(c9)#1*1*256*256
        out_norm = nn.Sigmoid()(c10)
        #out_norm = (out>=0.5).float()#该操作无法反向传播
        # refine_mask = out * rec_label
        # fore_onehot = (refine_mask>=0.5).float()#1*1*256*256
        # fore_num = fore_onehot.sum()
        # fore_onehots = fore_onehot.repeat(1,32,1,1)#1*32*256*256
        # back_onehot = (refine_mask<0.5).float()
        # back_onehots = back_onehot.repeat(1, 32, 1, 1)
        # back_num = back_onehots.sum()
        fore_onehots = fore_mask.repeat(1, 32, 1, 1)
        back_onehots = back_mask.repeat(1, 32, 1, 1)
        fore_num = fore_onehots.sum()
        back_num = back_onehots.sum()
        fore_protomap = c9 * fore_onehots
        back_protomap = c9 * back_onehots
        fore_prototype = fore_protomap.sum(dim=(-2,-1),keepdim=True) / (fore_num + 1)#前景原型1*32*1*1
        back_prototype = back_protomap.sum(dim=(-2, -1), keepdim=True) / (back_num + 1)#背景原型
        # fore_prototy = nn.BatchNorm2d(32)(fore_proto).to(device)
        # back_prototy = nn.BatchNorm2d(32)(back_proto).to(device)
        #fore_prototype = nn.Sigmoid()(fore_proto)
        #back_prototype = nn.Sigmoid()(back_proto)
        fore_prototypes = fore_prototype.repeat(1, 1, 256, 256)#1*32*256*256
        back_prototypes = back_prototype.repeat(1, 1, 256, 256)
        fore_distance = ((c9 - fore_prototypes) * (c9 - fore_prototypes)).sum(dim=1,keepdim=True)#1*1*256*256
        back_distance = ((c9 - back_prototypes) * (c9 - back_prototypes)).sum(dim=1, keepdim=True)
        out_proto = back_distance / (fore_distance + back_distance)

        criterion = 0.5 * torch.nn.BCELoss() + 0.5 *
        #back_acti = c9 * back_prototypes
        #fore_activate = fore_acti.sum(dim=1,keepdim=True)#1*1*256*256
        #back_activate = back_acti.sum(dim=1, keepdim=True)
        #prob_map = fore_activate / (fore_activate+back_activate)#每个像素值均为该像素是前景类别的概率
        # x_softmax = F.softmax(c10,dim=1)#每行最大值接近1，其余值接近0
        # y_softmax = F.softmax(c10, dim=0)#每列最大值接近1，其余值接近0
        # out = nn.Sigmoid()(c10)
        # x_map = x_softmax * out
        # y_map = y_softmax * out
        # out_x = self.x_ones(y_map)
        # out_y = self.y_ones(x_map)

        out_x = out_norm.max(dim=2,keepdim=True)[0]
        out_y = out_norm.max(dim=3, keepdim=True)[0]
        loss_norm_x = criterion(norms_x, labels_x)  # SAC分支
        loss_norm_y = criterion(norms_y, labels_y)
        loss_norm = loss_norm_x + loss_norm_y

        return out_x,out_y,out_norm,out_proto
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
# 指定 tensor 的形状和元素值
# x = torch.randn(8, 32, 256, 256)
#
# # 在第二个维度上进行元素求和
# y = x.sum(dim=1, keepdim=True)
#
# print(y.shape)