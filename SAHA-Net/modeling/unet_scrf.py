import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from daily_test.practical_function import extract_image_patches
f = torch.cuda.is_available()
device = torch.device("cuda" if f else "cpu")

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet_scrf(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet_scrf, self).__init__()

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

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x,fore_mask,back_mask):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        up_6 = self.up6(c5)
        # print(up_6.size(),c4.size())
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
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)

        loss1_map = -1 * (fore_mask * torch.log(out + 0.0001) + (1 - fore_mask) * torch.log(1 - out + 0.0001))
        loss1 = loss1_map.sum()
        #计算空间限制损失
        out_argmax = torch.argmax(torch.stack((1 - out, out)), dim=0).to(torch.float32)
        x_patches = extract_image_patches(x,3,1,1)
        out_patches = extract_image_patches(out,3,1,1)
        out_argmax_patches = extract_image_patches(out_argmax,3,1,1)

        u_ab = torch.exp(-1 * torch.pow(x - x_patches,2) / (2*0.5*0.5))
        u_signal = 2 * (out_argmax == out_argmax_patches).to(torch.float32) - 1
        u_ij = u_ab * u_signal
        p_ij = out * out_patches
        f_ij = u_ij * p_ij
        f_i = (f_ij.sum(dim=1,keepdim=True) - out**2) / (u_ab.sum(dim=1,keepdim=True) - 1 + 1e-9)#由于这种方法会多一个自己跟自己乘，此时u_ij=0,所以要记得减去
        loss2_map = 1-f_i
        loss2 = loss2_map.sum()#可以反向传播



        # print(loss.item())
        # unc_max_rect = unc_max * fore_mask
        #
        # # unc_rec_max.requires_grad=True
        # # out_resize = out.unsqueeze(2)#b*1*1*256*256
        # # out_cat = torch.cat([1-out_resize,out_resize],dim=2)#b*1*2*256*256
        # # out_argmax = torch.argmax(out_cat,dim=2)#b*2*256*256
        # # out_argmax = torch.argmax(torch.stack((1-out,out)),dim=0)
        # out_softmax = F.softmax(torch.stack((10*(1-out),10 * out)),dim=0)#2*b*1*256*256
        # out_argmax = out_softmax[1,]#b*1*256*256#可以反向传播
        # # out_argmax.requires_grad=True
        # refine_argmax = out_argmax * fore_mask
        # print(refine_argmax.size(),refine_unc_thres.size())

        return out,loss1,loss2

# a = np.random.randn(8,1,1,256,256)
# b = 1-a
# c = np.concatenate([b,a],axis=2)
# d = np.argmax(c,axis=2)
# e = a.max(dim=-2,keepdim=True)[0]
# print(e.shape)
# a = torch.randn(8,1,256,256)
# b = torch.argmax(torch.stack((1-a,a)),dim=0)
# print(torch.unique(b))