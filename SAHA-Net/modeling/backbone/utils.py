import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):#传入的是b*1*w*h
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, pred, target):
        num = target.size(0)
        # probs = torch.sigmoid(pred)
        probs = pred
        m1 = probs.view(num, -1)#第一个维度代表batch_size，其余维度拍成1
        m2 = target.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)#加1只是未为了防止其等于0
        score = 1 - score.sum() / num
        return score#loss是tensor可以反向传播，不用转换为标量

def soft_dice(pred, target):
    loss_f = SoftDiceLoss()
    return loss_f(pred, target)