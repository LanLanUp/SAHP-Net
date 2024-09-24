import matplotlib.pyplot as plt
import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
from skimage.io import imread
import cv2
from glob import glob
import imageio
from PIL import Image
import numpy as np
import torch
import json

f = torch.cuda.is_available()
device = torch.device("cuda" if f else "cpu")


class TrainDataset(data.Dataset):#
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'/data/lingeng/TN3K/train'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths = None, None
        self.train_mask_paths, self.val_mask_paths = None, None
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.train_img_paths = glob(self.root + r'/image/*')
        self.train_img_paths = sorted(self.train_img_paths,key=lambda x: int(x.split('/')[-1].split('.')[0]))
        # self.train_img_paths = sorted(self.train_img_paths)

        self.train_mask_paths = glob(self.root + r'/mask/*')
        # self.train_mask_paths = sorted(self.train_mask_paths, key=lambda x: int(x.split('/')[-1].split('_')[0]))
        # self.train_mask_paths = sorted(self.train_mask_paths)
        self.train_mask_paths = sorted(self.train_mask_paths, key=lambda x: int(x.split('/')[-1].split('_')[0]))#先排数字，排序是若两个对象优先级一样则不改变其相对位置
        self.train_mask_paths = sorted(self.train_mask_paths,key=lambda x: int(x.split('/')[-1].split('_')[0]) + ord(x.split('_')[-1].split('.')[0]))


        # self.val_img_paths = glob(self.root + r'/614_image/*')
        # self.val_img_paths = sorted(self.val_img_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        #
        # self.val_mask_paths = glob(self.root + r'/614_resize_mask/*')
        # self.val_mask_paths = sorted(self.val_mask_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        # self.val_mask_paths = sorted(self.val_mask_paths, key=lambda x: int(x.split('/')[-1].split('_')[0]))
        # self.val_mask_paths = sorted(self.val_mask_paths, key=lambda x: int(x.split('/')[-1].split('_')[0]) + ord(x.split('_')[-1].split('.')[0]))

        # self.img_paths = glob('/home/lingeng/Thyroid Dataset/tg3k/sat_0.5/image/*')
        # self.img_paths = sorted(self.img_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        # self.mask_paths = glob('/home/lingeng/Thyroid Dataset/tg3k/sat_0.5/mask/*')
        # self.mask_paths = sorted(self.mask_paths, key=lambda x: int(x.split('/')[-1].split('_')[0]))
        # self.mask_paths = sorted(self.mask_paths, key=lambda x: int(x.split('/')[-1].split('_')[0]) + ord(
        #     x.split('_')[-1].split('.')[0]))


        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        # if self.state == 'val':
        #     return self.val_img_paths, self.val_mask_paths
        # if self.state == 'test':
        #     #return test_img_paths, test_mask_paths
        #     return self.val_img_paths, self.val_mask_paths  # 因数据集没有测试集，所以用验证集代替

    def __getitem__(self, index):
        pic_path = self.pics[index]
        pic = cv2.imread(pic_path,0)
        pic_org = Image.fromarray(pic)
        pic = np.array(pic_org.resize((256, 256), Image.BICUBIC))
        pic = pic.astype('float32') / 255

        # pic = np.array(Image.open(pic_path)).astype('float32') / 255

        # mask_path = self.masks[index]
        # mask = cv2.imread(mask_path, 0)
        # mask_org = Image.fromarray(mask)
        # mask = np.array(mask_org.resize((256, 256), Image.BICUBIC))
        # mask = mask.astype('float32') / 255

        # mask = np.array(Image.open(mask_path)).astype('float32') / 255
        #pic = cv2.imread(pic_path)
        #mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)

        label_x_path = self.masks[2 * index]
        label_y_path = self.masks[2 * index + 1]
        # print(label_x_path)
        # print(label_y_path)
        # label_x = cv2.imread(label_x_path, 0)
        # label_y = cv2.imread(label_y_path, 0)
        # label_x = label_x.astype('float32') / 255
        # label_y = label_y.astype('float32') / 255
        # print(label_x.shape,label_y.shape)
        label_x = np.array(Image.open(label_x_path)).astype('float32') / 255
        label_y = np.array(Image.open(label_y_path)).astype('float32') / 255
        mask = np.dot(label_y, label_x)

        fore_mask = mask
        back_mask = (mask<0.5).astype('float32')

        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(fore_mask)
            back_mask = self.target_transform(back_mask)
            lab_x = self.target_transform(label_x)
            lab_y = self.target_transform(label_y)
            # rec_label = self.target_transform(rec_label)
        #return img_x, img_y, pic_path, mask_path
        #return img_x, lab_x, lab_y,rec_label
        return img_x, img_y, back_mask,lab_x,lab_y

    def __len__(self):
        return len(self.pics)

class ValDataset(data.Dataset):#
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        # self.root = r'/home/lingeng/weak_supervise/saved_model/xiaorong'
        self.root = r'/data/lingeng/TN3K/val'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths = None, None
        self.train_mask_paths, self.val_mask_paths = None, None
        self.detect_predict_paths = None
        self.pics, self.masks, self.detects = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):

        # self.val_img_paths = glob(self.root + r'/image/*')
        self.val_img_paths = glob(self.root + r'/614_image/*')
        # self.val_img_paths = sorted(self.val_img_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))

        # self.val_mask_paths = glob(self.root + r'/mask/*')
        self.val_mask_paths = glob(self.root + r'/614_pixel_mask/*')
        # self.val_mask_paths = sorted(self.val_mask_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        self.detect_predict_paths = glob(r'/data/lingeng/TN3K/val/614detect_predict/*')

        return self.val_img_paths, self.val_mask_paths,self.detect_predict_paths  # 因数据集没有测试集，所以用验证集代替

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        detect_path = self.detects[index]

        pic = cv2.imread(pic_path,0)
        # pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,0)
        detect_predict = cv2.imread(detect_path,0)

        pic_org = Image.fromarray(pic)
        mask_org = Image.fromarray(mask)
        detect_org = Image.fromarray(detect_predict)

        pic = np.array(pic_org.resize((256, 256),Image.BICUBIC))
        mask = np.array(mask_org.resize((256, 256), Image.BICUBIC))
        detect = np.array(detect_org.resize((256, 256), Image.BICUBIC))

        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        detect = (detect>125).astype('float32')

        # pic = np.array(Image.open(pic_path)).astype('float32') / 255
        # mask = np.array(Image.open(mask_path)).astype('float32') / 255

        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
            t = self.target_transform(detect)
        return img_x, img_y,t, pic_path, mask_path

    def __len__(self):
        return len(self.pics)
