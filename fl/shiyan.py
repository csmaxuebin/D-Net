from turtle import pd

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import matplotlib

from models.UpDateNumal import LocalUpdateNumal

matplotlib.use('Agg')

import torch

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from models.Nets import SimpleCNN
from models.test import test_img
from utils.options import args_parser


class MinstDataSetFromImages(Dataset):
    # 1. 初始化文件路径或文件名列表。
    # 也就是在这个模块里，我们所做的工作就是初始化该类的一些基本参数。
    def __init__(self, csv_path, transforms=None):
        self.transforms  = transforms
        self.data_info = pd.read_csv(csv_path,header=None)
        self.image_arr = np.asarray(self.data_info.iloc[:,0])
        self.label_arr = np.asarray(self.data_info.iloc[:,1])
        self.data_len = len(self.data_info.index)

    # 这里需要注意的是，第一步：read one data，是一个data
    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        # prefix = "E:\\code\\federated-learning\\data\\mnist\\"
        prefix = "E:\\code\\federated-learning\\data\\ct\\"
        single_image_name = prefix+single_image_name.strip("./")

        img_as_img = Image.open(single_image_name).convert('RGB')
        #img_as_img = cv2.imread(single_image_name)
        image_as_tensor = self.transforms(img_as_img)

        single_image_name = self.label_arr[index]
        return(image_as_tensor, single_image_name)

    def __len__(self):
        return self.data_len

if __name__ == '__main__':
    # m = max(int(0.1 * 100), 1)
    # idxs_users = np.random.choice(range(100), m, replace=False)
    # print(idxs_users)
    # dict_user=dict()
    # for idx in range (len(idxs_users)):
    #     print("2222222",idx)
    #     dict_user[idxs_users[idx]]=idx
    #
    # for key,value in dict_user.items():
    #     print("33333333333333",key)
    #     print(dict_user[key])


    # for iter in range(2):
    #     if iter == 0:
    #         print("111111111111111111111111111111111111")
    #         for idx in idxs_users:
    #             print("idx",idx)
    #     else:
    #         print("2222222222222222222222222222222222222222")
    #         for idx in idxs_users:
    #             print("idx",idx)
    #



    # loss_train=[]
    # acc_train=[]
    # acc=3
    # loss=1
    # for idx in range (10):
    #     loss_train.append(loss)
    #
    # for idx in range (10):
    #     acc_train.append(acc)
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig('./save/new_loss.png')
    #
    # # plot acc curve
    # plt.figure()
    # plt.plot(range(len(acc_train)), acc_train)
    # plt.ylabel('train_acc')
    # plt.savefig('./save/new_acc.png')
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    minst_train_from_csv = MinstDataSetFromImages("E:/code/federated-learning/data/ct/train.csv",
                                                  transforms=train_transformer)
    net_glob = SimpleCNN().to(args.device)



    acc_train, loss_train = test_img(net_glob, minst_train_from_csv, args)