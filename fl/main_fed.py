#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
from models.UpDateNumal import LocalUpdateNumal
from models.auc_roc_zhi import shuzhi
from testing1 import test_img1
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, SimpleCNN
from models.Fed import FedAvg
from models.test import test_img
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import os
import time
import random
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#有病�?，没病是1
#写自己的训练�?
class MinstDataSetFromImages(Dataset):
    # 1. 初始化文件路径或文件名列表�?    # 也就是在这个模块里，我们所做的工作就是初始化该类的一些基本参数�?    
    def __init__(self,csv_path,transforms=None):
        self.transforms  = transforms
        self.data_info = pd.read_csv(csv_path,header=None)
        self.image_arr = np.asarray(self.data_info.iloc[:,0])
        self.label_arr = np.asarray(self.data_info.iloc[:,1])
        self.data_len = len(self.data_info.index)

    # 这里需要注意的是，第一步：read one data，是一个data
    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        # prefix = "E:\\code\\federated-learning\\data\\mnist\\"
        # prefix = "E:\\code\\federated-learning\\data\\ct\\"
        prefix = "E:/code/fl/data/mnist/"
        single_image_name = prefix+single_image_name.strip("./")

        img_as_img = Image.open(single_image_name).convert('RGB')
        #img_as_img = cv2.imread(single_image_name)
        image_as_tensor = self.transforms(img_as_img)

        single_image_name = self.label_arr[index]
        return(image_as_tensor, single_image_name)

    def __len__(self):
        return self.data_len

#写自己的共享数据�?
class MinstDataSetShareFromImages(Dataset):
    # 1. 初始化文件路径或文件名列表�?    # 也就是在这个模块里，我们所做的工作就是初始化该类的一些基本参数�?    
    def __init__(self, sa,csv_path, transforms=None):
        self.transforms  = transforms
        self.data_info = pd.read_csv(csv_path,header=None)
        self.image_arr = np.asarray(self.data_info.iloc[0:int(sa*len(self.data_info.index))+1,0])
        self.label_arr = np.asarray(self.data_info.iloc[0:int(sa*len(self.data_info.index))+1,1])
        self.data_len = int(sa*len(self.data_info.index))
        #print(self.data_len,self.image_arr.shape)

    # 这里需要注意的是，第一步：read one data，是一个data
    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        prefix = "./data/ct/"
        single_image_name = prefix+single_image_name.strip("./")

        img_as_img = Image.open(single_image_name).convert('RGB')
        #img_as_img = cv2.imread(single_image_name)
        image_as_tensor = self.transforms(img_as_img)

        single_image_name = self.label_arr[index]
        return(image_as_tensor, single_image_name)

    def __len__(self):
        return self.data_len



if __name__ == '__main__':
    # parse args
    start=time.time()
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    acc=0.0

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    val_transformer = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # load dataset and split users
    #if args.dataset == 'mnist':
    #
    #
    #定义训练�?
    # transformations = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    # minst_train_from_csv = MinstDataSetFromImages("/home/ubuntu/fl/data/ct/train.csv", transforms=train_transformer)

        #定义测试�?    # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        #dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = MinstDataSetFromImages('E:/code/fl/data/mnist/test.csv', transforms=val_transformer)

        # sample users
        # if args.iid:
    # dict_users = mnist_iid(minst_train_from_csv, args.num_users)
        # else:
        #     dict_users = mnist_noniid(minst_train_from_csv, args.num_users)
    # elif args.dataset == 'cifar':
    #     trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #     dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    #     dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    #     if args.iid:
    #         dict_users = cifar_iid(dataset_train, args.num_users)
    #     else:
    #         exit('Error: only consider IID setting in CIFAR10')
    # else:
    #     exit('Error: unrecognized dataset')
    # img_size = minst_train_from_csv[0][0].shape

    # build model
    # if args.model == 'cnn' and args.dataset == 'cifar':
    #     net_glob = CNNCifar(args=args).to(args.device)
    # elif args.model == 'cnn' and args.dataset == 'mnist':

    #自带模型
    # net_glob = CNNMnist(args=args).to(args.device)

    #CT自带模型
    # model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    # net_glob = model.cpu()

    #CT自带模型2
    net_glob = SimpleCNN().to(args.device)


    # elif args.model == 'mlp':
    #     len_in = 1
    #     for x in img_size:
    #         len_in *= x
    #     net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    # else:
    #     exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    auc_train=[]
    auc_test=[]
    pr_train=[]
    pr_test=[]
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # if args.all_clients:
    #     print("Aggregation over all clients")
    #     w_locals = [w_glob for i in range(args.num_users)]
    acc = 0.0
    auc=0.0
    pr=0.0
    acc_totl = 0.0
    acc_avg = 0.0
    # acc_user=[]
    acc_max=0
    acc_min=0
    acc_train=[]


    # 创建客户�?    
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    print(idxs_users)
    # 创建空字�?    
    dict_user = dict()

          

    user_acc = []
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        # user_acc = []

        #训练
        if iter == 0 :
            print("iter_epoch",iter)
            i=0

            for idx in idxs_users:
                print("idx",idx)
                net = copy.deepcopy(net_glob).to(args.device)
                w, loss ,acc= LocalUpdateNumal(args,i,net)
                i=i+1

                #对数值加到数组里�?                
                user_acc.append((copy.deepcopy(acc)))

                #字典中添加键值对
                dict_user[idx] = acc
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
                # update global weights
            w_glob = FedAvg(w_locals)
            #利用数组计算数�?            #计算均�?            
            acc_avg=sum(user_acc)/len(user_acc)
            acc_train.append(acc_avg)
            # #计算最大�?            # acc_max=np.max(user_acc)
            #计算最小�?            
            acc_min=np.min(user_acc)
            # acc_avg=(acc_totl/len(idxs_users))
            # print('aaaaaaaaaaaaaaaa',acc_avg)
            loss_avg = sum(loss_locals) / len(loss_locals)


            loss_train.append(loss_avg)


            # print(user_acc)
            #更新user的acc
            for key,value in dict_user.items():
                # print("33333333333333", key)
                if dict_user[key]>=acc_avg:
                    dict_user[key]=0
                    # print(dict_user[key])
                else:
                    temp=dict_user[key]
                    if acc_avg==acc_min:
                        dict_user[key]=1
                    else:
                        dict_user[key]=(acc_avg-temp)/(acc_avg-acc_min)
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            print('Round {:3d}, Average acc {:.3f}'.format(iter, acc_avg))

        else:
            print("iter_epoch",iter)
            i=0

            for idx in idxs_users:
                print("idx", idx)
                ture_acc=dict_user[idx]
                minst_share_from_csv1 = MinstDataSetShareFromImages(ture_acc,
                                                                    "E:/code/fl/data/mnist/share.csv",
                                                                    transforms=train_transformer)
                net1 = copy.deepcopy(net_glob).to(args.device)
                w, loss ,acc= LocalUpdate(args,i,minst_share_from_csv1,net1)
                i=i+1
                # print("acc_idx",acc)
                user_acc.append((copy.deepcopy(acc)))
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
                # update global weights
            w_glob = FedAvg(w_locals)
            #计算均�?            
            acc_avg=sum(user_acc) / len(user_acc)

            # print("acc_avg",acc_avg)
            acc_train.append(acc_avg)
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            print('Round {:3d}, Average acc {:.3f}'.format(iter, acc_avg))
            loss_train.append(loss_avg)
  # testing
    net_glob.eval()
    acc_test, loss_test = test_img1(net_glob, dataset_test, args)
    end=time.time()
    print("code time: {}".format(end-start))
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

