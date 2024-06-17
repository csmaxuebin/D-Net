#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from sklearn.metrics import roc_curve, auc
from torch import nn, autograd

import matplotlib
matplotlib.use('Agg')
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import random
from itertools import cycle
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class DummyDataset(Dataset):
    def __init__(self,a=0,b=100):
        super(DummyDataset,self).__init__()
        self.a=a
        self.b=b
    def __len__(self):
        return self.b-self.a+1

    def __getitem__(self, index):
        return index

#写自己的共享数据集
class MinstDataSetShareFromImages(Dataset):
    # 1. 初始化文件路径或文件名列表。
    # 也就是在这个模块里，我们所做的工作就是初始化该类的一些基本参数。
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
        prefix = "E:\\code\\federated-learning\\data\\ct\\"
        single_image_name = prefix+single_image_name.strip("./")

        img_as_img = Image.open(single_image_name).convert('RGB')
        #img_as_img = cv2.imread(single_image_name)
        image_as_tensor = self.transforms(img_as_img)

        single_image_name = self.label_arr[index]
        return(image_as_tensor, single_image_name)

    def __len__(self):
        return self.data_len




#写自己的训练集
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
        prefix = "/home/SunWenhui/fl/data/ct/"
        single_image_name = prefix+single_image_name.strip("./")

        img_as_img = Image.open(single_image_name).convert('RGB')
        #img_as_img = cv2.imread(single_image_name)
        image_as_tensor = self.transforms(img_as_img)

        single_image_name = self.label_arr[index]
        return(image_as_tensor, single_image_name)

    def __len__(self):
        return self.data_len
#
# #分离数据
# class DatasetSplit(Dataset):
#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = list(idxs)
#
#     def __len__(self):
#         return len(self.idxs)
#
#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         return image, label

#传进来三个参数，分别是args，数据集，dict of image index
def LocalUpdate(args,datatest,net_g):
        #加载本地数据
        # transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        # gv=random.random()

        # minst_train_from_csv = MinstDataSetShareFromImages(gv,"E:/code/federated-learning/data/ct/train.csv",
        #                                               transforms=train_transformer)

        minst_train_from_csv = MinstDataSetFromImages("./data/ct/train.csv",
                                                      transforms=train_transformer)
        data_loader1 = DataLoader(minst_train_from_csv, batch_size=10)

        data_loader2 = DataLoader(datatest, batch_size=10)

        correct = 0

        epoch_correct=0
        net_g.train()
        # train and update
        optimizer = torch.torch.optim.Adam(net_g.parameters(), lr=0.0001)
        epoch_loss = []
        # epoch_roc_auc=[]
        number_epoch=5
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(number_epoch):
            # print("epoch",epoch)
            batch_loss = []
            # batch_loss=0
            # batch_correct=[]
            #每一个的epoch的正确个数
            epoch_correct = 0
            # roc_auc_batch=[]


            # for batch_idx, (images, labels) in enumerate(zip(cycle(data_loader1),data_loader2)):

            # print("开始共享数据集1111111111111111111111111")

            for batch_idx, (images, labels) in enumerate(data_loader1):
                # print("images",images)
                # print("labels",labels)
                batch_correct = 0
                # print("dataloder1")
                # print("batch",batch_idx)
                images, labels = images.to(args.device), labels.to(args.device)
                net_g.zero_grad()
                log_probs = net_g(images)
                # # 计算auc
                # y_scores = log_probs.detach().numpy()
                # y_scores = y_scores[:, 1]  # 关键一步得到scores
                # fpr, tpr, threshold = roc_curve(labels, y_scores,pos_label=0)
                # roc_auc = auc(fpr, tpr)
                # roc_auc_batch.append(roc_auc.item())

                y_pred = log_probs.data.max(1, keepdim=True)[1]
                # batch_correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
                batch_correct += y_pred.eq(labels.data.view_as(y_pred)).long().cuda().sum()

                loss = loss_func(log_probs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                iter, batch_idx * len(images), len(data_loader1),
                                       100. * batch_idx / len(data_loader1), loss.item()))
                batch_loss.append(loss.item())
                # batch_loss+=loss.item()
                # batch_acc.append(y_pred)
                # batch_correct += y_pred.eq(images.data.view_as(y_pred)).long().cpu().sum()
                # print('batch_correct',batch_correct)
                # print("一个batch结束")

                epoch_correct = (batch_correct.item() + epoch_correct)
                # print("dataloder1_correct", epoch_correct)
            #
            # print("开始训练数据集2222222222222222")



            for batch_idx, (images, labels) in enumerate(data_loader2):
                # print("data_loader2")
                # print("images",images)
                # print("labels",labels)
                # print("batch_idx",batch_idx)
                batch_correct = 0
                images, labels = images.to(args.device), labels.to(args.device)
                net_g.zero_grad()
                log_probs = net_g(images)
                # # 计算auc
                # y_scores = log_probs.detach().numpy()
                # y_scores = y_scores[:, 1]  # 关键一步得到scores
                # fpr, tpr, threshold = roc_curve(labels, y_scores,pos_label=0)
                # roc_auc = auc(fpr, tpr)
                # roc_auc_batch.append(roc_auc.item())

                y_pred = log_probs.data.max(1, keepdim=True)[1]
                batch_correct += y_pred.eq(labels.data.view_as(y_pred)).long().cuda().sum()
                # batch_correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()


                loss = loss_func(log_probs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(data_loader2),
                              100. * batch_idx / len(data_loader2), loss.item()))
                # batch_loss+=loss.item()
                batch_loss.append(loss.item())
                # batch_acc.append(y_pred)
                # batch_correct += y_pred.eq(images.data.view_as(y_pred)).long().cpu().sum()
                # print('batch_correct', batch_correct)
                # print("一个batch结束")

                # print('batch_correct',batch_correct)
                epoch_correct = (batch_correct.item() + epoch_correct)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # epoch_roc_auc.append(sum(roc_auc_batch)/len(roc_auc_batch))
            #将每一epoch的准确个数累加到总的正确个数里
            correct = epoch_correct + correct
            # print("dataloder2_correct",correct)


        #计算正确率
        correct=correct/number_epoch


        # # epoch_acc=[]
        # #训练本地数据
        # for idx, (images, labels) in enumerate(data_loader):
        #     images, labels = images.to(self.args.device), labels.to(self.args.device)
        #     net.zero_grad()
        #     log_probs = net(images)
        #     loss = self.loss_func(log_probs, labels)
        #     loss.backward()
        #     optimizer.step()
        # #训练共享数据
        # for iter in range(self.args.local_ep):
        #     batch_loss = []
        #     for batch_idx, (images, labels) in enumerate(self.ldr_train):
        #         images, labels = images.to(self.args.device), labels.to(self.args.device)
        #         net.zero_grad()
        #         log_probs = net(images)
        #         y_pred = log_probs.data.max(1, keepdim=True)[1]
        #         loss = self.loss_func(log_probs, labels)
        #         loss.backward()
        #         optimizer.step()
        #         if self.args.verbose and batch_idx % 10 == 0:
        #             print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #                 iter, batch_idx * len(images), len(self.ldr_train.dataset),
        #                        100. * batch_idx / len(self.ldr_train), loss.item()))
        #         batch_loss.append(loss.item())
        #         # batch_acc.append(y_pred)
        #         batch_correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
        #         # print('batch_correct',batch_correct)
        #         epoch_correct = (batch_correct.item() + epoch_correct)
        #     epoch_loss.append(sum(batch_loss)/len(batch_loss))
        #
        #     # print('2222222',epoch_correct)
        #     # correct = epoch_correct + correct
        #
        #
        # # print('333333333333333',epoch_correct)
        # # print('vvvvvvvvvvvvvv',self.number)

        return net_g.state_dict(), sum(epoch_loss) / len(epoch_loss),(correct/(len(minst_train_from_csv)+len(datatest)))

