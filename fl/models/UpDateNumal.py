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
import torch.nn.functional as F
from itertools import cycle
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class MinstDataSetFromImages(Dataset):
    # 1. 初始化文件路径或文件名列表�?
    # 也就是在这个模块里，我们所做的工作就是初始化该类的一些基本参数�?
    def __init__(self, csv_path, transforms=None):
        self.transforms  = transforms
        self.data_info = pd.read_csv(csv_path,header=None)
        self.image_arr = np.asarray(self.data_info.iloc[:,0])
        self.label_arr = np.asarray(self.data_info.iloc[:,1])
        self.data_len = len(self.data_info.index)

    # 这里需要注意的是，第一步：read one data，是一个data
    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
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
        prefix = "E:/code/fl/data/mnist/"
        single_image_name = prefix+single_image_name.strip("./")

        img_as_img = Image.open(single_image_name).convert('RGB')
        #img_as_img = cv2.imread(single_image_name)
        image_as_tensor = self.transforms(img_as_img)

        single_image_name = self.label_arr[index]
        return(image_as_tensor, single_image_name)

    def __len__(self):
        return self.data_len


def LocalUpdateNumal(args,i,net):
        #加载本地数据
        # print("update 第一�?)
        # transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        path = "E:/code/fl/data/mnist/{}.csv".format(i,i+1)

        minst_train_from_csv = MinstDataSetFromImages(path,
                                                      transforms=train_transformer)
        data_loader = DataLoader(minst_train_from_csv, batch_size=10,shuffle=False)
        #定义参数
        correct = 0
        train_loss = 0
        accuracy=0
        accuracy_avg=0
        accuracy_totl=0
        # loss = 0
        loss_func = nn.CrossEntropyLoss()
        # print("dao zhe le")
        epoch_loss = []
        epoch_roc_auc = []

        #开始训�?
        net.train()

        # 第二�?
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
        loss_func = nn.CrossEntropyLoss()


        for iter in range(args.local_ep):
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # print("iter",iter)
            batch_loss = []
            roc_auc_batch = []
            # accuracy=0
            for batch_idx, (images, labels) in enumerate(data_loader):
                correct = 0
                images, labels = images.to(args.device), labels.to(args.device)
                # print("images",images)
                # print("labels",labels)
                net.zero_grad()
                log_probs = net(images)

                loss = loss_func(log_probs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                y_pred = log_probs.data.max(1, keepdim=True)[1]
                # correct += y_pred.eq(labels.data.view_as(y_pred)).long().cuda().sum()
                correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
                accuracy += correct.item()

                batch_loss.append(loss.item())
            # accuracy = 100.00 * correct / len(data_loader.dataset)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            # accuracy_totl+=accuracy
        accuracy=accuracy/(args.local_ep)
        # print("ep!!!!!!!!!!",args.local_ep)
        # print("acc",accuracy)
        accuracy=accuracy/len(data_loader.dataset)

        # print("最后的acc",accuracy)


        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), accuracy
