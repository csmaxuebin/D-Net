
import matplotlib
matplotlib.use('Agg')
from torchvision import datasets, transforms
import torch
from utils.options import args_parser
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    acc = 0.0

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    net_glob = SimpleCNN().to(args.device)
    minst_train_from_csv = MinstDataSetFromImages("E:/code/federated-learning/data/ct/train.csv", transforms=train_transformer)

    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, minst_train_from_csv, args)
    print("Training accuracy: {:.2f}".format(acc_train))
