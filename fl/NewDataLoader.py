
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import pandas as pd
import numpy as np


#写自己的共享数据集
class MinstDataSetShareFromImages(Dataset):
	# 1. 初始化文件路径或文件名列表。
	# 也就是在这个模块里，我们所做的工作就是初始化该类的一些基本参数。
	def __init__(self, sa,csv_path, transforms=None):
		self.transforms  = transforms
		self.data_info = pd.read_csv(csv_path,header=None)
		self.image_arr = np.asarray(self.data_info.iloc[0:int(sa*len(self.data_info.index)),0])
		self.label_arr = np.asarray(self.data_info.iloc[0:int(sa*len(self.data_info.index)),1])
		self.data_len = int(sa*len(self.data_info.index))

	# 这里需要注意的是，第一步：read one data，是一个data
	def __getitem__(self, index):
		single_image_name = self.image_arr[index]
		prefix = "E:\\code\\federated-learning\\data\\mnist\\"
		single_image_name = prefix+single_image_name.strip("./")

		img_as_img = Image.open(single_image_name)
		#img_as_img = cv2.imread(single_image_name)
		image_as_tensor = self.transforms(img_as_img)

		single_image_name = self.label_arr[index]
		return(image_as_tensor, single_image_name)

	def __len__(self):
		return self.data_len
if __name__ == "__main__":
	transformations = transforms.Compose([transforms.ToTensor()])
	minst_from_csv = MinstDataSetShareFromImages(0.0011,"E:/code/federated-learning/data/mnist/share.csv",transforms=transformations)
	mnist_dataloader = DataLoader(dataset=minst_from_csv,batch_size=10,shuffle=False)
	for i in mnist_dataloader:
		print(i)
