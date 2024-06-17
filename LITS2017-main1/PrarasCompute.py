import torch
import torchvision
import argparse
# from torchsummary import summary
from net.Unet import newUnet
from net.Unet import UNet_3Plus
# from net.Unet import UNet_2Plus
from net.Unet import U_Net
from net.u2net import U2NET
from net.R2Unet import R2U_Net
from net.Unet import UNet_2Plus
from net.Unet import AttU_Net
from net.Unet import CE_Net_
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')


    # 换数据集需要修改的地方
    parser.add_argument('--dataset', default="LiTS",
                        help='dataset name')
    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='npy',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='npy',
                        help='mask file extension')

    # 换模型需要修改的地方
    # parser.add_argument('--epochs', default=250, type=int, metavar='N',
    #                     help='number of total epochs to run')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=100, type=int,
                        metavar='N', help='early stopping (default: 30)')

    # 换模型需要修改的地方
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')

    args = parser.parse_args()

    return args


# Model
print('==> Building model..')
model = newUnet()
# model = UNet_3Plus()
# model = U_Net(args)
# model = U2NET()
# model = R2U_Net()
# model = UNet_2Plus()
model = AttU_Net()
# model = CE_Net_()
# model = U_Net()


params = list(model.parameters())
k = 0
for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l
print("总参数数量和：" + str(k))
