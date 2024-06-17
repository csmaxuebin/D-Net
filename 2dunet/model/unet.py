""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


# 从UNet网络中可以看出，不管是下采样过程还是上采样过程，每一层都会连续进行两次卷积操作，
# 这种操作在UNet网络中重复很多次，可以单独写一个DoubleConv模块
# DoubleConv模块的in_channels和out_channels可以灵活设定，以便扩展使用。
# in_channels设为1，out_channels为64。
# 输入图片大小为572*572，经过步长为1，padding为0的3*3卷积，得到570*570的feature map，
# 再经过一次卷积得到568*568的feature map。
# 计算公式：O=(H−F+2×P)/S+1
# H为输入feature map的大小，O为输出feature map的大小，F为卷积核的大小，P为padding的大小，S为步长

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# 就是一个maxpool池化层，进行下采样，然后接一个DoubleConv模块
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# 首先是__init__初始化函数里定义的上采样方法以及卷积采用DoubleConv。
# 上采样，定义了两种方法：Upsample和ConvTranspose2d，也就是双线性插值和反卷积。

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    # 在forward前向传播函数中，
    # x1接收的是上采样的数据，x2接收的是特征融合的数据。特征融合方法就是
    # 先对小的feature map进行padding，再进行concat。
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" Full assembly of the parts to form the complete network """

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        inter_channel = 16

        self.inc = DoubleConv(n_channels, inter_channel)
        self.down1 = Down(inter_channel, inter_channel*2)
        self.down2 = Down(inter_channel*2, inter_channel*4)
        self.down3 = Down(inter_channel*4, inter_channel*8)
        self.down4 = Down(inter_channel*8, inter_channel*8)
        self.up1 = Up(inter_channel*16, inter_channel*4, bilinear)
        self.up2 = Up(inter_channel*8, inter_channel*2, bilinear)
        self.up3 = Up(inter_channel*4, inter_channel, bilinear)
        self.up4 = Up(inter_channel*2, inter_channel, bilinear)
        self.outc = OutConv(inter_channel, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # 1/16
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



if __name__ == "__main__":
    unet = UNet(n_channels=1, n_classes=2)
    aa = torch.ones((2, 1, 128, 128))
    bb = unet(aa)
    print (bb.shape)

# UNet(
#   (inc): DoubleConv(
#     (double_conv): Sequential(
#       (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
#       (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU(inplace=True)
#       (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
#       (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU(inplace=True)
#     )
#   )
#   (down1): Down(
#     (maxpool_conv): Sequential(
#       (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#       (1): DoubleConv(
#         (double_conv): Sequential(
#           (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
#           (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (2): ReLU(inplace=True)
#           (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
#           (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (5): ReLU(inplace=True)
#         )
#       )
#     )
#   )
#   (down2): Down(
#     (maxpool_conv): Sequential(
#       (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#       (1): DoubleConv(
#         (double_conv): Sequential(
#           (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
#           (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (2): ReLU(inplace=True)
#           (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
#           (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (5): ReLU(inplace=True)
#         )
#       )
#     )
#   )
#   (down3): Down(
#     (maxpool_conv): Sequential(
#       (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#       (1): DoubleConv(
#         (double_conv): Sequential(
#           (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1))
#           (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (2): ReLU(inplace=True)
#           (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
#           (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (5): ReLU(inplace=True)
#         )
#       )
#     )
#   )
#   (down4): Down(
#     (maxpool_conv): Sequential(
#       (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#       (1): DoubleConv(
#         (double_conv): Sequential(
#           (0): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1))
#           (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (2): ReLU(inplace=True)
#           (3): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
#           (4): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (5): ReLU(inplace=True)
#         )
#       )
#     )
#   )
#   (up1): Up(
#     (up): ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=(2, 2))
#     (conv): DoubleConv(
#       (double_conv): Sequential(
#         (0): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1))
#         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
#         (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): ReLU(inplace=True)
#       )
#     )
#   )
#   (up2): Up(
#     (up): ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2))
#     (conv): DoubleConv(
#       (double_conv): Sequential(
#         (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1))
#         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
#         (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): ReLU(inplace=True)
#       )
#     )
#   )
#   (up3): Up(
#     (up): ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2))
#     (conv): DoubleConv(
#       (double_conv): Sequential(
#         (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1))
#         (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
#         (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): ReLU(inplace=True)
#       )
#     )
#   )
#   (up4): Up(
#     (up): ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))
#     (conv): DoubleConv(
#       (double_conv): Sequential(
#         (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1))
#         (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (2): ReLU(inplace=True)
#         (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
#         (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (5): ReLU(inplace=True)
#       )
#     )
#   )
#   (outc): OutConv(
#     (conv): Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
#   )
# )