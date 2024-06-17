import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layer import unetConv2
from model.init_weight import init_weights

class newUnet(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True,is_ds = True):
        super(newUnet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.is_ds = is_ds

        filters = [64, 128, 256, 512, 1024]
        # filters = [16, 32, 64, 128, 256]

        ## -------------Encoder----------------------------
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool00 = nn.MaxPool2d(kernel_size=2)

        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool10= nn.MaxPool2d(kernel_size=2)

        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool20 = nn.MaxPool2d(kernel_size=2)

        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool30 = nn.MaxPool2d(kernel_size=2)

        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        ## -------------Decoder----------------------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''node 11'''
        self.h00_PT_hd11 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h00_PT_hd11_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h00_PT_hd11_bn = nn.BatchNorm2d(self.CatChannels)
        self.h00_PT_hd11_relu = nn.ReLU(inplace=True)

        self.h10_Cat_hd11_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h10_Cat_hd11_bn = nn.BatchNorm2d(self.CatChannels)
        self.h10_Cat_hd11_relu = nn.ReLU(inplace=True)

        self.hd20_UT_hd11 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd20_UT_hd11_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.hd20_UT_hd11_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd20_UT_hd11_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv11d_01 = nn.Conv2d(filters[0]*3, filters[0]*2, 3, padding=1)  # 16
        self.bn11d_01 = nn.BatchNorm2d(filters[0]*2)
        self.relu11d_01 = nn.ReLU(inplace=True)


        '''node 01'''

        self.h00_Cat_hd01_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h00_Cat_hd01_bn = nn.BatchNorm2d(self.CatChannels)
        self.h00_Cat_hd01_relu = nn.ReLU(inplace=True)


        self.hd11_UT_hd01 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd11_UT_hd01_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.hd11_UT_hd01_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd11_UT_hd01_relu = nn.ReLU(inplace=True)

        # fusion
        self.conv01d_01 = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)  # 16
        self.bn01d_01 = nn.BatchNorm2d(self.CatChannels)
        self.relu01d_01 = nn.ReLU(inplace=True)

        '''node 31'''

        self.h01_PT_hd31 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h01_PT_hd31_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h01_PT_hd31_bn = nn.BatchNorm2d(self.CatChannels)
        self.h01_PT_hd31_relu = nn.ReLU(inplace=True)


        self.h11_PT_hd31 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h11_PT_hd31_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h11_PT_hd31_bn = nn.BatchNorm2d(self.CatChannels)
        self.h11_PT_hd31_relu = nn.ReLU(inplace=True)


        self.h20_PT_hd31 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h20_PT_hd31_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h20_PT_hd31_bn = nn.BatchNorm2d(self.CatChannels)
        self.h20_PT_hd31_relu = nn.ReLU(inplace=True)


        self.h30_Cat_hd31_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h30_Cat_hd31_bn = nn.BatchNorm2d(self.CatChannels)
        self.h30_Cat_hd31_relu = nn.ReLU(inplace=True)


        self.hd40_UT_hd31 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd40_UT_hd31_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd40_UT_hd31_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd40_UT_hd31_relu = nn.ReLU(inplace=True)


        self.conv31d_02 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn31d_02 = nn.BatchNorm2d(self.UpChannels)
        self.relu31d_02 = nn.ReLU(inplace=True)

        '''node 21'''

        self.h01_PT_hd21 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h01_PT_hd21_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h01_PT_hd21_bn = nn.BatchNorm2d(self.CatChannels)
        self.h01_PT_hd21_relu = nn.ReLU(inplace=True)


        self.h11_PT_hd21 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h11_PT_hd21_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h11_PT_hd21_bn = nn.BatchNorm2d(self.CatChannels)
        self.h11_PT_hd21_relu = nn.ReLU(inplace=True)


        self.h20_Cat_hd21_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h20_Cat_hd21_bn = nn.BatchNorm2d(self.CatChannels)
        self.h20_Cat_hd21_relu = nn.ReLU(inplace=True)


        self.hd31_UT_hd21 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd31_UT_hd21_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd31_UT_hd21_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd31_UT_hd21_relu = nn.ReLU(inplace=True)


        self.hd40_UT_hd21 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd40_UT_hd21_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd40_UT_hd21_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd40_UT_hd21_relu = nn.ReLU(inplace=True)

        # fusion
        self.conv21d_02 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn21d_02 = nn.BatchNorm2d(self.UpChannels)
        self.relu21d_02 = nn.ReLU(inplace=True)

        '''node 12'''

        self.h01_PT_hd12 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h01_PT_hd12_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h01_PT_hd12_bn = nn.BatchNorm2d(self.CatChannels)
        self.h01_PT_hd12_relu = nn.ReLU(inplace=True)


        self.h11_Cat_hd12_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h11_Cat_hd12_bn = nn.BatchNorm2d(self.CatChannels)
        self.h11_Cat_hd12_relu = nn.ReLU(inplace=True)

        self.hd21_UT_hd12 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd21_UT_hd12_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd21_UT_hd12_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd21_UT_hd12_relu = nn.ReLU(inplace=True)

        self.hd31_UT_hd12 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd31_UT_hd12_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd31_UT_hd12_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd31_UT_hd12_relu = nn.ReLU(inplace=True)


        self.hd40_UT_hd12 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd40_UT_hd12_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd40_UT_hd12_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd40_UT_hd12_relu = nn.ReLU(inplace=True)

        # fusion
        self.conv12d_02 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn12d_02 = nn.BatchNorm2d(self.UpChannels)
        self.relu12d_02 = nn.ReLU(inplace=True)

        '''node 02'''

        self.h01_Cat_hd02_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h01_Cat_hd02_bn = nn.BatchNorm2d(self.CatChannels)
        self.h01_Cat_hd02_relu = nn.ReLU(inplace=True)


        self.hd12_UT_hd02 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd12_UT_hd02_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd12_UT_hd02_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd12_UT_hd02_relu = nn.ReLU(inplace=True)


        self.hd21_UT_hd02 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd21_UT_hd02_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd21_UT_hd02_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd21_UT_hd02_relu = nn.ReLU(inplace=True)


        self.hd31_UT_hd02 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd31_UT_hd02_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd31_UT_hd02_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd31_UT_hd02_relu = nn.ReLU(inplace=True)


        self.hd40_UT_hd02 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd40_UT_hd02_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd40_UT_hd02_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd40_UT_hd02_relu = nn.ReLU(inplace=True)


        self.conv02d_02 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn02d_02 = nn.BatchNorm2d(self.UpChannels)
        self.relu02d_02 = nn.ReLU(inplace=True)

        # output
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(self.UpChannels, n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h00 = self.conv00(inputs)
        # print("h00",h00.shape)

        h10 = self.maxpool00(h00)
        h10 = self.conv10(h10)
        # print("h10",h00.shape)

        h20 = self.maxpool10(h10)
        h20 = self.conv20(h20)
        # print("h20",h20.shape)

        h30 = self.maxpool20(h20)
        h30 = self.conv30(h30)
        # print("h30", h30.shape)

        h40 = self.maxpool30(h30)
        hd40 = self.conv40(h40)
        # print("h40", h40.shape)

        ## -------------Decoder-------------
        '''node 11'''
        h00_PT_hd11=self.h00_PT_hd11(h00)
        h00_PT_hd11=self.h00_PT_hd11_conv(h00_PT_hd11)
        h00_PT_hd11=self.h00_PT_hd11_bn(h00_PT_hd11)
        h00_PT_hd11 = self.h00_PT_hd11_relu(h00_PT_hd11)
        # print("h00_PT_hd11",h00_PT_hd11.shape)

        h10_Cat_hd11= self.h10_Cat_hd11_conv(h10)
        h10_Cat_hd11= self.h10_Cat_hd11_bn(h10_Cat_hd11)
        h10_Cat_hd11 = self.h10_Cat_hd11_relu(h10_Cat_hd11)
        # print("h10_Cat_hd11",h10_Cat_hd11.shape)

        hd20_UT_hd11=self.hd20_UT_hd11(h20)
        hd20_UT_hd11= self.hd20_UT_hd11_conv(hd20_UT_hd11)
        hd20_UT_hd11 = self.hd20_UT_hd11_bn(hd20_UT_hd11)
        hd20_UT_hd11 = self.hd20_UT_hd11_relu(hd20_UT_hd11)
        # print("hd20_UT_hd11",hd20_UT_hd11.shape)

        hd11 = torch.cat((h00_PT_hd11, h10_Cat_hd11, hd20_UT_hd11), 1)
        # print("1  hd11", hd11.shape)
        hd11 = self.conv11d_01(hd11)
        # print("2  hd11", hd11.shape)
        hd11 = self.bn11d_01(hd11)
        # print("3  hd11", hd11.shape)
        hd11 = self.relu11d_01(hd11)
        # print("4  hd11", hd11.shape)

        '''node 01'''
        h00_Cat_hd1 = self.h00_Cat_hd01_conv(h00)
        h00_Cat_hd1 = self.h00_Cat_hd01_bn(h00_Cat_hd1)
        # print("h00_Cat_hd1",h00_Cat_hd1.shape)
        h00_Cat_hd1 = self.h00_Cat_hd01_relu(h00_Cat_hd1)
        # print("h00_Cat_hd1", h00_Cat_hd1.shape)

        # print("hd11",hd11.shape)
        hd11_UT_hd01 = self.hd11_UT_hd01(hd11)
        # print("hd11_UT_hd01",hd11_UT_hd01.shape)
        hd11_UT_hd01 = self.hd11_UT_hd01_conv(hd11_UT_hd01)
        hd11_UT_hd01 = self.hd11_UT_hd01_bn(hd11_UT_hd01)
        hd11_UT_hd01 = self.hd11_UT_hd01_relu(hd11_UT_hd01)

        hd01 = torch.cat((h00_Cat_hd1, hd11_UT_hd01), 1)
        # print("1 hd01", hd01.shape)
        hd01 = self.conv01d_01(hd01)
        # print("2 hd01", hd01.shape)
        hd01 = self.bn01d_01(hd01)
        # print("3 hd01", hd01.shape)
        hd01 = self.relu01d_01(hd01)
        # print("4 hd01",hd01.shape)

        '''node 31'''
        h01_PT_hd31 = self.h01_PT_hd31(hd01)
        # print("h01_PT_hd31",h01_PT_hd31.shape)
        h01_PT_hd31 = self.h01_PT_hd31_conv(h01_PT_hd31)
        h01_PT_hd31 = self.h01_PT_hd31_bn(h01_PT_hd31)
        h01_PT_hd31 = self.h01_PT_hd31_relu(h01_PT_hd31)

        # print("hd11",hd11.shape)
        h11_PT_hd31 = self.h11_PT_hd31(hd11)
        # print("h11_PT_hd31",h11_PT_hd31.shape)
        h11_PT_hd31 = self.h11_PT_hd31_conv(h11_PT_hd31)
        h11_PT_hd31 = self.h11_PT_hd31_bn(h11_PT_hd31)
        h11_PT_hd31 = self.h11_PT_hd31_relu(h11_PT_hd31)

        h20_PT_hd31 = self.h20_PT_hd31(h20)
        h20_PT_hd31 = self.h20_PT_hd31_conv(h20_PT_hd31)
        h20_PT_hd31 = self.h20_PT_hd31_bn(h20_PT_hd31)
        h20_PT_hd31 = self.h20_PT_hd31_relu(h20_PT_hd31)

        h30_Cat_hd31 = self.h30_Cat_hd31_conv(h30)
        h30_Cat_hd31 = self.h30_Cat_hd31_bn(h30_Cat_hd31)
        h30_Cat_hd31 = self.h30_Cat_hd31_relu(h30_Cat_hd31)

        hd40_UT_hd31 = self.hd40_UT_hd31(hd40)
        hd40_UT_hd31 = self.hd40_UT_hd31_conv(hd40_UT_hd31)
        hd40_UT_hd31 = self.hd40_UT_hd31_bn(hd40_UT_hd31)
        hd40_UT_hd31 = self.hd40_UT_hd31_relu(hd40_UT_hd31)

        hd31 = torch.cat((h01_PT_hd31, h11_PT_hd31, h20_PT_hd31, h30_Cat_hd31, hd40_UT_hd31), 1)
        hd31 = self.conv31d_02(hd31)
        hd31 = self.bn31d_02(hd31)
        hd31 = self.relu31d_02(hd31)

        '''node 21'''
        h01_PT_hd21 = self.h01_PT_hd21(hd01)
        h01_PT_hd21 = self.h01_PT_hd21_conv(h01_PT_hd21)
        h01_PT_hd21 = self.h01_PT_hd21_bn(h01_PT_hd21)
        h01_PT_hd21 = self.h01_PT_hd21_relu(h01_PT_hd21)

        h11_PT_hd21 = self.h11_PT_hd21(hd11)
        h11_PT_hd21 = self.h11_PT_hd21_conv(h11_PT_hd21)
        h11_PT_hd21 = self.h11_PT_hd21_bn(h11_PT_hd21)
        h11_PT_hd21 = self.h11_PT_hd21_relu(h11_PT_hd21)

        h20_Cat_hd21 = self.h20_Cat_hd21_conv(h20)
        h20_Cat_hd21 = self.h20_Cat_hd21_bn(h20_Cat_hd21)
        h20_Cat_hd21 = self.h20_Cat_hd21_relu(h20_Cat_hd21)

        hd31_UT_hd21 = self.hd31_UT_hd21(hd31)
        hd31_UT_hd21 = self.hd31_UT_hd21_conv(hd31_UT_hd21)
        hd31_UT_hd21 = self.hd31_UT_hd21_bn(hd31_UT_hd21)
        hd31_UT_hd21 = self.hd31_UT_hd21_relu(hd31_UT_hd21)

        hd40_UT_hd21 = self.hd40_UT_hd21(hd40)
        hd40_UT_hd21 = self.hd40_UT_hd21_conv(hd40_UT_hd21)
        hd40_UT_hd21 = self.hd40_UT_hd21_bn(hd40_UT_hd21)
        hd40_UT_hd21 = self.hd40_UT_hd21_relu(hd40_UT_hd21)

        hd21 = torch.cat((h01_PT_hd21, h11_PT_hd21, h20_Cat_hd21, hd31_UT_hd21, hd40_UT_hd21), 1)
        hd21 = self.conv21d_02(hd21)
        hd21 = self.bn21d_02(hd21)
        hd21 = self.relu21d_02(hd21)

        '''node 12'''
        h01_PT_hd12 = self.h01_PT_hd12(hd01)
        h01_PT_hd12 = self.h01_PT_hd12_conv(h01_PT_hd12)
        h01_PT_hd12 = self.h01_PT_hd12_bn(h01_PT_hd12)
        h01_PT_hd12 = self.h01_PT_hd12_relu(h01_PT_hd12)

        h11_Cat_hd12 = self.h11_Cat_hd12_conv(hd11)
        h11_Cat_hd12= self.h11_Cat_hd12_bn(h11_Cat_hd12)
        h11_Cat_hd12 = self.h11_Cat_hd12_relu(h11_Cat_hd12)

        hd21_UT_hd12 = self.hd21_UT_hd12(hd21)
        # print("hd21_UT_hd12",hd21_UT_hd12.shape)
        hd21_UT_hd12 = self.hd21_UT_hd12_conv(hd21_UT_hd12)
        hd21_UT_hd12 = self.hd21_UT_hd12_bn(hd21_UT_hd12)
        hd21_UT_hd12 = self.hd21_UT_hd12_relu(hd21_UT_hd12)

        hd31_UT_hd12 = self.hd31_UT_hd12(hd31)
        hd31_UT_hd12 = self.hd31_UT_hd12_conv(hd31_UT_hd12)
        hd31_UT_hd12 = self.hd31_UT_hd12_bn(hd31_UT_hd12)
        hd31_UT_hd12 = self.hd31_UT_hd12_relu(hd31_UT_hd12)

        hd40_UT_hd12 = self.hd40_UT_hd12(hd40)
        hd40_UT_hd12 = self.hd40_UT_hd12_conv(hd40_UT_hd12)
        hd40_UT_hd12 = self.hd40_UT_hd12_bn(hd40_UT_hd12)
        hd40_UT_hd12 = self.hd40_UT_hd12_relu(hd40_UT_hd12)

        hd12 = torch.cat((h01_PT_hd12, h11_Cat_hd12, hd21_UT_hd12, hd31_UT_hd12 , hd40_UT_hd12), 1)
        hd12 = self.conv12d_02(hd12)
        hd12 = self.bn12d_02(hd12)
        hd12 = self.relu12d_02(hd12)  # hd2->160*160*UpChannels

        '''node 02'''

        h01_Cat_hd02 = self.h01_Cat_hd02_conv(hd01)
        h01_Cat_hd02 = self.h01_Cat_hd02_bn(h01_Cat_hd02)
        h01_Cat_hd02 = self.h01_Cat_hd02_relu(h01_Cat_hd02)

        hd12_UT_hd02 = self.hd12_UT_hd02(hd12)
        hd12_UT_hd02 = self.hd12_UT_hd02_conv(hd12_UT_hd02)
        hd12_UT_hd02 = self.hd12_UT_hd02_bn(hd12_UT_hd02)
        hd12_UT_hd02 = self.hd12_UT_hd02_relu(hd12_UT_hd02)

        hd21_UT_hd02 = self.hd21_UT_hd02(hd21)
        hd21_UT_hd02 = self.hd21_UT_hd02_conv(hd21_UT_hd02)
        hd21_UT_hd02 = self.hd21_UT_hd02_bn(hd21_UT_hd02)
        hd21_UT_hd02 = self.hd21_UT_hd02_relu(hd21_UT_hd02)

        hd31_UT_hd02 = self.hd31_UT_hd02(hd31)
        hd31_UT_hd02 = self.hd31_UT_hd02_conv(hd31_UT_hd02)
        hd31_UT_hd02 = self.hd31_UT_hd02_bn(hd31_UT_hd02)
        hd31_UT_hd02 = self.hd31_UT_hd02_relu(hd31_UT_hd02)

        hd40_UT_hd02 = self.hd40_UT_hd02(hd40)
        hd40_UT_hd02 = self.hd40_UT_hd02_conv(hd40_UT_hd02)
        hd40_UT_hd02 = self.hd40_UT_hd02_bn(hd40_UT_hd02)
        hd40_UT_hd02 = self.hd40_UT_hd02_relu(hd40_UT_hd02)

        hd02 = torch.cat((h01_Cat_hd02, hd12_UT_hd02, hd21_UT_hd02, hd31_UT_hd02, hd40_UT_hd02), 1)
        hd02 = self.conv02d_02(hd02)
        hd02 = self.bn02d_02(hd02)
        hd02 = self.relu02d_02(hd02)

        final_1 = self.final_1(hd01)
        final_2 = self.final_2(hd02)

        final = (final_1 + final_2) / 2

        if self.is_ds:
            return F.sigmoid(final)
        else:
            return F.sigmoid(final_2)



if __name__ == "__main__":
    print("形状")
    model=newUnet()

