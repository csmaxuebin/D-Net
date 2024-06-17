from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from net.layer2 import unetConv2, unetUp_origin
from net.init_weight import init_weights

import torch
import torch.nn as nn
import torch.nn.functional as F
# from net.layer2 import unetConv2
# from init_weights import init_weights
import numpy as np
from torchvision import models
from functools import partial



nonlinearity = partial(F.relu, inplace=True)


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class CE_Net_(nn.Module):
    def __init__(self, num_channels=3,num_classes=2):
        super(CE_Net_, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class UNet_2Plus(nn.Module):

    def __init__(self, in_channels=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(UNet_2Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.feature_scale = feature_scale

        # filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)


        # upsampling
        self.up_concat01 = unetUp_origin(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp_origin(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp_origin(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp_origin(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp_origin(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp_origin(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp_origin(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp_origin(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp_origin(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)
        maxpool0 = self.maxpool0(X_00)
        X_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool1(X_10)
        X_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool2(X_20)
        X_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool3(X_30)
        X_40 = self.conv40(maxpool3)

        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4

        if self.is_ds:
            return F.sigmoid(final)
        else:
            return F.sigmoid(final_4)


class newUnet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,is_ds = True):
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



class UNet_3Plus(nn.Module):

    def __init__(self, in_channels=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        # 利用ceil_mode参数向上取整
        # 定义输入
        # 四个参数分别表示 (batch_size, C_in, H_in, W_in)
        # 分别对应，批处理大小，输入通道数，图像高度（像素），图像宽度（像素）
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64
        # print("h1",h1.shape)

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128
        # print("h2", h2.shape)

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256
        # print("h3", h3.shape)

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512
        # print("h4", h4.shape)

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024
        # print("h5", h5.shape)


        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  # hd4->40*40*UpChannels
        # print("hd4", hd4.shape)

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels
        # print("hd3", hd3.shape)

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels
        # print("hd2", hd2.shape)

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels
        # print("hd1", hd1.shape)

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        return F.sigmoid(d1)


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self):
        super(U_Net, self).__init__()
        # self.args = args
        in_ch = 3
        out_ch = 2
        
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out


class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """
    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out


class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """
    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out


class R2U_Net(nn.Module):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """
    def __init__(self, img_ch=3, output_ch=2, t=2):
        super(R2U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(img_ch, filters[0], t=t)

        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)

        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)

        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)

        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.RRCNN1(x)

        e2 = self.Maxpool(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool1(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool2(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool3(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

      # out = self.active(out)

        return out


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self):
        super(AttU_Net, self).__init__()
        # self.args = args
        in_ch = 3
        out_ch = 2

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        #print(x5.shape)
        d5 = self.Up5(e5)
        #print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out


class R2AttU_Net(nn.Module):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """
    def __init__(self, in_ch=3, out_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RRCNN1 = RRCNN_block(in_ch, filters[0], t=t)
        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)
        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)
        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)
        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.RRCNN1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        e4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        e3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out

#For nested 3 channels are required

class conv_block_nested(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output
    
#Nested Unet

class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(NestedUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)


    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output

#Dictioary Unet
#if required for getting the filters and model parameters for each step 

class ConvolutionBlock(nn.Module):
    """Convolution block"""

    def __init__(self, in_filters, out_filters, kernel_size=3, batchnorm=True, last_active=F.relu):
        super(ConvolutionBlock, self).__init__()

        self.bn = batchnorm
        self.last_active = last_active
        self.c1 = nn.Conv2d(in_filters, out_filters, kernel_size, padding=1)
        self.b1 = nn.BatchNorm2d(out_filters)
        self.c2 = nn.Conv2d(out_filters, out_filters, kernel_size, padding=1)
        self.b2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.c1(x)
        if self.bn:
            x = self.b1(x)
        x = F.relu(x)
        x = self.c2(x)
        if self.bn:
            x = self.b2(x)
        x = self.last_active(x)
        return x


class ContractiveBlock(nn.Module):
    """Deconvuling Block"""

    def __init__(self, in_filters, out_filters, conv_kern=3, pool_kern=2, dropout=0.5, batchnorm=True):
        super(ContractiveBlock, self).__init__()
        self.c1 = ConvolutionBlock(in_filters=in_filters, out_filters=out_filters, kernel_size=conv_kern,
                                   batchnorm=batchnorm)
        self.p1 = nn.MaxPool2d(kernel_size=pool_kern, ceil_mode=True)
        self.d1 = nn.Dropout2d(dropout)

    def forward(self, x):
        c = self.c1(x)
        return c, self.d1(self.p1(c))


class ExpansiveBlock(nn.Module):
    """Upconvole Block"""

    def __init__(self, in_filters1, in_filters2, out_filters, tr_kern=3, conv_kern=3, stride=2, dropout=0.5):
        super(ExpansiveBlock, self).__init__()
        self.t1 = nn.ConvTranspose2d(in_filters1, out_filters, tr_kern, stride=2, padding=1, output_padding=1)
        self.d1 = nn.Dropout(dropout)
        self.c1 = ConvolutionBlock(out_filters + in_filters2, out_filters, conv_kern)

    def forward(self, x, contractive_x):
        x_ups = self.t1(x)
        x_concat = torch.cat([x_ups, contractive_x], 1)
        x_fin = self.c1(self.d1(x_concat))
        return x_fin


class Unet_dict(nn.Module):
    """Unet which operates with filters dictionary values"""

    def __init__(self, n_labels, n_filters=32, p_dropout=0.5, batchnorm=True):
        super(Unet_dict, self).__init__()
        filters_dict = {}
        filt_pair = [3, n_filters]

        for i in range(4):
            self.add_module('contractive_' + str(i), ContractiveBlock(filt_pair[0], filt_pair[1], batchnorm=batchnorm))
            filters_dict['contractive_' + str(i)] = (filt_pair[0], filt_pair[1])
            filt_pair[0] = filt_pair[1]
            filt_pair[1] = filt_pair[1] * 2

        self.bottleneck = ConvolutionBlock(filt_pair[0], filt_pair[1], batchnorm=batchnorm)
        filters_dict['bottleneck'] = (filt_pair[0], filt_pair[1])

        for i in reversed(range(4)):
            self.add_module('expansive_' + str(i),
                            ExpansiveBlock(filt_pair[1], filters_dict['contractive_' + str(i)][1], filt_pair[0]))
            filters_dict['expansive_' + str(i)] = (filt_pair[1], filt_pair[0])
            filt_pair[1] = filt_pair[0]
            filt_pair[0] = filt_pair[0] // 2

        self.output = nn.Conv2d(filt_pair[1], n_labels, kernel_size=1)
        filters_dict['output'] = (filt_pair[1], n_labels)
        self.filters_dict = filters_dict

    # final_forward
    def forward(self, x):
        c00, c0 = self.contractive_0(x)
        c11, c1 = self.contractive_1(c0)
        c22, c2 = self.contractive_2(c1)
        c33, c3 = self.contractive_3(c2)
        bottle = self.bottleneck(c3)
        u3 = F.relu(self.expansive_3(bottle, c33))
        u2 = F.relu(self.expansive_2(u3, c22))
        u1 = F.relu(self.expansive_1(u2, c11))
        u0 = F.relu(self.expansive_0(u1, c00))
        return F.softmax(self.output(u0), dim=1)