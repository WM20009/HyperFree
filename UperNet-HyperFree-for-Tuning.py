import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import torch

from HyperFree.utils.spectral_process_utils import read_img, write_img
from HyperFree import SamAutomaticMaskGenerator, HyperFree_model_registry, HyperFree_Predictor
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize

"""
Supprting tuning HyperFree with UperNet method
"""

device = 'cuda:0'

class Concat(nn.Module):

    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.dim = dimension

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        return torch.cat(x, dim=self.dim)


def autopad(kernel, padding):
    if padding is None:
        return kernel // 2 if isinstance(kernel, int) else [p // 2 for p in kernel]
    else:
        return padding


class Upsample(nn.Module):

    def __init__(self, factor=2) -> None:
        super(Upsample, self).__init__()
        self.factor = factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.factor, mode="bilinear")



class ConvBnAct(nn.Module):

    def __init__(self, in_channel, out_channel, kernel, stride, padding, dilation=1, bias=False, act=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=padding, dilation=dilation, bias=bias)
        self.in_ = nn.InstanceNorm2d(out_channel)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        if x.shape[2] > 1:
            x = self.in_(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class PyramidPoolingModule(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(PyramidPoolingModule, self).__init__()
        inter_channels = in_channels //4
        self.cba1 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.cba2 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.cba3 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.cba4 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.out  = ConvBnAct(in_channels * 2, out_channels, 1, 1, 0)

    def pool(self, x, size):
        return nn.AdaptiveAvgPool2d(size)(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode="bilinear", align_corners=True)

    def forward(self, x):
        size = x.shape[2:]
        f1 = self.upsample(self.cba1(self.pool(x, 1)), size)
        f2 = self.upsample(self.cba2(self.pool(x, 2)), size)
        f3 = self.upsample(self.cba3(self.pool(x, 3)), size)
        f4 = self.upsample(self.cba4(self.pool(x, 6)), size)
        f = torch.cat([x, f1, f2, f3, f4], dim=1)
        return self.out(f)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride, downsample):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvBnAct(inplanes, planes, 1, 1, 0, act=True)
        self.conv2 = ConvBnAct(planes, planes, 3, stride, 1, act=True)
        self.conv3 = ConvBnAct(planes, planes*self.expansion, 1, 1, 0, act=False)
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)  # conv with 1x1 kernel and stride 2
        out = out + residual
        return self.act(out)



class ResidualNet(nn.Module):
    
    def __init__(self, in_channel, block, layers) -> None:
        super(ResidualNet, self).__init__()
        self.stem = nn.Sequential(ConvBnAct(in_channel, out_channel=64, kernel=3, stride=2, padding=1, dilation=1),   # /2; c64
                                  ConvBnAct(64, 64, 1, 1, 0),                                                         # c64
                                  ConvBnAct(64, 128, 1, 1, 0),                                                        # c128
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                                    # /2; c128
                                  )
        self.inplanes = 128

        self.layer1 = self.make_layer(block, 64 , layers[0])             # c256
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)  # /2; c512
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)  # /2; c1024
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)  # /2; c2048


    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = ConvBnAct(self.inplanes, planes*block.expansion, 1, stride, 0, act=False)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Inputs:
            x: (bs, c, h, w)
        Outputs:
            y: (bs, out, h, w)

        """
        out_dict = {}
        x = self.stem(x)
        x = self.layer1(x)
        out_dict['resnet_layer1'] = x  # c256
        x = self.layer2(x)
        out_dict['resnet_layer2'] = x  # c512
        x = self.layer3(x)
        out_dict['resnet_layer3'] = x  # c1024  
        x = self.layer4(x)
        out_dict['resnet_layer4'] = x  # c2048
        return out_dict



class FeaturePyramidNet(nn.Module):

    def __init__(self, fpn_dim=256):
        self.fpn_dim = fpn_dim
        super(FeaturePyramidNet, self).__init__()
        self.fpn_in = nn.ModuleDict({'fpn_layer1': ConvBnAct(768 , self.fpn_dim, 1, 1, 0), 
                                     "fpn_layer2": ConvBnAct(768 , self.fpn_dim, 1, 1, 0), 
                                     "fpn_layer3": ConvBnAct(768, self.fpn_dim, 1, 1, 0), 
                                    })
        self.fpn_out = nn.ModuleDict({'fpn_layer1': ConvBnAct(self.fpn_dim, self.fpn_dim, 3, 1, 1), 
                                      "fpn_layer2": ConvBnAct(self.fpn_dim, self.fpn_dim, 3, 1, 1), 
                                      "fpn_layer3": ConvBnAct(self.fpn_dim, self.fpn_dim, 3, 1, 1), 
                                      })

    def forward(self, pyramid_features):
        """
        
        """
        fpn_out = {}
        
        f = pyramid_features[3]
        fpn_out['fpn_layer4'] = f
        x = self.fpn_in['fpn_layer3'](pyramid_features[2])
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        fpn_out['fpn_layer3'] = self.fpn_out['fpn_layer3'](f)

        x = self.fpn_in['fpn_layer2'](pyramid_features[1])
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        fpn_out['fpn_layer2'] = self.fpn_out['fpn_layer2'](f)

        x = self.fpn_in['fpn_layer1'](pyramid_features[0])
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        fpn_out['fpn_layer1'] = self.fpn_out['fpn_layer1'](f)

        return fpn_out


class UPerNet(nn.Module):

    def __init__(self, GSD, wavelengths, num_class=20, fpn_dim=256, hyperfree_predictor=None):
        super(UPerNet, self).__init__()
        global device
        self.fpn_dim = fpn_dim
        self.hyperfree_predictor = hyperfree_predictor
        self.GSD = GSD
        self.wavelengths = wavelengths

        self.ppm = PyramidPoolingModule(256, self.fpn_dim)
        self.fpn = FeaturePyramidNet(self.fpn_dim)
        self.fuse = ConvBnAct(fpn_dim*4, fpn_dim, 1, 1, 0)
        self.seg = nn.Sequential(ConvBnAct(fpn_dim, fpn_dim, 1, 1, 0), nn.Conv2d(fpn_dim, num_class, 1, 1, 0, bias=True))
        self.out = nn.Conv2d(num_class, num_class, 3, 1, 1)
        self.backbone = self.hyperfree_predictor.model.image_encoder

    def forward(self, x):
        self.hyperfree_predictor.set_image(x, True, self.wavelengths, self.GSD,)
        self.multi_scale_features = self.hyperfree_predictor.multi_scale_features

        seg_size = x.shape[:2]
        max_length = max(x.shape[:2])
        ppm = self.ppm(self.multi_scale_features[3])
        self.multi_scale_features[3] = ppm
        fpn = self.fpn(self.multi_scale_features)
        out_size = fpn['fpn_layer1'].shape[2:]
        list_f = []
        list_f.append(fpn['fpn_layer1'])
        list_f.append(F.interpolate(fpn['fpn_layer2'], out_size, mode='bilinear', align_corners=False))
        list_f.append(F.interpolate(fpn['fpn_layer3'], out_size, mode='bilinear', align_corners=False))
        list_f.append(F.interpolate(fpn['fpn_layer4'], out_size, mode='bilinear', align_corners=False))
        x = self.seg(self.fuse(torch.cat(list_f, dim=1)))
        pred = self.out(F.interpolate(x, (max_length, max_length), mode='bilinear', align_corners=False))
        pred = pred[:,:,:seg_size[0], :seg_size[1]]
        return pred # B, C, H, W
        
        
        

if __name__ == "__main__":
    img = read_img("../../Data/hyperspectral_classification/WHU-Hi-LongKou.tif")
    pretrained_ckpt = "./../../Ckpt/HyperFree-b.pth"

    GSD = 0.456
    height, width = img.shape[0], img.shape[1]
    ratio = 1024 / (max(height, width))
    GSD = GSD / ratio
    GSD = torch.tensor([GSD])

    wavelengths = [
        401.809998, 404.031006, 406.252014, 408.472992, 410.694000, 412.915009,
        415.135986, 417.356995, 419.578003, 421.799988, 424.020996, 426.242004,
        428.463013, 430.683990, 432.904999, 435.126007, 437.346985, 439.567993,
        441.789001, 444.010010, 446.230988, 448.451996, 450.674011, 452.894989,
        455.115997, 457.337006, 459.558014, 461.778992, 464.000000, 466.221008,
        468.441986, 470.662994, 472.884003, 475.105011, 477.326996, 479.548004,
        481.769012, 483.989990, 486.210999, 488.432007, 490.653015, 492.873993,
        495.095001, 497.316010, 499.536987, 501.757996, 503.979004, 506.200989,
        508.421997, 510.643005, 512.864014, 515.085022, 517.306030, 519.526978,
        521.747986, 523.968994, 526.190002, 528.411011, 530.632019, 532.854004,
        535.075012, 537.296021, 539.517029, 541.737976, 543.958984, 546.179993,
        548.401001, 550.622009, 552.843018, 555.064026, 557.284973, 559.505981,
        561.728027, 563.948975, 566.169983, 568.390991, 570.612000, 572.833008,
        575.054016, 577.275024, 579.495972, 581.716980, 583.937988, 586.158997,
        588.380981, 590.601990, 592.822998, 595.044006, 597.265015, 599.486023,
        601.706970, 603.927979, 606.148987, 608.369995, 610.591003, 612.812012,
        615.033020, 617.255005, 619.476013, 621.697021, 623.918030, 626.138977,
        628.359985, 630.580994, 632.802002, 635.023010, 637.244019, 639.465027,
        641.685974, 643.908020, 646.129028, 648.349976, 650.570984, 652.791992,
        655.013000, 657.234009, 659.455017, 661.676025, 663.896973, 666.117981,
        668.338989, 670.559998, 672.781982, 675.002991, 677.223999, 679.445007,
        681.666016, 683.887024, 686.107971, 688.328979, 690.549988, 692.770996,
        694.992004, 697.213013, 699.434998, 701.656006, 703.877014, 706.098022,
        708.318970, 710.539978, 712.760986, 714.981995, 717.203003, 719.424011,
        721.645020, 723.866028, 726.086975, 728.309021, 730.530029, 732.750977,
        734.971985, 737.192993, 739.414001, 741.635010, 743.856018, 746.077026,
        748.297974, 750.518982, 752.739990, 754.961975, 757.182983, 759.403992,
        761.625000, 763.846008, 766.067017, 768.288025, 770.508972, 772.729980,
        774.950989, 777.171997, 779.393005, 781.614014, 783.835999, 786.057007,
        788.278015, 790.499023, 792.719971, 794.940979, 797.161987, 799.382996,
        801.604004, 803.825012, 806.046021, 808.267029, 810.489014, 812.710022,
        814.931030, 817.151978, 819.372986, 821.593994, 823.815002, 826.036011,
        828.257019, 830.478027, 832.698975, 834.919983, 837.140991, 839.362976,
        841.583984, 843.804993, 846.026001, 848.247009, 850.468018, 852.689026,
        854.909973, 857.130981, 859.351990, 861.572998, 863.794006, 866.015991,
        868.237000, 870.458008, 872.679016, 874.900024, 877.120972, 879.341980,
        881.562988, 883.783997, 886.005005, 888.226013, 890.447021, 892.668030,
        894.890015, 897.111023, 899.331970, 901.552979, 903.773987, 905.994995,
        908.216003, 910.437012, 912.658020, 914.879028, 917.099976, 919.320984,
        921.543030, 923.763977, 925.984985, 928.205994, 930.427002, 932.648010,
        934.869019, 937.090027, 939.310974, 941.531982, 943.752991, 945.973999,
        948.195007, 950.416992, 952.638000, 954.859009, 957.080017, 959.301025,
        961.521973, 963.742981, 965.963989, 968.184998, 970.406006, 972.627014,
        974.848022, 977.070007, 979.291016, 981.512024, 983.732971, 985.953979,
        988.174988, 990.395996, 992.617004, 994.838013, 997.059021, 999.280029]
    HyperFree = HyperFree_model_registry["vit_b"](checkpoint=pretrained_ckpt, image_size=1024, vit_patch_size=8, \
        encoder_global_attn_indexes=[5, 8, 11], merge_indexs = [3, 12]).to(device)
    
    # large version
    # HyperFree = HyperFree_model_registry["vit_l"](checkpoint=pretrained_ckpt, image_size=1024, vit_patch_size=8, \
    # encoder_global_attn_indexes=[11, 17, 23], merge_indexs = [5, 24]).to(device)

    # huge version
    # HyperFree = HyperFree_model_registry["vit_h"](checkpoint=pretrained_ckpt, image_size=1024, vit_patch_size=8, \
    # encoder_global_attn_indexes=[15, 23, 31], merge_indexs = [7, 32]).to(device)

    Hyperfree_predictor = HyperFree_Predictor(HyperFree)
    net = UPerNet(num_class=22, fpn_dim=256, GSD=GSD, wavelengths = wavelengths, hyperfree_predictor=Hyperfree_predictor)
    net = net.to(device)
    out = net(img)
    
    print(f"{out.shape}")

