import torch
import torch.nn as nn
from localutils.SPPF import *
from localutils.ResidualBlock import *
from localutils.VGGBlock import *

def weights_init_normal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        # (3, 256, 256)+(3, 256, 256)=(6, 256, 256)
        img_input = torch.cat((img_A, img_B), 1)
        # (6, 256, 256) -> (1, 16, 16)
        return self.model(img_input)


class SPPF_Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(SPPF_Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = nn.Sequential(SPPF777(in_filters,out_filters),
                                   nn.MaxPool2d(kernel_size=2,stride=2),
                                   )
            return layers


        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        # (3, 256, 256)+(3, 256, 256)=(6, 256, 256)
        img_input = torch.cat((img_A, img_B), 1)
        # (6, 256, 256) -> (1, 16, 16)
        return self.model(img_input)


class SPPF7_Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(SPPF7_Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = nn.Sequential(SPPF777(in_filters,out_filters),
                                   nn.MaxPool2d(kernel_size=7,stride=2,padding=3),
                                   )
            return layers


        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        # (3, 256, 256)+(3, 256, 256)=(6, 256, 256)
        img_input = torch.cat((img_A, img_B), 1)
        # (6, 256, 256) -> (1, 16, 16)
        return self.model(img_input)

class Vgg_Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Vgg_Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = nn.Sequential(VGGBlock(in_filters, out_filters,out_filters),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   )
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        # (3, 256, 256)+(3, 256, 256)=(6, 256, 256)
        img_input = torch.cat((img_A, img_B), 1)
        # (6, 256, 256) -> (1, 16, 16)
        return self.model(img_input)


class Res_Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Res_Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = nn.Sequential(nn.Conv2d(in_filters,out_filters,kernel_size=3,stride=1,padding=1),
                                    ResidualBlock( out_filters),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   )
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        # (3, 256, 256)+(3, 256, 256)=(6, 256, 256)
        img_input = torch.cat((img_A, img_B), 1)
        # (6, 256, 256) -> (1, 16, 16)
        return self.model(img_input)

if __name__ == '__main__':
    A = torch.randn(1,3,256,256)
    B = torch.randn(1,3,256,256)
    layer = SPPF7_Discriminator()
    y = layer(A,B)
    print(y.shape)
