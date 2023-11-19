import torch
import torch.nn as nn
from collections import deque

from .unet_parts import *


class Unet(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # Contraction path.
        self.conv_blk_1 = ConvBlock(3, 64)
        self.conv_blk_2 = ConvBlock(64, 64)
        self.down_1 = Downsample()
        self.conv_blk_3 = ConvBlock(64, 128)
        self.conv_blk_4 = ConvBlock(128, 128)
        self.down_2 = Downsample()
        self.conv_blk_5 = ConvBlock(128, 256)
        self.conv_blk_6 = ConvBlock(256, 256)
        self.down_3 = Downsample()
        self.conv_blk_7 = ConvBlock(256, 512)
        self.conv_blk_8 = ConvBlock(512, 512)
        self.down_4 = Downsample()
        self.conv_blk_9 = ConvBlock(512, 1024)
        self.conv_blk_10 = ConvBlock(1024, 1024)

        # Expansion path.
        self.up_1 = Upsample()
        self.padcat_1 = PadCat()
        self.conv_blk_11 = ConvBlock(1024 + 512, 512)
        self.conv_blk_12 = ConvBlock(512, 512)
        self.up_2 = Upsample()
        self.padcat_2 = PadCat()
        self.conv_blk_13 = ConvBlock(512 + 256, 256)
        self.conv_blk_14 = ConvBlock(256, 256)
        self.up_3 = Upsample()
        self.padcat_3 = PadCat()
        self.conv_blk_15 = ConvBlock(256 + 128, 128)
        self.conv_blk_16 = ConvBlock(128, 128)
        self.up_4 = Upsample()
        self.padcat_4 = PadCat()
        self.conv_blk_17 = ConvBlock(128 + 64, 64)
        self.conv_blk_18 = ConvBlock(64, 64)

        # Final conv.
        self.conv_out = ConvOut(64, 2)

    def forward(self, x):
        intrim_feats = deque()
        x = self.conv_blk_1(x)
        x = self.conv_blk_2(x)
        intrim_feats.append(x.clone())
        x = self.down_1(x)

        x = self.conv_blk_3(x)
        x = self.conv_blk_4(x)
        intrim_feats.append(x.clone())
        x = self.down_2(x)

        x = self.conv_blk_5(x)
        x = self.conv_blk_6(x)
        intrim_feats.append(x)
        x = self.down_3(x)

        x = self.conv_blk_7(x)
        x = self.conv_blk_8(x)
        intrim_feats.append(x)
        x = self.down_4(x)

        x = self.conv_blk_9(x)
        x = self.conv_blk_10(x)
        x = self.up_1(x)
        y = intrim_feats.pop()
        x = self.padcat_1(x, y)

        x = self.conv_blk_11(x)
        x = self.conv_blk_12(x)
        x = self.up_2(x)
        y = intrim_feats.pop()
        x = self.padcat_2(x, y)

        x = self.conv_blk_13(x)
        x = self.conv_blk_14(x)
        x = self.up_3(x)
        y = intrim_feats.pop()
        x = self.padcat_3(x, y)

        x = self.conv_blk_15(x)
        x = self.conv_blk_16(x)
        x = self.up_4(x)
        y = intrim_feats.pop()
        x = self.padcat_4(x, y)

        x = self.conv_blk_17(x)
        x = self.conv_blk_18(x)

        x = self.conv_out(x)
        return x


tens = torch.rand(size=(1, 3, 256, 256), dtype=torch.float32)
tens = F.pad(tens, (4, 4, 4, 4))
model = Unet()
tens = model(tens)