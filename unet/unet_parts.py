import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        return self.up(x)


class PadCat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x1, x2):
        """ Pad x1 to match the shape of x2 and then
            concate both tensors.
        """
        H1, W1 = x1.shape[-2:]
        H2, W2 = x2.shape[-2:]

        diff_h = abs(H2 - H1)
        pad_top, pad_down = (diff_h // 2, diff_h // 2 + 1)
        if diff_h % 2 == 0:
            pad_top = pad_down = diff_h // 2
        
        diff_w = abs(W2 - W1)
        pad_left, pad_right = (diff_w // 2, diff_w // 2 + 1)
        if diff_w % 2 == 0:
            pad_left = pad_right = diff_w // 2

        x1 = F.pad(x1, (pad_left, pad_right, pad_top, pad_down))
        x = torch.concat([x1, x2], dim=1)

        return x


class ConvOut(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.out(x)