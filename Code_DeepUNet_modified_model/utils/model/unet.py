import torch
from torch import nn
from torch.nn import functional as F

class Unet(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.first_block = ConvBlock(in_chans, 64, 64)
        self.down1 = Down(64, 64, 64)
        self.down2 = Down(64, 32, 64)
        self.down3 = Down(64, 64, 64)
        self.down4 = Downlast(64, 64, 64)
        self.up1 = Up(64, 32, 64)
        self.up2 = Up(64, 64, 64)
        self.up3 = Up(64, 32, 64)
        self.up4 = Uplast(64, 64, 64)
        self.last_block = OutConv(64, out_chans)

    def norm(self, x):
        b, h, w = x.shape
        x = x.view(b, h * w)
        mean = x.mean(dim=1).view(b, 1, 1)
        std = x.std(dim=1).view(b, 1, 1)
        x = x.view(b, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, input):
        input, mean, std = self.norm(input)
        input = input.unsqueeze(1)
        d1 = self.first_block(input)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        output = self.last_block(u4)
        output = output.squeeze(1)
        output = self.unnorm(output, mean, std)

        return output


class ConvBlock(nn.Module):

    def __init__(self, in_chans, mid_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.mid_chans = mid_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, mid_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Down(nn.Module):

    def __init__(self, in_chans, mid_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.mid_chans = mid_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_chans, mid_chans, out_chans)
        )

    def forward(self, x):
        xl = self.layers(x)
        m = nn.MaxPool2d(2)
        xplus = m(x)
        xla = torch.add(xl, xplus)
        return xla


class Downlast(nn.Module):

    def __init__(self, in_chans, mid_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.mid_chans = mid_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_chans, mid_chans, out_chans)
        )

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):

    def __init__(self, in_chans, mid_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.mid_chans = mid_chans
        self.out_chans = out_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_chans, mid_chans, out_chans)
        self.convh = nn.Conv2d(in_chans, in_chans // 2, 1)

    def forward(self, x, concat_input):
        x = self.up(x)
        concat_in = self.convh(concat_input)
        concat_output = torch.cat([concat_in, x], dim=1)
        return torch.add(self.conv(concat_output), concat_output)


class Uplast(nn.Module):

    def __init__(self, in_chans, mid_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.mid_chans = mid_chans
        self.out_chans = out_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_chans, mid_chans, out_chans)
        self.convh = nn.Conv2d(in_chans, in_chans // 2, 1)

    def forward(self, x, concat_input):
        x = self.up(x)
        concat_in = self.convh(concat_input)
        concat_output = torch.cat([concat_in, x], dim=1)
        return self.conv(concat_output)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

