import torch
import torch.nn as nn
import torch.nn.functional as F 


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, **generator_kwargs):
        super(Generator, self).__init__()

        # c7s1-64
        modules = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # d128, d256
        in_channels = 64
        out_channels = in_channels * 2
        for _ in range(2):
            modules += [
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
            out_channels = in_channels * 2

        # r256 x 9times
        for _ in range(generator_kwargs['n_residual_blocks']):
            modules.append(ResidualBlock(in_channels))

        # u128, u64
        out_channels = in_channels // 2
        for _ in range(2):
            modules += [
                # fractional-strided-convolution (stride 1/2) -> ConvTranspose2d with stride 2
                nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
            out_channels = in_channels // 2

        # c7s1-3
        modules += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*modules)

        self.apply(weights_init_normal)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels, **discriminator_kwargs):
        super(Discriminator, self).__init__()

        modules = []
        prev_channels = input_channels
        for i, channels in enumerate(discriminator_kwargs['channels_list']):
            modules.append(nn.Conv2d(prev_channels, channels, discriminator_kwargs['kernel_size'], stride=2, padding=1))
            modules.append(nn.LeakyReLU(0.2, inplace=True))
            if i != 0:
                modules.append(nn.InstanceNorm2d(channels))
            prev_channels = channels

        modules.append(nn.Conv2d(channels, 1, discriminator_kwargs['kernel_size'], padding=1))

        self.model = nn.Sequential(*modules)

        self.apply(weights_init_normal)

    def forward(self, x):
        x = self.model(x)
        # return x
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
