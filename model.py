import torch
import numpy as np


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='reflect',
        )
        self.bn1 = torch.nn.InstanceNorm2d(in_channel)
        self.conv2 = torch.nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='reflect',
        )
        self.bn2 = torch.nn.InstanceNorm2d(in_channel)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        #x = self.relu(x)

        return x


class ResNetGenerator(torch.nn.Module):
    def __init__(self, input_shape):
        super(ResNetGenerator, self).__init__()

        channels = input_shape[0]
        self.gen = torch.nn.Sequential(
            # initial convolution block
            torch.nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode='reflect',
            ),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(inplace=True),

            # downsampling
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode='reflect',
            ),
            torch.nn.InstanceNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode='reflect',
            ),
            torch.nn.InstanceNorm2d(256),
            torch.nn.ReLU(inplace=True),

            # residual blocks
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            # upsampling
            torch.nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            torch.nn.InstanceNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(inplace=True),

            # output layer
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=3,
                kernel_size=7,
                stride=1,
                padding=3
            ),
            torch.nn.Tanh()
        )

    def forward(self,x):
        return self.gen(x)

class Discriminator(torch.nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()


        channels, height, width = input_shape
        self.output_shape = (1, 6, 6)

        self.dis = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.InstanceNorm2d(256),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.InstanceNorm2d(512),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.ZeroPad2d((1,0,1,0)),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=1,
                kernel_size=4,
                padding=1
            )
        )

    def forward(self,x):
        return self.dis(x)

# if __name__ == "__main__":
#    device = "cuda" if torch.cuda.is_available() else "cpu"
#    x = torch.randn(3, 3, 224, 224).to(device)
#    model = Discriminator().to(device)
#    print(model(x).shape)