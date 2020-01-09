import torch
import torch.nn as nn
from spectral import SpectralNorm
from torch.nn import functional as F

# Channel Attention
class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, H, W)
        outputs :
            feature maps weighted by attention along a channel dimension
        """

        N, C, H, W = x.shape
        query = x.view(N, C, -1)    # (N, C, H*W)
        key = x.view(N, C, -1).permute(0, 2, 1)    # (N, H*W, C)

        # calculate correlation
        energy = torch.bmm(query, key)    # (N, C, C)
        energy = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy)

        value = x.view(N, C, -1)

        out = torch.bmm(attention, value)
        out = out.view(N, C, H, W)
        out = self.gamma*out + x
        return out

# generator
class generator(torch.nn.Module):
    def __init__(self, in_nc, out_nc, nf):
        super(generator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf

        self.downconv1 = nn.Sequential(  # input H,W 3 output H,W 64  1
            SpectralNorm(nn.Conv2d(in_nc, nf, 5, 1, 2)),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
        )

        self.downconv2 = nn.Sequential(  # input H,W 64 output H/2,W/2 128  2
            SpectralNorm(nn.Conv2d(nf, nf * 2, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
        )

        self.downconv3 = nn.Sequential(  # input H/2,W/2 128 output H/4,W/4 256  3
            SpectralNorm(nn.Conv2d(nf * 2, nf * 4, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),

        )

        self.downconv4 = nn.Sequential(  # input H/4,W/4 256 output H/8,W/8 512  4
            SpectralNorm(nn.Conv2d(nf * 4, nf * 8, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.ReLU(True),

        )

        self.downconv5 = nn.Sequential(  # input H/8,W/8 512 output H/8,W/8 512  5
            SpectralNorm(nn.Conv2d(nf * 8, nf * 8, 1, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.ReLU(True),

        )

        self.upconv3 = nn.Sequential(  # input H/8,W/8 1024 output H/4,W/4 256  6
            SpectralNorm(nn.ConvTranspose2d(nf * 16, nf * 4, 4, 2, 1)),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.upconv2 = nn.Sequential(  # input H/4,W/4 512 output H/2,W/2 128  7
            SpectralNorm(nn.ConvTranspose2d(nf * 8, nf * 2, 4, 2, 1)),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
        )

        self.upconv1 = nn.Sequential(  # input H/2,W/2 256 output H,W 3  8
            SpectralNorm(nn.ConvTranspose2d(nf * 4, nf, 4, 2, 1)),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),

        )

        self.CAM = ChannelAttentionModule()


    def forward(self, image):

        x1 = self.downconv1(image) #64 H,W
        x2 = self.downconv2(x1) #128 H/2,W/2
        x3 = self.downconv3(x2) #256 H/4,W/4
        x4 = self.downconv4(x3) #512 H/8,W/8
        x5 = self.downconv5(x4) #512 H/8,W/8

        y3 = self.upconv3(torch.cat([x4, x5], dim=1)) #256 H/4,W/4
        y3 = self.CAM(y3)
        y2 = self.upconv2(torch.cat([y3, x3], dim=1)) #128 H/2,W/2
        y2 = self.CAM(y2)
        y1 = self.upconv1(torch.cat([y2, x2], dim=1)) #64 H,W
        output = self.CAM(y1)

        return output

class segmantation(nn.Module):
    def __init__(self, in_nc, out_nc, nf):
        super(segmantation, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf

        self.segmnet = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, 22, 3, 1, 1),
        )

    def forward(self, x):

        x = self.segmnet(x)

        return x

class colorization(nn.Module):
    def __init__(self, in_nc, out_nc, nf):
        super(colorization, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf

        self.colornet = nn.Sequential(
            nn.Conv2d(nf, out_nc, 5, 1, 2),
            nn.Tanh(),
        )

    def forward(self, x):

        x = self.colornet(x)

        return x

# discriminator
class discriminator(nn.Module):
    def __init__(self, in_nc, out_nc, nf=32):
        super(discriminator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.dis = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_nc, nf, 3, 2, 1)),
            nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(nf, nf * 2, 3, 1, 1)),
            nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, True),

            SpectralNorm(nn.Conv2d(nf * 2, nf * 4, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(nf * 4, nf * 8, 3, 1, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),

            SpectralNorm(nn.Conv2d(nf * 8, nf * 8, 3, 1, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, out_nc, 3, 1, 1),
        )

    def forward(self, input):

        output = self.dis(input)

        return F.avg_pool2d(output, output.size()[2:]).view(output.size()[0], -1)


class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs