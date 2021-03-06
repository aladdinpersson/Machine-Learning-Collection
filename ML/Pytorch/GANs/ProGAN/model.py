"""
Implementation of ProGAN generator and discriminator with the key
attributions from the paper. We have tried to make the implementation
compact but a goal is also to keep it readable and understandable.
Specifically the key points implemented are:

1) Progressive growing (of model and layers)
2) Minibatch std on Discriminator
3) Normalization with PixelNorm
4) Equalized Learning Rate (here I cheated and only did it on Conv layers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

"""
Factors is used in Discrmininator and Generator for how much
the channels should be multiplied and expanded for each layer,
so specifically the first 5 layers the channels stay the same,
whereas when we increase the img_size (towards the later layers)
we decrease the number of chanels by 1/2, 1/4, etc.
"""
factors = [1, 1, 1, 1, 1/2, 1/4, 1/4, 1/8, 1/16]


class WSConv2d(nn.Module):
    """
    Weight scaled Conv2d (Equalized Learning Rate)
    Note that input is multiplied rather than changing weights
    this will have the same result.

    Inspired by:
    https://github.com/nvnbny/progressive_growing_of_gans/blob/master/modelUtils.py
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2
    ):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.scale = (gain / (self.conv.weight[0].numel())) ** 0.5

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x * self.scale)


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(
            torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon
        )


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixelnorm
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_size, img_channels=3):
        super(Generator, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])

        # initial takes 1x1 -> 4x4
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        # Create progression blocks and rgb layers
        channels = in_channels

        # we need to double img for log2(img_size/4) and
        # +1 in loop for initial 4x4
        for idx in range(int(log2(img_size/4)) + 1):
            conv_in = channels
            conv_out = int(in_channels*factors[idx])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out))
            self.rgb_layers.append(WSConv2d(conv_out, img_channels, kernel_size=1, stride=1, padding=0))
            channels = conv_out

    def fade_in(self, alpha, upscaled, generated):
        #assert 0 <= alpha <= 1, "Alpha not between 0 and 1"
        #assert upscaled.shape == generated.shape
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps):
        upscaled = self.initial(x)
        out = self.prog_blocks[0](upscaled)

        if steps == 0:
            return self.rgb_layers[0](out)

        for step in range(1, steps+1):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        # The number of channels in upscale will stay the same, while
        # out which has moved through prog_blocks might change. To ensure
        # we can convert both to rgb we use different rgb_layers
        # (steps-1) and steps for upscaled, out respectively
        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)


class Discriminator(nn.Module):
    def __init__(self, img_size, z_dim, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])

        # Create progression blocks and rgb layers
        channels = in_channels
        for idx in range(int(log2(img_size/4)) + 1):
            conv_in = int(in_channels * factors[idx])
            conv_out = channels
            self.rgb_layers.append(WSConv2d(img_channels, conv_in, kernel_size=1, stride=1, padding=0))
            self.prog_blocks.append(ConvBlock(conv_in, conv_out, use_pixelnorm=False))
            channels = conv_in

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # +1 to in_channels because we concatenate from minibatch std
        self.conv = WSConv2d(in_channels + 1, z_dim, kernel_size=4, stride=1, padding=0)
        self.linear = nn.Linear(z_dim, 1)

    def fade_in(self, alpha, downscaled, out):
        """Used to fade in downscaled using avgpooling and output from CNN"""
        #assert 0 <= alpha <= 1, "Alpha needs to be between [0, 1]"
        #assert downscaled.shape == out.shape
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0)
            .mean()
            .repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        out = self.rgb_layers[steps](x) # convert from rgb as initial step

        if steps == 0: # i.e, image is 4x4
            out = self.minibatch_std(out)
            out = self.conv(out)
            return self.linear(out.view(-1, out.shape[1]))

        # index steps which has the "reverse" fade_in
        downscaled = self.rgb_layers[steps - 1](self.avg_pool(x))
        out = self.avg_pool(self.prog_blocks[steps](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(steps - 1, 0, -1):
            downscaled = self.avg_pool(out)
            out = self.prog_blocks[step](downscaled)

        out = self.minibatch_std(out)
        out = self.conv(out)
        return self.linear(out.view(-1, out.shape[1]))


if __name__ == "__main__":
    import time
    Z_DIM = 100
    IN_CHANNELS = 16
    img_size = 512
    num_steps = int(log2(img_size / 4))
    x = torch.randn((5, Z_DIM, 1, 1))
    gen = Generator(Z_DIM, IN_CHANNELS, img_size=img_size)
    disc = Discriminator(img_size, Z_DIM, IN_CHANNELS)
    start = time.time()
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        z = gen(x, alpha=0.5, steps=num_steps)
    print(prof)
    gen_time = time.time()-start
    t = time.time()
    out = disc(z, 0.01, num_steps)
    disc_time = time.time()-t
    print(gen_time, disc_time)
    #print(disc(z, 0.01, num_steps).shape)
