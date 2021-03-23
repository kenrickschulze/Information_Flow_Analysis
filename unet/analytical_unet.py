"""
The standard UNet for medical image segmentation

Adapted from https://github.com/jvanvugt/pytorch-unet,
which itself is adapted from
https://discuss.pytorch.org/t/unet-implementation/426.

Copyright (c) 2018 Joris
"""

import torch
from torch import nn
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
noise_variance =  .05

def Kget_dists(X):
    pass
#TODO faster way to get entropy

def entropy_estimator_kl(activations, var=0.05):
    # Upper Bound
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    N, dims = activations.shape
    dists = euclidean_distances(activations, activations, squared=True)

    dists2 = dists / (2*var)
    normconst = (dims/2.0)*np.log(2*np.pi*var)
    lprobs = np.log(np.sum(np.exp(-dists2), axis=1)) - np.log(N) - normconst
    h = -np.mean(lprobs, axis=0)
    return dims/2 + h
def entropy_estimator_bd(x, var= .05):
    # Lower bound of Marginal Entropy
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    N, dims = x.shape
    val = entropy_estimator_kl(x,4*var)
    return val + np.log(0.25)*dims/2

def kde_condentropy(output, var= 0.05):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.shape[1]
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)


class AnalyticalUNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            n_classes: int = 1,
            depth: int = 5,
            wf: int = 6,
            kernel_size: int = 3,
            padding: bool = False,
            batch_norm: bool = False,
            up_mode: str = "upconv",
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(AnalyticalUNet, self).__init__()
        assert up_mode in ("upconv", "upsample")

        self.padding = padding
        self.depth = depth
        self.activations = []
        self.counter_2conv = 0
        self.maxpool2D = nn.MaxPool2d(kernel_size=2)
        self.down_path = nn.ModuleList()
        prev_channels = in_channels
        for i in range(depth - 1):
            self.down_path.append(
                UNetConvBlock(
                    prev_channels, 2 ** (wf + i), kernel_size, padding, batch_norm
                )
            )
            # self.down_path.append(nn.MaxPool2d(2))
            prev_channels = 2 ** (wf + i)

        # bottom down conv
        self.down_path.append(
            UNetConvBlock(
                prev_channels, 2 ** (wf + depth - 1), kernel_size, padding, batch_norm
            )
        )
        prev_channels = 2 ** (wf + depth - 1)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(
                    prev_channels,
                    2 ** (wf + i),
                    kernel_size,
                    up_mode,
                    padding,
                    batch_norm,
                )
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x, conv_acts, conc_acts):
        blocks = []

        for i, down in enumerate(self.down_path):

            x = down(x)
            # activation of x
            conv_acts.append(x)

            # for all but last layer
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = self.maxpool2D(x)
        for i, up in enumerate(self.up_path):

            x, up_acti = up(x, blocks[-i - 1])

            # activations of x
            conv_acts.append(x)
            conc_acts.append(up_acti)


        last = self.last(x).reshape(-1, *x.shape[-2:])
        conv_acts.append(last)

        return last, conv_acts, conc_acts


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=int(padding))
        )
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(
            nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=int(padding))
        )
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(
            in_size, out_size, kernel_size, padding, batch_norm
        )

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
               :, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
               ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        # extract activations after concatenating two blocks
        a = out.clone().detach()
        out = self.conv_block(out)

        return out, a


if __name__ == "__main__":
    depth = 5
    n_2convs = depth * 2 - 1
    WF=4

    net = AnalyticalUNet(padding=True, wf=4, depth=5)

    x0 = torch.randn(1, 1, 128, 128)
    x1 = torch.randn(1, 1, 128, 128)
    x2 = torch.randn(1, 1, 128, 128)

    size = x0.size()[-1]
    N_SAMPLES = 3

    # create container for the activations of a single epoch
    encoder_act = [np.zeros((2**(WF+i)*int((size/(2**i)))**2,N_SAMPLES))
                     for i, w in enumerate(range(depth))]
    a_rev = [m for m in encoder_act[-2::-1]]
    act_conc_cont = [m for m in a_rev[1:]]

    act_conc_cont.append(np.zeros((32*size**2, N_SAMPLES)))
    a_rev.append(np.zeros((size**2,N_SAMPLES)))
    act_container = encoder_act + a_rev
    print('predicted shapes')




    ##### start run #####
    print('START')

    for sidx, sample in enumerate([x0, x1, x2]):
        act_conv = []
        act_conc = []
        out, acts, conc_acts = net(sample,act_conv, act_conc)

        for layer_type, layer_cont in zip([acts, conc_acts],[act_container,act_conc_cont]):
            for lidx in range(len(layer_type)):
                layer_cont[lidx][:,sidx] = torch.Tensor.cpu(layer_type[lidx]).detach().numpy().flatten()


    ### calculate mutual information
    # activations after conv
    for t_i, T in enumerate(act_container):
        print(T.shape)
        #entropy = entropy_estimator_kl(T)
        print(f'Size of T_{t_i}= {T.shape}')


