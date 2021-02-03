import torch
import os
import argparse
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--n", required=True, help="number of training epoch")
parser.add_argument("--b", required=True, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("--b1", type=float, default=0.05, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--s", type=int, default=64, help="size of each image dimension")
parser.add_argument("--c", type=int, default=3, help="number of image channel")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feature, out_feature, normalization=True):
            layers = [nn.Linear(in_feature, out_feature)]
            if normalization:
                layers.append(nn.BatchNorm1d(out_feature, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalization=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )


    def forwart(self, z):



