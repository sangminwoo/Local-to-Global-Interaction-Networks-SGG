import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

############### Split (Coord-Conv) ################
class AddCoordinates:
    def __init__(self, with_r=False):
        self.with_r = with_r

    def __call__(self, image):
        batch_size, _, image_height, image_width = image.size()

        y_coords = 2.0 * torch.arange(image_height).float().unsqueeze(
            1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
        x_coords = 2.0 * torch.arange(image_width).float().unsqueeze(
            0).expand(image_height, image_width) / (image_width - 1.0) - 1.0

        coords = torch.stack((y_coords, x_coords), dim=0)

        if self.with_r:
            rs = ((y_coords ** 2) + (x_coords ** 2)) ** 0.5
            rs = rs / torch.max(rs)
            rs = torch.unsqueeze(rs, dim=0)
            coords = torch.cat((coords, rs), dim=0)

        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)

        image = torch.cat((coords.to(image.device), image), dim=1)

        return image


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_r=False):
        super(CoordConv, self).__init__()

        in_channels += 2
        if with_r:
            in_channels += 1

        self.conv_layer = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    groups=groups, bias=bias)
        self.coord_adder = AddCoordinates(with_r)

    def forward(self, x):
        x = self.coord_adder(x)
        x = self.conv_layer(x)

        return x


############### Entity-Interact ################
class RelationalEmbedding(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(RelationalEmbedding, self).__init__()
        self.rel_embedding = nn.Sequential(
            nn.Linear(3 * in_dim, hid_dim),
            nn.ReLU(True),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, subj, obj, bg):
        sob = torch.cat((subj, obj, bg), dim=1) # NxD*3
        sbo = torch.cat((subj, bg, obj), dim=1) # NxD*3
        bso = torch.cat((bg, subj, obj), dim=1) # NxD*3

        rel_sob = self.rel_embedding(sob) # NxD*3 -> NxO
        rel_sbo = self.rel_embedding(sbo) # NxD*3 -> NxO
        rel_bso = self.rel_embedding(bso) # NxD*3 -> NxO

        rel_emb = rel_sob + rel_sbo + rel_bso # NxO

        return rel_emb # NxO