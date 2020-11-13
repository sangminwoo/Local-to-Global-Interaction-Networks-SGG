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

    def forward_no_permute(self, subj, obj, bg):
        sob = torch.cat((subj, obj, bg), dim=1) # NxD*3

        rel_sob = self.rel_embedding(sob) # NxD*3 -> NxO

        rel_emb = rel_sob

        return rel_emb # NxO

    def forward(self, subj, obj, bg):
        sob = torch.cat((subj, obj, bg), dim=1) # NxD*3
        sbo = torch.cat((subj, bg, obj), dim=1) # NxD*3
        bso = torch.cat((bg, subj, obj), dim=1) # NxD*3

        rel_sob = self.rel_embedding(sob) # NxD*3 -> NxO
        rel_sbo = self.rel_embedding(sbo) # NxD*3 -> NxO
        rel_bso = self.rel_embedding(bso) # NxD*3 -> NxO

        rel_emb = rel_sob + rel_sbo + rel_bso # NxO

        return rel_emb # NxO

    def forward_full_permute(self, subj, obj, bg):
        sob = torch.cat((subj, obj, bg), dim=1) # NxD*3
        sbo = torch.cat((subj, bg, obj), dim=1) # NxD*3
        bso = torch.cat((bg, subj, obj), dim=1) # NxD*3
        osb = torch.cat((obj, subj, bg), dim=1) # NxD*3
        obs = torch.cat((obj, bg, subj), dim=1) # NxD*3
        bos = torch.cat((bg, obj, subj), dim=1) # NxD*3

        rel_sob = self.rel_embedding(sob) # NxD*3 -> NxO
        rel_sbo = self.rel_embedding(sbo) # NxD*3 -> NxO
        rel_bso = self.rel_embedding(bso) # NxD*3 -> NxO
        rel_osb = self.rel_embedding(osb) # NxD*3 -> NxO
        rel_obs = self.rel_embedding(obs) # NxD*3 -> NxO
        rel_bos = self.rel_embedding(bos) # NxD*3 -> NxO

        rel_emb = rel_sob + rel_sbo + rel_bso + rel_osb + rel_obs + rel_bos # NxO

        return rel_emb # NxO

def masking(union_features, proposals, rel_pair_idxs, mask_size=14):
    device = union_features.device
    bboxes = torch.cat([proposal.bbox for proposal in proposals], 0) # 20x4
    sbj_boxes = bboxes[torch.cat(rel_pair_idxs, dim=0)[:, 0]]
    obj_boxes = bboxes[torch.cat(rel_pair_idxs, dim=0)[:, 1]]
    pair_boxes = torch.cat([sbj_boxes, obj_boxes], dim=1)
    union_boxes = torch.cat((
        torch.min(sbj_boxes[:,:2], obj_boxes[:,:2]),
        torch.max(sbj_boxes[:,2:], obj_boxes[:,2:])
        ),dim=1)

    x = pair_boxes[:,[0,2,4,6]]-union_boxes[:,0].view(-1, 1).repeat(1, 4) # Nx4
    y = pair_boxes[:,[1,3,5,7]]-union_boxes[:,1].view(-1, 1).repeat(1, 4) # Nx4

    num_pairs = pair_boxes.shape[0]
    x_rescale_factor = (mask_size / torch.max(x[:,1], x[:,3])).view(-1, 1) # Nx1
    y_rescale_factor = (mask_size / torch.max(y[:,1], y[:,3])).view(-1, 1) # Nx1

    x_pooled = (x * x_rescale_factor).round().long() # Nx4
    y_pooled = (y * y_rescale_factor).round().long() # Nx4
    xy_pooled = torch.stack((x_pooled, y_pooled), dim=2) # Nx4x2

    subj_xy = xy_pooled[:, :2, :].reshape(num_pairs, 4) # Nx4
    obj_xy = xy_pooled[:, 2:, :].reshape(num_pairs, 4) # Nx4

    sbj_mask = torch.zeros(num_pairs, mask_size, mask_size, device=device) # Nx14x14
    obj_mask = torch.zeros(num_pairs, mask_size, mask_size, device=device) # Nx14x14

    for i in range(num_pairs):
        sbj_mask[i, subj_xy[i,0]:subj_xy[i,2], subj_xy[i,1]:subj_xy[i,3]] = 1
        obj_mask[i, obj_xy[i,0]:obj_xy[i,2], obj_xy[i,1]:obj_xy[i,3]] = 1

    sbj_mask = sbj_mask.view(num_pairs, 1, mask_size, mask_size) # Nx1x14x14
    obj_mask = obj_mask.view(num_pairs, 1, mask_size, mask_size) # Nx1x14x14

    bg_mask = torch.ones(num_pairs, 1, mask_size, mask_size, device=device) # Nx1x14x14
    bg_mask = bg_mask - sbj_mask - obj_mask # Nx1x14x14
    bg_mask[bg_mask < 0] = 0 # Nx1x14x14

    return union_features*sbj_mask, union_features*obj_mask, union_features*bg_mask # Nx1x14x14, Nx1x14x14, Nx1x14x14
