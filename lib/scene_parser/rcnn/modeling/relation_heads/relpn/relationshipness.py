import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import box_pos_encoder

class Relationshipness(nn.Module):
    """
    compute relationshipness between subjects and objects
    """
    def __init__(self, vis_dim, num_classes, pos_encoding=True):
        super(Relationshipness, self).__init__()

        self.vis_subj = nn.Sequential(
            nn.Linear(vis_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 64)
        )

        self.vis_obj = nn.Sequential(
            nn.Linear(vis_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 64)
        )

        self.logit_subj = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(True),
            nn.Linear(64, 64)
        )

        self.logit_obj = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(True),
            nn.Linear(64, 64)
        )

        self.pos_encoding = pos_encoding
        if self.pos_encoding:
            self.sub_pos_encoder = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(True),
                nn.Linear(64, 64)
            )

            self.obj_pos_encoder = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(True),
                nn.Linear(64, 64)
            )

    def forward(self, features, logits, bbox=None, imsize=None):
        #features = F.avg_pool2d(features, kernel_size=(features.size(2), features.size(3))).squeeze() # 256x1024xHxW -> 256x1024x1x1 -> 256x1024

        #feats_subj = self.vis_subj(features) # k x 64
        #feats_obj = self.vis_obj(features) # k x 64
        #feats_scores = torch.mm(feats_subj, feats_obj.t()) # k x k
        #feats_scores = torch.sigmoid(feats_scores)

        logits_subj = self.logit_subj(logits) # k x 64
        logits_obj = self.logit_obj(logits)   # k x 64
        logits_scores = torch.mm(logits_subj, logits_obj.t()) # k x k
        
        score = logits_scores

        if self.pos_encoding:
            pos = box_pos_encoder(bbox, imsize[0], imsize[1])
            pos_subj = self.sub_pos_encoder(pos)
            pos_obj = self.obj_pos_encoder(pos)
            pos_scores = torch.mm(pos_subj, pos_obj.t()) # k x k

            score = logits_scores + pos_scores

        relness = torch.sigmoid(score)
        return relness # k x k