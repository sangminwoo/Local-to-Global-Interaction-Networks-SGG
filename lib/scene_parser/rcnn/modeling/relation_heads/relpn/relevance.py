import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import box_pos_encoder

class Relevance(nn.Module):
    """
    calculate relavance score between subjects and objects
    """
    def __init__(self, vis_dim, num_classes, embed_dim=128):
        super(Relevance, self).__init__()

        self.vis_subj = nn.Sequential(
            nn.Linear(vis_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.vis_obj = nn.Sequential(
            nn.Linear(vis_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )

        self.logit_subj = nn.Sequential(
            nn.Linear(num_classes, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.logit_obj = nn.Sequential(
            nn.Linear(num_classes, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )

        self.pos_subj = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.pos_obj = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, vis_feats, logits, bbox=None, imsize=None):
        vis_subj = self.vis_subj(vis_feats) # kx128
        vis_obj = self.vis_obj(vis_feats) # kx128
        vis_scores = torch.mm(vis_subj, vis_obj.t()) # kxk
        # vis_scores = torch.sigmoid(vis_scores)

        logits_subj = self.logit_subj(logits) # k x 128
        logits_obj = self.logit_obj(logits)   # k x 128
        logits_scores = torch.mm(logits_subj, logits_obj.t()) # k x k
        # logits_scores = torch.sigmoid(logits_scores)

        pos = box_pos_encoder(bbox, imsize[0], imsize[1])
        pos_subj = self.pos_subj(pos_subj) # k x 128
        pos_obj = self.pos_obj(pos_obj) # k x 128
        pos_scores = torch.mm(pos_subj, pos_obj.t()) # k x k
        # pos_scores = torch.sigmoid(pos_scores)

        scores = vis_scores + logits_scores + pos_scores
        relness = torch.sigmoid(scores)
        
        return relness # k x k