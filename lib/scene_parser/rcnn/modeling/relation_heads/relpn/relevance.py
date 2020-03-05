import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import box_pos_encoder

class Relevance(nn.Module):
    """
    calculate relavance score between subjects and objects
    """
    def __init__(self, num_classes, embed_dim=128, pos_embed=True):
        super(Relevance, self).__init__()

        self.logit_emb = nn.Sequential(
            nn.Linear(num_classes, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )

        self.pos_embed = pos_embed
        if self.pos_embed:
            self.pos_emb = nn.Sequential(
                nn.Linear(6, embed_dim),
                nn.ReLU(True),
                nn.Linear(embed_dim, embed_dim)
            )

    def forward(self, vis_feats, logits, bbox=None, imsize=None):
        vis_feats = vis_feats.squeeze() # kx2048x1x1 -> kx2048
        vis_scores = torch.mm(vis_feats, vis_feats.t()) # kxk
        vis_score = torch.sigmoid(vis_scores)

        logits_subj = self.logit_subj(logits) # k x 128
        logits_obj = self.logit_obj(logits)   # k x 128
        logits_scores = torch.mm(logits_subj, logits_obj.t()) # k x k
        logits_scores = torch.sigmoid(logits_scores)

        scores = vis_scores + logits_scores

        if self.pos_embed:
            pos = box_pos_encoder(bbox, imsize[0], imsize[1])
            pos = self.pos_emb(pos) # k x 128
            pos_scores = torch.mm(pos, pos.t()) # k x k
            pos_scores = torch.sigmoid(pos_scores)

            scores = scores + pos_scores

        # relness = torch.sigmoid(scores)
        return relness # k x k