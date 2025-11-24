import torch
from torch import nn
import torch.nn.functional as F


class BaselineModel(nn.Module):
    """简单的baseline模型，MLP融合"""
    def __init__(self, hyp_params):
        super(BaselineModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.embed_dropout = hyp_params.embed_dropout
        self.out_dropout = hyp_params.out_dropout
        self.d_common = 30
        self.partial_mode = self.lonly + self.aonly + self.vonly
        
        self.proj_l = nn.Linear(self.orig_d_l, self.d_common)
        self.proj_a = nn.Linear(self.orig_d_a, self.d_common)
        self.proj_v = nn.Linear(self.orig_d_v, self.d_common)
        
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_common
        else:
            combined_dim = self.partial_mode * self.d_common
        
        output_dim = hyp_params.output_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(combined_dim, combined_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.out_dropout),
            nn.Linear(combined_dim * 2, combined_dim),
            nn.ReLU(),
            nn.Dropout(self.out_dropout),
            nn.Linear(combined_dim, output_dim)
        )
        
    def forward(self, x_l, x_a, x_v):
        proj_l = self.proj_l(x_l)
        proj_a = self.proj_a(x_a)
        proj_v = self.proj_v(x_v)
        
        proj_l = F.dropout(proj_l, p=self.embed_dropout, training=self.training)
        proj_a = F.dropout(proj_a, p=self.embed_dropout, training=self.training)
        proj_v = F.dropout(proj_v, p=self.embed_dropout, training=self.training)
        
        pooled_l = torch.mean(proj_l, dim=1)
        pooled_a = torch.mean(proj_a, dim=1)
        pooled_v = torch.mean(proj_v, dim=1)
        
        if self.partial_mode == 3:
            combined = torch.cat([pooled_l, pooled_a, pooled_v], dim=1)
        elif self.lonly:
            combined = torch.cat([pooled_l, pooled_l], dim=1)
        elif self.aonly:
            combined = torch.cat([pooled_a, pooled_a], dim=1)
        elif self.vonly:
            combined = torch.cat([pooled_v, pooled_v], dim=1)
        else:
            raise ValueError("Invalid partial mode")
        
        output = self.fusion_net(combined)
        return output, combined

