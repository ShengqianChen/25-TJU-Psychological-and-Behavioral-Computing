import torch
from torch import nn
import torch.nn.functional as F


class BaselineModel(nn.Module):
    """
    Simple baseline model for multimodal fusion.
    
    Architecture:
    1. Project each modality to a common dimension
    2. Simple concatenation of all modalities
    3. Multi-layer MLP for fusion and prediction
    
    This is a simple baseline without any attention mechanism.
    """
    def __init__(self, hyp_params):
        """
        Construct a simple baseline model.
        """
        super(BaselineModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.embed_dropout = hyp_params.embed_dropout
        self.out_dropout = hyp_params.out_dropout
        
        # Common dimension for all modalities
        self.d_common = 30
        
        self.partial_mode = self.lonly + self.aonly + self.vonly
        
        # Projection layers to common dimension
        self.proj_l = nn.Linear(self.orig_d_l, self.d_common)
        self.proj_a = nn.Linear(self.orig_d_a, self.d_common)
        self.proj_v = nn.Linear(self.orig_d_v, self.d_common)
        
        # Determine combined dimension
        if self.partial_mode == 1:
            # Single modality: use 2 * d_common (for consistency with MulT)
            combined_dim = 2 * self.d_common
        else:
            # Multiple modalities: concatenate all
            combined_dim = self.partial_mode * self.d_common
        
        output_dim = hyp_params.output_dim
        
        # Simple MLP for fusion and prediction
        # Use average pooling over time dimension, then MLP
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
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        batch_size = x_l.size(0)
        
        # Project each modality to common dimension
        # x_l: (batch_size, seq_len, orig_d_l)
        # x_a: (batch_size, seq_len, orig_d_a)
        # x_v: (batch_size, seq_len, orig_d_v)
        
        proj_l = self.proj_l(x_l)  # (batch_size, seq_len, d_common)
        proj_a = self.proj_a(x_a)  # (batch_size, seq_len, d_common)
        proj_v = self.proj_v(x_v)  # (batch_size, seq_len, d_common)
        
        # Apply dropout
        proj_l = F.dropout(proj_l, p=self.embed_dropout, training=self.training)
        proj_a = F.dropout(proj_a, p=self.embed_dropout, training=self.training)
        proj_v = F.dropout(proj_v, p=self.embed_dropout, training=self.training)
        
        # Average pooling over time dimension
        # Take mean across sequence length
        pooled_l = torch.mean(proj_l, dim=1)  # (batch_size, d_common)
        pooled_a = torch.mean(proj_a, dim=1)  # (batch_size, d_common)
        pooled_v = torch.mean(proj_v, dim=1)  # (batch_size, d_common)
        
        # Concatenate modalities
        if self.partial_mode == 3:
            # All three modalities
            combined = torch.cat([pooled_l, pooled_a, pooled_v], dim=1)  # (batch_size, 3*d_common)
        elif self.lonly:
            # Only text
            combined = torch.cat([pooled_l, pooled_l], dim=1)  # (batch_size, 2*d_common) for consistency
        elif self.aonly:
            # Only audio
            combined = torch.cat([pooled_a, pooled_a], dim=1)  # (batch_size, 2*d_common)
        elif self.vonly:
            # Only vision
            combined = torch.cat([pooled_v, pooled_v], dim=1)  # (batch_size, 2*d_common)
        else:
            raise ValueError("Invalid partial mode")
        
        # Pass through fusion network
        output = self.fusion_net(combined)  # (batch_size, output_dim)
        
        # Return output and hidden representation (for compatibility with train.py)
        return output, combined

