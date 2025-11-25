import torch
from torch import nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    """简单的MLP模块"""
    def __init__(self, input_dim, hidden_dim=None, dropout=0.1):
        super(SimpleMLP, self).__init__()
        hidden_dim = hidden_dim or input_dim * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, *args, **kwargs):
        # 忽略其他参数，只处理第一个输入
        # 输入形状: (seq_len, batch_size, embed_dim)
        seq_len, batch_size, embed_dim = x.shape
        
        # 重塑为 (seq_len * batch_size, embed_dim)
        x_flat = x.reshape(-1, embed_dim)
        
        # 通过MLP
        output_flat = self.mlp(x_flat)
        
        # 恢复形状
        output = output_flat.reshape(seq_len, batch_size, embed_dim)
        
        return output


class MULTModelMLP(nn.Module):
    def __init__(self, hyp_params):
        """
        使用MLP替换Transformer的MulT模型
        """
        super(MULTModelMLP, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.d_l = getattr(hyp_params, 'd_l', 30)
        self.d_a = getattr(hyp_params, 'd_a', 30)
        self.d_v = getattr(hyp_params, 'd_v', 30)
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.relu_dropout = hyp_params.relu_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l   # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        
        output_dim = hyp_params.output_dim

        # 1. Temporal convolutional layers (保持不变)
        k_l = getattr(hyp_params, 'kernel_size_l', 1)
        k_v = getattr(hyp_params, 'kernel_size_v', 1)
        k_a = getattr(hyp_params, 'kernel_size_a', 1)
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=k_l, padding=max(k_l - 1, 0) // 2, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=k_a, padding=max(k_a - 1, 0) // 2, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=k_v, padding=max(k_v - 1, 0) // 2, bias=False)

        # 2. 跨模态MLP (替换跨模态Transformer)
        if self.lonly:
            self.mlp_l_with_a = self.get_mlp(self_type='la')
            self.mlp_l_with_v = self.get_mlp(self_type='lv')
        if self.aonly:
            self.mlp_a_with_l = self.get_mlp(self_type='al')
            self.mlp_a_with_v = self.get_mlp(self_type='av')
        if self.vonly:
            self.mlp_v_with_l = self.get_mlp(self_type='vl')
            self.mlp_v_with_a = self.get_mlp(self_type='va')
        
        # 3. 模态内MLP (替换自注意力Transformer)
        self.mlp_l_mem = self.get_mlp(self_type='l_mem')
        self.mlp_a_mem = self.get_mlp(self_type='a_mem')
        self.mlp_v_mem = self.get_mlp(self_type='v_mem')
       
        # Projection layers (保持不变)
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_mlp(self, self_type='l'):
        """获取相应维度的MLP"""
        if self_type in ['l', 'al', 'vl']:
            embed_dim = self.d_l
        elif self_type in ['a', 'la', 'va']:
            embed_dim = self.d_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim = self.d_v
        elif self_type == 'l_mem':
            embed_dim = 2 * self.d_l
        elif self_type == 'a_mem':
            embed_dim = 2 * self.d_a
        elif self_type == 'v_mem':
            embed_dim = 2 * self.d_v
        else:
            raise ValueError("Unknown network type")
        
        return SimpleMLP(input_dim=embed_dim, dropout=self.relu_dropout)
            
    def forward(self, x_l, x_a, x_v):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
       
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.lonly:
            # 使用MLP处理跨模态信息
            # 注意：MLP版本无法直接进行跨模态注意力，这里我们简单地对每个模态应用MLP
            h_l_with_as = self.mlp_l_with_a(proj_x_l)    # 原本应该用音频信息增强文本
            h_l_with_vs = self.mlp_l_with_v(proj_x_l)    # 原本应该用视觉信息增强文本
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.mlp_l_mem(h_ls)
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.aonly:
            h_a_with_ls = self.mlp_a_with_l(proj_x_a)
            h_a_with_vs = self.mlp_a_with_v(proj_x_a)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.mlp_a_mem(h_as)
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            h_v_with_ls = self.mlp_v_with_l(proj_x_v)
            h_v_with_as = self.mlp_v_with_a(proj_x_v)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.mlp_v_mem(h_vs)
            last_h_v = last_hs = h_vs[-1]
        
        if self.partial_mode == 3:
            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output, last_hs