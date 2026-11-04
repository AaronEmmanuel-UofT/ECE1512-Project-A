"""Gated attention MIL; implementation of the supplied Project A scaffold."""
import torch
from torch import nn
import torch.nn.functional as F

class Classifier1FC(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super().__init__()
        self.dropout = nn.Dropout(droprate) if droprate > 0 else nn.Identity()
        self.fc = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        return self.fc(self.dropout(x))

class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim):
        super().__init__()
        self.fc = nn.Linear(n_channels, m_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class AttentionGated(nn.Module):
    def __init__(self, L=256, D=128, K=1):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_weights_w = nn.Linear(D, K)

    def forward(self, x):
        return self.attention_weights_w(
            self.attention_V(x) * self.attention_U(x)).transpose(1, 2)

class ABMIL(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = AttentionGated(conf.D_inner, conf.D_attn, K=1)
        self.classifier = Classifier1FC(conf.D_inner, conf.n_class, conf.droprate)

    def _encode_attention(self, x, mask=None):
        # x: [slides, patches, features]. Mask is True for real patches.
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError('Expected nonempty [B,N,D] bags')
        med_feat = self.dimreduction(x)
        a_logits = self.attention(med_feat)
        if mask is not None:
            if mask.dtype != torch.bool or mask.shape != x.shape[:2]:
                raise ValueError('Mask must be bool [B,N]')
            if not mask.any(dim=1).all():
                raise ValueError('Each slide needs at least one valid patch')
            a_logits = a_logits.masked_fill(~mask.unsqueeze(1), float('-inf'))
        attention = F.softmax(a_logits, dim=-1)
        return med_feat, attention

    def forward(self, x, return_attention=False, mask=None):
        med_feat, attention = self._encode_attention(x, mask)
        bag_feat = torch.bmm(attention, med_feat).squeeze(1)
        logits = self.classifier(bag_feat)
        return (logits, attention) if return_attention else logits
