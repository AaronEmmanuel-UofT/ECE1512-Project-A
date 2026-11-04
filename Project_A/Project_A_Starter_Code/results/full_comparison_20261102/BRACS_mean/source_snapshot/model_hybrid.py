"""Experimental attention/mean pooling from EXPERIMENT_PLAN.md, proposal 1.

No additional trainable parameters. A hypothesis to evaluate, not an established
improvement. The mean excludes padding and is mixed before classification.
"""
import math
from model import ABMIL


class AttentionMeanMIL(ABMIL):
    def __init__(self, conf):
        super().__init__(conf)
        self.mean_weight = float(getattr(conf, 'hybrid_mean_weight', 0.5))
        if not math.isfinite(self.mean_weight) or not 0 <= self.mean_weight <= 1:
            raise ValueError('hybrid_mean_weight must be in [0,1]')

    def forward(self, x, return_attention=False, mask=None):
        if self.mean_weight == 0:
            return super().forward(x, return_attention=return_attention, mask=mask)
        med_feat, attention = self._encode_attention(x, mask)
        attention_pool = (attention @ med_feat).squeeze(1)
        if mask is None:
            mean_pool = med_feat.mean(dim=1)
        else:
            valid = mask.unsqueeze(-1)
            mean_pool = (med_feat * valid).sum(dim=1) / valid.sum(dim=1)
        bag_feat = (1-self.mean_weight)*attention_pool + self.mean_weight*mean_pool
        logits = self.classifier(bag_feat)
        return (logits, attention) if return_attention else logits
