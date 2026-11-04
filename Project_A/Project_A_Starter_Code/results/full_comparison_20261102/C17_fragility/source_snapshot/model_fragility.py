"""Research prototype: use evidence-deletion sensitivity to choose mean pooling.

No new trainable parameters; not a causal explanation or calibrated uncertainty.
See research/PROTOCOL.md for the hypothesis, prior art and fixed experiment plan.
Original baseline modules are deliberately unchanged.
"""
import math
import torch
from model import ABMIL


def suppress_top_fraction(scores, valid, fraction):
    """Return re-normalized attention after a tie-symmetric top-fraction probe.

    For N>1 remove k=min(N-1, ceil(fraction*N)) patch equivalents. If scores
    tie at the cutoff, divide the remaining removal equally over the tie group.
    This avoids selecting arbitrary positions and preserves order invariance.
    The probe is diagnostic: callers compute it without gradient tracking.
    """
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError('probe_fraction must be finite and in [0,1)')
    if scores.ndim != 2 or valid.shape != scores.shape or valid.dtype != torch.bool:
        raise ValueError('Expected scores and Boolean validity mask [B,N]')
    if not valid.any(1).all():
        raise ValueError('Every slide needs a valid patch')
    retained = valid.to(scores.dtype)
    removed_counts = scores.new_zeros(scores.shape[0])
    for b in range(scores.shape[0]):
        n = int(valid[b].sum())
        if n <= 1 or fraction == 0:
            continue
        k = min(n-1, int(math.ceil(fraction*n)))
        cutoff = torch.topk(scores[b, valid[b]], k).values[-1]
        above = valid[b] & (scores[b] > cutoff)
        tied = valid[b] & (scores[b] == cutoff)
        share = (k-int(above.sum()))/int(tied.sum())
        retained[b, above] = 0
        retained[b, tied] = 1-share
        removed_counts[b] = k
    # Use original scores rather than dividing tiny surviving softmax weights.
    log_retained = retained.log()  # log(0) = -inf, intentionally excluded
    probed = (scores+log_retained).softmax(1)
    return probed, retained, removed_counts


class FragilityGuidedMIL(ABMIL):
    """alpha = TV(p_attention, p_suppressed); z=(1-alpha)*z_att+alpha*z_mean.

    Alpha is a detached, label-free sensitivity measurement recomputed on each
    slide during both training and evaluation. It is not a learned gate. Ordinary
    cross-entropy trains the original ABMIL parameters through the mixed vector.
    """
    def __init__(self, conf):
        super().__init__(conf)
        self.probe_fraction = float(getattr(conf, 'probe_fraction', 0.1))
        if not math.isfinite(self.probe_fraction) or not 0 <= self.probe_fraction < 1:
            raise ValueError('probe_fraction must be finite and in [0,1)')
        if float(conf.droprate) != 0:
            raise ValueError('Research protocol requires dropout=0 for comparable probes')

    def forward(self, x, return_attention=False, mask=None, return_diagnostics=False):
        # Same validity checks and learned layers as the frozen ABMIL baseline.
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError('Expected nonempty [B,N,D] bags')
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if mask is None else mask
        if valid.dtype != torch.bool or valid.shape != x.shape[:2]:
            raise ValueError('Mask must be bool [B,N]')
        if not valid.any(1).all():
            raise ValueError('Each slide needs at least one valid patch')
        h = self.dimreduction(x)
        scores = self.attention(h).squeeze(1).masked_fill(~valid, float('-inf'))
        a = scores.softmax(1)
        z_att = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        z_mean = (h*valid.unsqueeze(-1)).sum(1)/valid.sum(1, keepdim=True)
        with torch.no_grad():
            a_probe, retained, removed_counts = suppress_top_fraction(scores.detach(), valid, self.probe_fraction)
            z_probe = torch.bmm(a_probe.unsqueeze(1), h.detach()).squeeze(1)
            p_att = self.classifier(z_att.detach()).softmax(1)
            p_probe = self.classifier(z_probe).softmax(1)
            alpha = (0.5*(p_att-p_probe).abs().sum(1)).clamp(0, 1)
        z = (1-alpha.unsqueeze(1))*z_att + alpha.unsqueeze(1)*z_mean
        logits = self.classifier(z)
        if return_diagnostics:
            effective = (1-alpha.unsqueeze(1))*a + alpha.unsqueeze(1)*valid/valid.sum(1, keepdim=True)
            diagnostics = dict(alpha=alpha, attention=a.detach(), effective_weights=effective.detach(),
                attention_probabilities=p_att, probe_probabilities=p_probe,
                removed_patch_equivalents=removed_counts,
                removed_attention_mass=(a.detach()*(1-retained)).sum(1),
                valid_patches=valid.sum(1), probe_attention=a_probe)
            return logits, diagnostics
        return (logits, a.unsqueeze(1)) if return_attention else logits
