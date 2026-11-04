"""Check the proposed pooling extension, without using real benchmark slides."""
from types import SimpleNamespace
import torch
from model import ABMIL
from model_hybrid import AttentionMeanMIL


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    conf = SimpleNamespace(D_feat=8, D_inner=16, D_attn=8, n_class=3, droprate=0.0)
    baseline = ABMIL(conf).eval()
    x = torch.randn(2, 5, 8)
    padded = torch.cat([x, torch.randn(2, 3, 8)*100], dim=1)
    mask = torch.tensor([[True]*5+[False]*3]*2)
    for weight in (0.0, 0.5, 1.0):
        conf.hybrid_mean_weight = weight
        model = AttentionMeanMIL(conf).eval()
        model.load_state_dict(baseline.state_dict())
        assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in baseline.parameters())
        with torch.no_grad():
            logits, attention = model(padded, mask=mask, return_attention=True)
            torch.testing.assert_close(logits, model(x))
            torch.testing.assert_close(model(x[:, [3, 0, 4, 2, 1]]), model(x))
            assert (attention[:, :, 5:] == 0).all()
            if weight == 0:
                torch.testing.assert_close(logits, baseline(padded, mask=mask), rtol=0, atol=0)
            if weight == 1:
                torch.testing.assert_close(model(x), model.classifier(model.dimreduction(x).mean(1)))
            torch.testing.assert_close(model(x[:, :1]), baseline(x[:, :1]))
        loss = torch.nn.functional.cross_entropy(model(padded, mask=mask), torch.tensor([0, 2]))
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print('PASS: hybrid padding/order invariance, same parameter count, finite gradients,')
    print('zero-weight exact baseline equivalence, masked mean endpoint and single-patch bags.')


if __name__ == '__main__':
    main()
