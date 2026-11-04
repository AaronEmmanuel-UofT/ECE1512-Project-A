"""Synthetic checks only. These are not pathology benchmark results."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import ABMIL
from engine import train_one_epoch, evaluate
from dataloader import mil_collate_fn, build_HDF5_feat_dataset
from utils import seed_everything

def main():
    torch.set_num_threads(1)
    seed_everything(42)
    conf = SimpleNamespace(D_feat=8, D_inner=16, D_attn=8, n_class=2,
                           droprate=0.0, dataset='CAMELYON16', seed=42,
                           val_fraction=0.25, n_shot=0)
    model = ABMIL(conf).eval()
    x = torch.randn(1, 5, 8)
    padded = torch.cat([x, torch.randn(1, 4, 8)*100], dim=1)
    mask = torch.tensor([[True]*5+[False]*4])
    with torch.no_grad():
        logits, attention = model(padded, mask=mask, return_attention=True)
        torch.testing.assert_close(logits, model(x))
        torch.testing.assert_close(attention.sum(-1), torch.ones(1,1))
        assert (attention[:,:,5:] == 0).all()
        perm = torch.randperm(5)
        torch.testing.assert_close(model(x[:,perm]), model(x))
    try:
        model(padded, mask=torch.zeros_like(mask))
    except ValueError:
        pass
    else:
        raise AssertionError('Empty mask was not rejected')
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp)/'synthetic.h5'
        rng = np.random.default_rng(42)
        with h5py.File(file, 'w') as f:
            for i in range(48):
                label = i % 2
                group = f.create_group(('train_' if i < 32 else 'test_') + str(i))
                group.create_dataset('feat', data=rng.normal((2*label-1)*2, .2, (5+i%7, 8)).astype('float32'))
                group.attrs['label'] = label
        tr, va, te = build_HDF5_feat_dataset(str(file), conf)
        assert not (set(tr.data_names) & set(va.data_names))
        assert not ((set(tr.data_names) | set(va.data_names)) & set(te.data_names))
        loader = DataLoader(tr,batch_size=4,shuffle=True,collate_fn=mil_collate_fn)
        vl = DataLoader(va,batch_size=4,collate_fn=mil_collate_fn)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(),lr=.01)
        before = evaluate(model,criterion,vl,'cpu',conf)['loss']
        for _ in range(8):
            train_one_epoch(model,criterion,loader,optimizer,'cpu')
        after = evaluate(model,criterion,vl,'cpu',conf)
        assert after['loss'] < before, (before, after)
        checkpoint = Path(tmp)/'state.pt'
        torch.save(model.state_dict(),checkpoint)
        restored = ABMIL(conf).eval()
        restored.load_state_dict(torch.load(checkpoint,weights_only=True))
        model.eval()
        torch.testing.assert_close(restored(x),model(x))
        print('PASS: masks, attention normalization, permutation invariance, empty-bag rejection,')
        print('HDF5 loading, disjoint splits, training loss decrease, metrics and checkpoint reload.')
        print('Synthetic validation loss:',before,'->',after['loss'])

if __name__ == '__main__':
    main()
