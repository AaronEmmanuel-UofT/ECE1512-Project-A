"""Mathematical, invariance and full synthetic training checks, not benchmarks."""
import unittest,tempfile,json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from model import ABMIL
from model_fragility import FragilityGuidedMIL,suppress_top_fraction
from research.train_extension import run
from verify_baseline import make_fixture
from research.analyze_validation import analyze,digest
from research.evaluate_extension import evaluate_run

class FragilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        self.conf=SimpleNamespace(D_feat=8,D_inner=16,D_attn=8,n_class=3,droprate=0.,probe_fraction=.1)

    def test_no_probe_exact_baseline_and_parameter_parity(self):
        self.conf.probe_fraction=0
        base=ABMIL(self.conf);model=FragilityGuidedMIL(self.conf);model.load_state_dict(base.state_dict())
        x=torch.randn(2,6,8)
        actual,diag=model(x,return_diagnostics=True)
        torch.testing.assert_close(actual,base(x),rtol=0,atol=0)
        self.assertTrue((diag['alpha']==0).all())
        self.assertEqual(sum(p.numel() for p in base.parameters()),sum(p.numel() for p in model.parameters()))

    def test_padding_order_and_single_patch(self):
        m=FragilityGuidedMIL(self.conf).eval();x=torch.randn(2,6,8)
        padded=torch.cat([x,torch.randn(2,3,8)*100],1)
        mask=torch.tensor([[True]*6+[False]*3]*2)
        logits,d=m(padded,mask=mask,return_diagnostics=True)
        torch.testing.assert_close(logits,m(x))
        torch.testing.assert_close(m(x[:,[5,2,1,3,0,4]]),m(x))
        self.assertTrue((d['effective_weights'][:,6:]==0).all())
        self.assertTrue((d['probe_attention'][:,6:]==0).all())
        one,diag=m(x[:,:1],return_diagnostics=True)
        b=ABMIL(self.conf);b.load_state_dict(m.state_dict())
        torch.testing.assert_close(one,b(x[:,:1]),rtol=0,atol=0)
        self.assertTrue((diag['alpha']==0).all())

    def test_cutoff_ties_are_symmetric_and_remove_exact_budget(self):
        s=torch.tensor([[3.,2.,2.,0.,999.]])
        valid=torch.tensor([[True,True,True,True,False]])
        p,r,n=suppress_top_fraction(s,valid,.5)
        torch.testing.assert_close(r,torch.tensor([[0.,.5,.5,1.,0.]]))
        self.assertEqual(n.item(),2)
        perm=[2,4,0,3,1]
        p2,r2,_=suppress_top_fraction(s[:,perm],valid[:,perm],.5)
        torch.testing.assert_close(p2,p[:,perm]);torch.testing.assert_close(r2,r[:,perm])
        self.assertEqual(p[0,-1].item(),0)

    def test_extreme_score_stability_and_uniform_ties(self):
        s=torch.tensor([[1000.,0.,-1000.],[1.,1.,1.]])
        p,r,n=suppress_top_fraction(s,torch.ones_like(s,dtype=torch.bool),.1)
        self.assertTrue(torch.isfinite(p).all())
        torch.testing.assert_close(p.sum(1),torch.ones(2))
        torch.testing.assert_close(p[0],torch.tensor([0.,1.,0.]))
        torch.testing.assert_close(p[1],torch.ones(3)/3)
        self.assertTrue((r[1]==r[1,0]).all())

    def test_gate_formula_effective_weights_and_finite_gradients(self):
        m=FragilityGuidedMIL(self.conf);x=torch.randn(2,7,8)
        logits,d=m(x,return_diagnostics=True)
        expected=.5*(d['attention_probabilities']-d['probe_probabilities']).abs().sum(1)
        torch.testing.assert_close(d['alpha'],expected)
        self.assertFalse(d['alpha'].requires_grad)
        self.assertTrue(((d['alpha']>=0)&(d['alpha']<=1)).all())
        torch.testing.assert_close(d['effective_weights'].sum(1),torch.ones(2))
        z=(d['effective_weights'].unsqueeze(-1)*m.dimreduction(x)).sum(1)
        torch.testing.assert_close(logits,m.classifier(z))
        torch.nn.functional.cross_entropy(logits,torch.tensor([0,2])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_invalid_settings_and_masks(self):
        for value in [-.1,1.,float('nan'),float('inf')]:
            self.conf.probe_fraction=value
            with self.assertRaises(ValueError):FragilityGuidedMIL(self.conf)
        self.conf.probe_fraction=.1;m=FragilityGuidedMIL(self.conf)
        for x,mask in [(torch.empty(2,0,8),None),(torch.randn(2,3,8),torch.zeros(2,3,dtype=torch.bool)),(torch.randn(2,3,8),torch.ones(2,3))]:
            with self.assertRaises(ValueError):m(x,mask=mask)

    def test_synthetic_training_all_datasets_and_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            for ds in ['C16','C17','BRACS']:
                data=tmp/f'{ds}.h5';make_fixture(ds,data)
                methods=['fragility','abmil','attention_mean','mean'] if ds=='C16' else ['fragility']
                saved_splits=None
                for method in methods:
                    out=tmp/f'{ds}_{method}';summary=run(ds,out,method,epochs=2,data=data,threads=1)
                    self.assertFalse(summary['test_evaluated'])
                    self.assertEqual(summary['kind'],'debug only; not benchmark')
                    history=json.loads((out/'history.json').read_text());self.assertEqual(len(history),2)
                    splits=json.loads((out/'splits.json').read_text())
                    if saved_splits is not None:self.assertEqual(saved_splits,splits)
                    saved_splits=splits
                    for name in ['config.json','environment.json','best_state.pt','loss_curves.png','source_snapshot/model_fragility.py']:
                        self.assertTrue((out/name).is_file(),name)
                    self.assertFalse((out/'heldout_test').exists())
                    if ds=='C16' and method=='fragility':
                        sha=digest(out/'best_state.pt')
                        result=analyze(out,out/'validation_diagnostics',threads=1)
                        self.assertFalse(result['test_evaluated'])
                        self.assertEqual(result['validation_slides'],len(splits['val']))
                        self.assertEqual(digest(out/'best_state.pt'),sha)
                        with self.assertRaises(FileExistsError):analyze(out,out/'validation_diagnostics')
                        with self.assertRaises(ValueError):evaluate_run(out,out/'rejected_debug_test')
                        archived=out/'source_snapshot/model_fragility.py';original=archived.read_bytes()
                        archived.write_bytes(original+b'\n# altered snapshot\n')
                        with self.assertRaises(ValueError):evaluate_run(out,out/'rejected_archive',expected_epochs=2)
                        archived.write_bytes(original)
                        # Synthetic fixtures ONLY: test the evaluation machinery at two epochs.
                        result=evaluate_run(out,out/'synthetic_heldout',expected_epochs=2,threads=1)
                        self.assertFalse(result['benchmark'])
                        self.assertTrue((out/'synthetic_heldout/gate_diagnostics.csv').is_file())
                        self.assertEqual(digest(out/'best_state.pt'),sha)
                        with self.assertRaises(ValueError):evaluate_run(out,out/'duplicate',expected_epochs=2)
                    with self.assertRaises(FileExistsError):run(ds,out,method,epochs=2,data=data,threads=1)

if __name__=='__main__':unittest.main(verbosity=2)
