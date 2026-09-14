#!/usr/bin/env python
"""Mechanism audit of a WHU-trained CMCE checkpoint.

Four blocks of evidence from one checkpoint:

  A. shapes         Q/K/V and attention shapes at every scale, read from the live tensors,
                    with the post-softmax attention values and the spatial spread of the
                    attention output.
  B. gradients      gradient norms of every functional group (anchor heads, scale-anchor
                    affinity, offset networks, Q/K/V projections, confidence gate, BCLR
                    modules) under the segmentation loss alone and under the full objective.
  C. iterations     how many refinement iterations execute in train and eval mode, given the
                    early-stopping threshold epsilon.
  D. interventions  anchors zeroed, shuffled across images, rolled across the anchor index or
                    replaced by Gaussian noise (and, for contrast, the text embedding zeroed),
                    measured as IoU change and pixel disagreement on an evenly spaced subset
                    of the Inria test split.

Example:
    python experiments/mechanism_probe.py --checkpoint weights/cmce_whu_seed42.pth
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import create_cleaned_model  # noqa: E402
from utils.transforms import BuildingExtractionTransforms  # noqa: E402
from utils.unified_data_manager import UnifiedDataManager, custom_collate_fn  # noqa: E402

PARAM_GROUPS = [
    ('anchor_decomposer.shared_encoder', 'SAD shared encoder'),
    ('anchor_decomposer.anchor_heads',   'SAD anchor heads (a_k)'),
    ('scale_anchor_affinity',            'scale-anchor affinity W_sa'),
    ('anchor_coherence',                 'anchor coherence (aux loss only)'),
    ('visual_refiner.scale_refiners.0.offset_module', 'OffsetNet, scale 0'),
    ('visual_refiner.scale_refiners.1.offset_module', 'OffsetNet, scale 1'),
    ('visual_refiner.scale_refiners.2.offset_module', 'OffsetNet, scale 2'),
    ('visual_refiner.scale_refiners.3.offset_module', 'OffsetNet, scale 3'),
    ('q_proj',                           'W_Q  query projection'),
    ('k_proj',                           'W_K  key projection'),
    ('v_proj',                           'W_V  value projection'),
    ('out_proj',                         'attention output projection'),
    ('confidence_gate',                  'confidence gate'),
    ('text_refiner',                     'text refinement (BCLR)'),
    ('inconsistency_detector',           'inconsistency detector (BCLR)'),
    ('alignment_estimator',              'alignment estimator (BCLR)'),
]


def build_loader(config, n_sub, bs, workers, text_dir):
    dm = UnifiedDataManager(config)
    tf = BuildingExtractionTransforms(phase='test', image_size=config['data']['image_size'],
                                      augmentation_config={})
    full = dm.get_dataloader(dataset_name='inria', split='test', batch_size=bs, shuffle=False,
                             num_workers=workers, alignment_type='cmce', text_type='janus',
                             transform=tf, text_feature_subdir=text_dir,
                             text_subdir='unified_janus_texts').dataset
    if n_sub <= 0 or n_sub >= len(full):
        ds = full
    else:
        idx = np.linspace(0, len(full) - 1, n_sub).astype(int)
        ds = Subset(full, sorted(set(idx.tolist())))
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                      collate_fn=custom_collate_fn), len(ds)


def get_cmce(model):
    return model.cmce_module


def grad_norm(model, prefix):
    tot, n = 0.0, 0
    for name, p in model.named_parameters():
        if prefix in name and p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
            n += p.numel()
    return (tot ** 0.5), n


# ---------------------------------------------------------------- A. shapes
def probe_shapes(model, batch, device):
    cmce = get_cmce(model)
    rec = {'scales': [], 'attn_check': {}}
    handles = []
    cap = {}

    def mk(tag, s):
        def hook(mod, inp, out):
            cap.setdefault(s, {})[tag] = out.detach()
        return hook

    for s, ref in enumerate(cmce.visual_refiner.scale_refiners):
        handles.append(ref.q_proj.register_forward_hook(mk('Q', s)))
        handles.append(ref.k_proj.register_forward_hook(mk('K', s)))
        handles.append(ref.v_proj.register_forward_hook(mk('V', s)))
        if getattr(ref, 'use_deformable', False):
            handles.append(ref.offset_module.register_forward_hook(mk('offset', s)))
    anchor_cap = {}
    handles.append(cmce.anchor_decomposer.register_forward_hook(
        lambda m, i, o: anchor_cap.__setitem__('anchors', o.detach())))

    model.eval()
    with torch.no_grad():
        model(batch['image'].to(device), batch['text_feature'].to(device))
    for h in handles:
        h.remove()

    for s, ref in enumerate(cmce.visual_refiner.scale_refiners):
        c = cap[s]
        B, L, D = c['Q'].shape
        heads, hd = ref.num_heads, ref.head_dim
        Qh = c['Q'].view(B, L, heads, hd).transpose(1, 2)
        Kh = c['K'].view(B, 1, heads, hd).transpose(1, 2)
        Vh = c['V'].view(B, 1, heads, hd).transpose(1, 2)
        logits = torch.matmul(Qh, Kh.transpose(-2, -1)) / ref.scale
        attn = F.softmax(logits, dim=-1)
        out = torch.matmul(attn, Vh)
        # is the attention output identical at every spatial position?
        spread = float((out - out[:, :, :1, :]).abs().max())
        rec['scales'].append({
            'scale': s,
            'visual_dim': int(ref.visual_dim),
            'spatial_HW': int(L),
            'Q_shape_pre_head_split': list(c['Q'].shape),
            'K_shape_pre_head_split': list(c['K'].shape),
            'V_shape_pre_head_split': list(c['V'].shape),
            'Q_shape_multihead': list(Qh.shape),
            'K_shape_multihead': list(Kh.shape),
            'V_shape_multihead': list(Vh.shape),
            'key_value_sequence_length': int(Kh.shape[2]),
            'attn_shape': list(attn.shape),
            'attn_min': float(attn.min()), 'attn_max': float(attn.max()),
            'attn_all_exactly_one': bool(torch.all(attn == 1.0)),
            'attn_output_spatial_spread_maxabs': spread,
            'offset_shape': list(c['offset'].shape) if 'offset' in c else None,
            'offset_abs_mean_px': float(c['offset'].abs().mean()) if 'offset' in c else None,
            'offset_abs_max_px': float(c['offset'].abs().max()) if 'offset' in c else None,
        })
    rec['anchors_shape'] = list(anchor_cap['anchors'].shape)
    rec['affinity_matrix'] = cmce.scale_anchor_affinity().detach().cpu().tolist()
    return rec


# ------------------------------------------------------------- B. gradients
def probe_gradients(model, batch, device, objective):
    """objective: 'seg_only' = segmentation loss alone;
                  'full'     = segmentation + CMCE auxiliary loss, i.e. what training optimises."""
    cmce = get_cmce(model)
    stash = {}
    handles = []

    def keep(key):
        def hook(mod, inp, out):
            out.retain_grad()
            stash[key] = out
            return out
        return hook

    handles.append(cmce.anchor_decomposer.register_forward_hook(keep('anchors')))
    for s, ref in enumerate(cmce.visual_refiner.scale_refiners):
        if getattr(ref, 'use_deformable', False):
            handles.append(ref.offset_module.register_forward_hook(keep(f'offset_s{s}')))

    model.train()                      # training-mode graph, but deterministic input
    for m in model.modules():          # keep dropout off so numbers are reproducible
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.eval()
    model.zero_grad(set_to_none=True)

    img = batch['image'].to(device)
    txt = batch['text_feature'].to(device)
    msk = batch['label'].to(device).float()
    if objective == 'full':
        logits, inter = model(img, txt, return_intermediate=True)
        loss_d = model.compute_loss(logits, msk, inter)
    else:
        logits = model(img, txt)
        loss_d = model.compute_loss(logits, msk, None)
    loss = loss_d['total_loss'] if isinstance(loss_d, dict) else loss_d
    loss.backward()
    for h in handles:
        h.remove()

    out = {'objective': objective,
           'loss': float(loss.detach()),
           'loss_components': {k: (float(v.detach()) if isinstance(v, torch.Tensor) else float(v))
                               for k, v in loss_d.items()
                               if (isinstance(v, torch.Tensor) and v.numel() == 1)
                               or isinstance(v, float)}
           if isinstance(loss_d, dict) else {},
           'param_grad_norms': {}, 'tensor_grad_norms': {}}
    for prefix, label in PARAM_GROUPS:
        gn, n = grad_norm(model, prefix)
        out['param_grad_norms'][label] = {'prefix': prefix, 'grad_l2': gn, 'n_params': n,
                                          'is_exactly_zero': gn == 0.0}
    for k, t in stash.items():
        g = t.grad
        out['tensor_grad_norms'][k] = {
            'shape': list(t.shape),
            'grad_l2': float(g.detach().norm()) if g is not None else None,
            'grad_absmax': float(g.detach().abs().max()) if g is not None else None,
        }
    model.zero_grad(set_to_none=True)
    return out


# ------------------------------------------------------- C/D. interventions
def anchor_hook(mode, gen):
    def hook(mod, inp, out):
        if mode == 'none':
            return out
        if mode == 'zero':
            return torch.zeros_like(out)
        if mode == 'shuffle_batch':
            B = out.shape[0]
            if B < 2:
                return out
            p = torch.randperm(B, generator=gen, device='cpu').to(out.device)
            p = torch.where(p == torch.arange(B, device=out.device), (p + 1) % B, p)
            return out[p]
        if mode == 'shuffle_anchor':
            K = out.shape[1]
            if K < 2:
                return out
            p = torch.roll(torch.arange(K, device=out.device), 1)
            return out[:, p]
        if mode == 'randn':
            return torch.randn_like(out) * out.std() + out.mean()
        raise ValueError(mode)
    return hook


@torch.no_grad()
def run_condition(model, loader, device, thr, mode, ref_masks=None, seed=0, text_mode='none'):
    cmce = get_cmce(model)
    gen = torch.Generator().manual_seed(seed)
    h = cmce.anchor_decomposer.register_forward_hook(anchor_hook(mode, gen))
    model.eval()
    tp = fp = fn = 0.0
    diff = tot = 0.0
    masks = []
    for bi, batch in enumerate(loader):
        img = batch['image'].to(device)
        txt = batch['text_feature'].to(device)
        if text_mode == 'zero':
            txt = torch.zeros_like(txt)
        lbl = batch['label'].to(device).bool()
        pred = torch.sigmoid(model(img, txt)) > thr
        tp += float((pred & lbl).sum())
        fp += float((pred & ~lbl).sum())
        fn += float((~pred & lbl).sum())
        if ref_masks is None:
            masks.append(pred.cpu())
        else:
            diff += float((pred.cpu() != ref_masks[bi]).sum())
            tot += pred.numel()
    h.remove()
    res = {'mode': mode, 'text_mode': text_mode,
           'iou': tp / max(tp + fp + fn, 1),
           'precision': tp / max(tp + fp, 1),
           'recall': tp / max(tp + fn, 1)}
    if ref_masks is not None:
        res['pixel_disagreement_vs_real'] = diff / max(tot, 1)
    return res, (masks if ref_masks is None else None)


@torch.no_grad()
def count_iterations(model, batch, device, force_train_mode=False):
    """How many refinement iterations actually execute, given the epsilon early-stopping rule."""
    cmce = get_cmce(model)
    seen = {'n': 0}
    h = cmce.alignment_estimator.register_forward_hook(
        lambda m, i, o: seen.__setitem__('n', seen['n'] + 1))
    model.train(force_train_mode)
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.eval()
    model(batch['image'].to(device), batch['text_feature'].to(device))
    h.remove()
    model.eval()
    return seen['n']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default='weights/cmce_whu_seed42.pth')
    p.add_argument('--config', default='configs/cmce_v5_canonical_inria.yaml')
    p.add_argument('--data-root', default=None, help='directory holding inria/test/...; defaults to data/')
    p.add_argument('--text-dir', default='unified_janus_features')
    p.add_argument('--n-sub', type=int, default=400, help='evenly spaced Inria test images; <= 0 uses all')
    p.add_argument('--bs', type=int, default=4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default='results/mechanism_probe.json')
    a = p.parse_args()

    ckpt_path = Path(a.checkpoint).resolve()
    out_path = Path(a.out).resolve()
    config_path = Path(a.config).resolve()
    data_root = Path(a.data_root).resolve() if a.data_root else None
    os.chdir(ROOT)

    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['data']['dataset'] = 'inria'
    if data_root is not None:
        config['data']['root_dir'] = str(data_root)
        config.setdefault('datasets', {}).setdefault('inria', {})['root_dir'] = str(data_root / 'inria')
    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    loader, n_sub = build_loader(config, a.n_sub, a.bs, a.workers, a.text_dir)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    thr = float(sd.get('best_threshold', 0.5))
    model = create_cleaned_model(config, 'cmce', 'janus').to(device)
    model.load_state_dict(sd['model_state_dict'], strict=True)
    model.eval()

    batch = next(iter(loader))
    out = {'analysis': 'CMCE mechanism audit',
           'checkpoint': ckpt_path.name, 'source_threshold': thr, 'n_subset_images': n_sub,
           'config': config_path.name, 'text_dir': a.text_dir,
           'cmce_config': {'num_anchors': get_cmce(model).num_anchors,
                           'max_iterations': get_cmce(model).max_iterations,
                           'convergence_threshold': get_cmce(model).convergence_threshold,
                           'use_deformable': get_cmce(model).use_deformable}}

    print('--- A. shapes ---', flush=True)
    out['A_shapes'] = probe_shapes(model, batch, device)
    for s in out['A_shapes']['scales']:
        print('  scale %d  Q %s  K %s  V %s  attn %s  kv_len=%d  attn==1: %s  out spread %.3e'
              % (s['scale'], s['Q_shape_multihead'], s['K_shape_multihead'],
                 s['V_shape_multihead'], s['attn_shape'], s['key_value_sequence_length'],
                 s['attn_all_exactly_one'], s['attn_output_spatial_spread_maxabs']), flush=True)

    print('--- B. gradients ---', flush=True)
    out['B_gradients'] = {obj: probe_gradients(model, batch, device, obj)
                          for obj in ('seg_only', 'full')}
    print('  %-34s %-14s %-14s' % ('parameter group', 'seg-loss only', 'seg+auxiliary'), flush=True)
    for label in out['B_gradients']['seg_only']['param_grad_norms']:
        a_ = out['B_gradients']['seg_only']['param_grad_norms'][label]
        b_ = out['B_gradients']['full']['param_grad_norms'][label]
        print('  %-34s %-14.6e %-14.6e %s' % (label, a_['grad_l2'], b_['grad_l2'],
                                              'ZERO in both' if a_['is_exactly_zero']
                                              and b_['is_exactly_zero'] else ''), flush=True)
    for k in out['B_gradients']['full']['tensor_grad_norms']:
        a_ = out['B_gradients']['seg_only']['tensor_grad_norms'][k]
        b_ = out['B_gradients']['full']['tensor_grad_norms'][k]
        print('  tensor %-14s %-18s %-14.6e %-14.6e' % (k, a_['shape'], a_['grad_l2'],
                                                        b_['grad_l2']), flush=True)

    print('--- C. iteration count under epsilon early stopping ---', flush=True)
    out['C_iterations'] = {
        'max_iterations_config': get_cmce(model).max_iterations,
        'epsilon_convergence_threshold': get_cmce(model).convergence_threshold,
        'executed_iterations_eval_mode': count_iterations(model, batch, device, False),
        'executed_iterations_train_mode': count_iterations(model, batch, device, True),
        'note': 'early stopping is gated on `not self.training`, so it is active at eval only',
    }
    print('  ', out['C_iterations'], flush=True)

    print('--- D. anchor interventions on %d Inria images ---' % n_sub, flush=True)
    base, ref_masks = run_condition(model, loader, device, thr, 'none')
    out['D_interventions'] = {'real': base}
    print('  %-16s IoU %.4f' % ('real', base['iou']), flush=True)
    for mode, tmode, key in [('zero', 'none', 'anchors_zero'),
                             ('shuffle_batch', 'none', 'anchors_shuffled_across_images'),
                             ('shuffle_anchor', 'none', 'anchors_rolled_across_k'),
                             ('randn', 'none', 'anchors_gaussian_noise'),
                             ('none', 'zero', 'text_embedding_zeroed')]:
        r, _ = run_condition(model, loader, device, thr, mode, ref_masks=ref_masks,
                             seed=20260826, text_mode=tmode)
        r['delta_iou_vs_real'] = r['iou'] - base['iou']
        out['D_interventions'][key] = r
        print('  %-30s IoU %.4f  dIoU %+.4f  pixel disagreement %.4f%%'
              % (key, r['iou'], r['delta_iou_vs_real'],
                 100 * r['pixel_disagreement_vs_real']), flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('saved ' + str(out_path), flush=True)


if __name__ == '__main__':
    main()
