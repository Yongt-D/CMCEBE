#!/usr/bin/env python
"""Anchor-collapse interventions under WHU -> Inria transfer.

Each K=3 CMCE checkpoint is evaluated on the full Inria test split with its anchor set replaced
at inference time. If the three anchors carried distinct, load-bearing information, forcing them
to be identical would cost accuracy.

    K3_real        the trained model, untouched
    collapse_mean  all K anchors replaced by their mean (behaves as a single anchor)
    only_a1/2/3    all anchors replaced by one of them
    zero           all anchors zeroed

Protocols (read from one threshold sweep): source (WHU validation threshold, no target labels),
window (best within +-0.15 of it, uses target labels), oracle (full grid), fixed05 (0.5).

Example:
    python experiments/anchor_collapse.py --out results/anchor_collapse.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import create_cleaned_model  # noqa: E402
from utils.transforms import BuildingExtractionTransforms  # noqa: E402
from utils.unified_data_manager import UnifiedDataManager  # noqa: E402

GRID = np.round(np.arange(0.05, 0.9501, 0.01), 3)
COLLAPSE = ['K3_real', 'collapse_mean', 'only_a1', 'only_a2', 'only_a3', 'zero']


def make_loader(config, bs, workers, text_dir):
    dm = UnifiedDataManager(config)
    tf = BuildingExtractionTransforms(phase='test', image_size=config['data']['image_size'],
                                      augmentation_config={})
    loader = dm.get_dataloader(dataset_name='inria', split='test', batch_size=bs, shuffle=False,
                               num_workers=workers, alignment_type='cmce', text_type='janus',
                               transform=tf, text_feature_subdir=text_dir,
                               text_subdir='unified_janus_texts')
    missing = [s['name'] for s in loader.dataset.samples if not s.get('text_path')]
    if not loader.dataset.load_text or missing:
        raise SystemExit(f'inria/test: {len(missing)} images lack a feature in {text_dir}')
    return loader


def collapse_hook(mode):
    def hook(mod, inp, out):          # out: (B, K, d_h)
        if mode == 'K3_real':
            return out
        if mode == 'collapse_mean':
            return out.mean(dim=1, keepdim=True).expand_as(out).contiguous()
        if mode.startswith('only_a'):
            k = int(mode[-1]) - 1
            if k >= out.shape[1]:
                return out
            return out[:, k:k + 1].expand_as(out).contiguous()
        if mode == 'zero':
            return torch.zeros_like(out)
        raise ValueError(mode)
    return hook


@torch.no_grad()
def sweep(model, loader, device, hook_mode=None):
    h = None
    if hook_mode is not None:
        h = model.cmce_module.anchor_decomposer.register_forward_hook(collapse_hook(hook_mode))
    g = torch.tensor(GRID, device=device).view(-1, 1)
    tp = torch.zeros(len(GRID), dtype=torch.float64, device=device)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)
    n = 0
    for batch in loader:
        img = batch['image'].to(device, non_blocking=True)
        lbl = batch['label'].to(device, non_blocking=True).bool().view(1, -1)
        prob = torch.sigmoid(model(img, batch['text_feature'].to(device))).view(1, -1)
        pred = prob > g
        tp += (pred & lbl).sum(1).double()
        fp += (pred & ~lbl).sum(1).double()
        fn += (~pred & lbl).sum(1).double()
        n += img.shape[0]
    if h is not None:
        h.remove()
    return ((tp / (tp + fp + fn).clamp(min=1)).cpu().numpy(),
            (tp / (tp + fp).clamp(min=1)).cpu().numpy(),
            (tp / (tp + fn).clamp(min=1)).cpu().numpy(), n)


def pick(iou, pre, rec, mask):
    i = int(np.argmax(np.where(mask, iou, -1)))
    return {'threshold': float(GRID[i]), 'iou': float(iou[i]),
            'precision': float(pre[i]), 'recall': float(rec[i])}


def at(iou, pre, rec, t):
    i = int(np.argmin(np.abs(GRID - t)))
    return {'threshold': float(GRID[i]), 'iou': float(iou[i]),
            'precision': float(pre[i]), 'recall': float(rec[i])}


def protocols(iou, pre, rec, vthr):
    lo, hi = max(0.05, vthr - 0.15), min(0.95, vthr + 0.15)
    win = (GRID >= lo - 1e-9) & (GRID <= hi + 1e-9)
    return {'source': at(iou, pre, rec, vthr),
            'window': pick(iou, pre, rec, win),
            'oracle': pick(iou, pre, rec, np.ones_like(win, dtype=bool)),
            'fixed05': at(iou, pre, rec, 0.5)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights-dir', default='weights')
    p.add_argument('--seeds', default='0,42,123,456,666')
    p.add_argument('--config', default='configs/cmce_v5_canonical_inria.yaml')
    p.add_argument('--data-root', default=None, help='directory holding inria/test/...; defaults to data/')
    p.add_argument('--text-dir', default='unified_janus_features')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--bs', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', default='results/anchor_collapse.json')
    a = p.parse_args()

    weights_dir = Path(a.weights_dir).resolve()
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
    loader = make_loader(config, a.bs, a.workers, a.text_dir)

    out = {'analysis': 'inference-time anchor collapse on K=3 CMCE checkpoints, WHU -> Inria',
           'dataset': 'inria', 'split': 'test', 'text_dir': a.text_dir,
           'grid': [float(x) for x in GRID], 'runs': []}

    for s in a.seeds.split(','):
        ck = weights_dir / f'cmce_whu_seed{s}.pth'
        if not ck.exists():
            print(f'MISSING {ck}', flush=True)
            continue
        sd = torch.load(ck, map_location='cpu', weights_only=False)
        vthr = float(sd['best_threshold'])
        model = create_cleaned_model(config, 'cmce', 'janus').to(device)
        model.load_state_dict(sd['model_state_dict'], strict=True)
        model.eval()

        row = {'seed': s, 'checkpoint': ck.name, 'source_threshold': vthr, 'conditions': {}}
        for mode in COLLAPSE:
            iou, pre, rec, n = sweep(model, loader, device, hook_mode=mode)
            row['conditions'][mode] = protocols(iou, pre, rec, vthr)
            row['n_images'] = n
        base = row['conditions']['K3_real']['source']['iou']
        print('  seed %-4s ' % s + '  '.join(
            '%s %.2f(%+.2f)' % (m, row['conditions'][m]['source']['iou'] * 100,
                                (row['conditions'][m]['source']['iou'] - base) * 100)
            for m in COLLAPSE), flush=True)
        out['runs'].append(row)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')

    print('saved ' + str(out_path), flush=True)


if __name__ == '__main__':
    main()
