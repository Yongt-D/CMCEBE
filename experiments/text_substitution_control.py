#!/usr/bin/env python
"""Text substitution controls under WHU -> Inria transfer.

For each WHU-trained CMCE checkpoint, evaluate the full Inria test split under four text
conditions in one pass over the data:

  real         each test image's own description embedding
  mean_target  the mean embedding over the Inria test descriptions, given to every image
  mean_source  the mean embedding over the WHU training descriptions, given to every image
  shuffled     a fixed derangement: each image receives a different image's description

Pixel-pooled tp/fp/fn are accumulated on a threshold grid and read under four protocols:
  source   threshold selected on WHU validation (checkpoint best_threshold); no target labels
  window   best IoU within [source - 0.15, source + 0.15] (uses target labels)
  oracle   best IoU over the full 0.05-0.95 grid (uses target labels)
  fixed05  fixed threshold 0.5

--text-dir unified_janus_features reproduces the paper's canonical text-control table;
--text-dir text_features reproduces its legacy-directory table (the Prompt-A features that the
released CMCE checkpoints were trained on).

Example:
    python experiments/text_substitution_control.py --out results/text_controls_unified.json
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
CONDITIONS = ['real', 'mean_target', 'mean_source', 'shuffled']


def make_loader(config, dataset, split, feat_subdir, text_subdir, bs, workers):
    dm = UnifiedDataManager(config)
    tf = BuildingExtractionTransforms(phase='test', image_size=config['data']['image_size'],
                                      augmentation_config={})
    loader = dm.get_dataloader(dataset_name=dataset, split=split, batch_size=bs, shuffle=False,
                               num_workers=workers, alignment_type='cmce', text_type='janus',
                               transform=tf, text_feature_subdir=feat_subdir,
                               text_subdir=text_subdir)
    missing = [s['name'] for s in loader.dataset.samples if not s.get('text_path')]
    if not loader.dataset.load_text or missing:
        raise SystemExit(f'{dataset}/{split}: {len(missing)} images lack a feature in {feat_subdir}')
    return loader


def collect_text(loader, desc):
    """One data-only pass: stack every post-transform text embedding, in loader order."""
    feats = []
    for i, batch in enumerate(loader):
        feats.append(batch['text_feature'].float())
        if (i + 1) % 200 == 0:
            print(f'  [{desc}] {len(feats)} batches', flush=True)
    out = torch.cat(feats, 0)
    print(f'  [{desc}] collected {tuple(out.shape)}', flush=True)
    return out


def derangement(n, seed):
    """Random permutation with no fixed point (every image gets a DIFFERENT image's text)."""
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p
        fixed = np.where(p == np.arange(n))[0]
        for i in fixed:                       # repair rather than resample
            j = int(rng.integers(0, n))
            if j != i and p[j] != i:
                p[i], p[j] = p[j], p[i]
        if not np.any(p == np.arange(n)):
            return p


@torch.no_grad()
def eval_conditions(model, loader, device, text_all, mean_t, mean_s, perm):
    g = torch.tensor(GRID, device=device).view(-1, 1)
    acc = {c: {k: torch.zeros(len(GRID), dtype=torch.float64, device=device)
               for k in ('tp', 'fp', 'fn')} for c in CONDITIONS}
    text_all = text_all.to(device)
    mean_t = mean_t.to(device)
    mean_s = mean_s.to(device)
    pos = 0
    n_img = 0
    for bi, batch in enumerate(loader):
        img = batch['image'].to(device, non_blocking=True)
        lbl = batch['label'].to(device, non_blocking=True).bool().view(1, -1)
        b = img.shape[0]
        idx = torch.arange(pos, pos + b, device=device)

        feed = {
            'real':        text_all[idx],
            'mean_target': mean_t.unsqueeze(0).expand(b, -1).contiguous(),
            'mean_source': mean_s.unsqueeze(0).expand(b, -1).contiguous(),
            'shuffled':    text_all[torch.as_tensor(perm[pos:pos + b], device=device)],
        }
        for c in CONDITIONS:
            prob = torch.sigmoid(model(img, feed[c])).view(1, -1)
            pred = prob > g
            acc[c]['tp'] += (pred & lbl).sum(1).double()
            acc[c]['fp'] += (pred & ~lbl).sum(1).double()
            acc[c]['fn'] += (~pred & lbl).sum(1).double()
        pos += b
        n_img += b
        if (bi + 1) % 100 == 0:
            print(f'    {n_img} images', flush=True)

    out = {}
    for c in CONDITIONS:
        tp, fp, fn = acc[c]['tp'], acc[c]['fp'], acc[c]['fn']
        out[c] = (
            (tp / (tp + fp + fn).clamp(min=1)).cpu().numpy(),
            (tp / (tp + fp).clamp(min=1)).cpu().numpy(),
            (tp / (tp + fn).clamp(min=1)).cpu().numpy(),
        )
    return out, n_img


def pick(iou, pre, rec, mask):
    i = int(np.argmax(np.where(mask, iou, -1)))
    return {'threshold': float(GRID[i]), 'iou': float(iou[i]),
            'precision': float(pre[i]), 'recall': float(rec[i])}


def at(iou, pre, rec, t):
    i = int(np.argmin(np.abs(GRID - t)))
    return {'threshold': float(GRID[i]), 'iou': float(iou[i]),
            'precision': float(pre[i]), 'recall': float(rec[i])}


def with_data_root(config, dataset, data_root):
    config['data']['dataset'] = dataset
    if data_root is not None:
        config['data']['root_dir'] = str(data_root)
        config.setdefault('datasets', {}).setdefault(dataset, {})['root_dir'] = str(data_root / dataset)
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights-dir', default='weights')
    p.add_argument('--seeds', default='0,42,123,456,666')
    p.add_argument('--config', default='configs/cmce_v5_canonical_inria.yaml')
    p.add_argument('--source-config', default='configs/cmce_v5_canonical.yaml')
    p.add_argument('--data-root', default=None, help='directory holding inria/ and whu_building/; defaults to data/')
    p.add_argument('--text-dir', default='unified_janus_features',
                   help='unified_janus_features (canonical) or text_features (Prompt-A, training-matched)')
    p.add_argument('--text-sub', default='unified_janus_texts')
    p.add_argument('--perm-seed', type=int, default=20260826)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--bs', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', default='results/text_substitution_control.json')
    a = p.parse_args()

    weights_dir = Path(a.weights_dir).resolve()
    out_path = Path(a.out).resolve()
    config_path = Path(a.config).resolve()
    source_config_path = Path(a.source_config).resolve()
    data_root = Path(a.data_root).resolve() if a.data_root else None
    os.chdir(ROOT)

    with open(config_path, encoding='utf-8') as f:
        config = with_data_root(yaml.safe_load(f), 'inria', data_root)
    with open(source_config_path, encoding='utf-8') as f:
        src_config = with_data_root(yaml.safe_load(f), 'whu_building', data_root)

    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    tgt_loader = make_loader(config, 'inria', 'test', a.text_dir, a.text_sub, a.bs, a.workers)
    print('collecting Inria test text embeddings ...', flush=True)
    text_all = collect_text(tgt_loader, 'inria/test')
    mean_t = text_all.mean(0)

    src_loader = make_loader(src_config, 'whu_building', 'train', a.text_dir, a.text_sub, 16, a.workers)
    print('collecting WHU train text embeddings ...', flush=True)
    src_text = collect_text(src_loader, 'whu/train')
    mean_s = src_text.mean(0)
    del src_text

    n = text_all.shape[0]
    perm = derangement(n, a.perm_seed)
    assert not np.any(perm == np.arange(n)), 'derangement has a fixed point'

    out = {
        'analysis': 'text substitution controls, WHU -> Inria',
        'dataset': 'inria', 'split': 'test', 'n_expected': n,
        'text_dir': a.text_dir, 'perm_seed': a.perm_seed,
        'grid': [float(x) for x in GRID],
        'conditions': {
            'real': "each test image's own description embedding (control)",
            'mean_target': 'mean over the Inria test description embeddings',
            'mean_source': 'mean over the WHU train description embeddings',
            'shuffled': "fixed derangement: each image receives a different image's description",
        },
        'protocol_note': 'global tp/fp/fn accumulated over the full target test set',
        'text_stats': {
            'mean_target_norm': float(mean_t.norm()),
            'mean_source_norm': float(mean_s.norm()),
            'per_image_norm_mean': float(text_all.norm(dim=1).mean()),
            'per_image_norm_std': float(text_all.norm(dim=1).std()),
            'cos_mean_target_vs_mean_source': float(
                torch.nn.functional.cosine_similarity(mean_t, mean_s, dim=0)),
            'mean_cos_to_target_mean': float(torch.nn.functional.cosine_similarity(
                text_all, mean_t.unsqueeze(0).expand_as(text_all), dim=1).mean()),
        },
        'runs': [],
    }

    for s in a.seeds.split(','):
        ck = weights_dir / f'cmce_whu_seed{s}.pth'
        if not ck.exists():
            print(f'MISSING {ck}', flush=True)
            continue
        sd = torch.load(ck, map_location='cpu', weights_only=False)
        vthr = float(sd['best_threshold'])
        state = sd['model_state_dict'] if 'model_state_dict' in sd else sd

        model = create_cleaned_model(config, 'cmce', 'janus').to(device)
        model.load_state_dict(state, strict=True)
        model.eval()

        print(f'=== seed {s}  source threshold={vthr:.2f} ===', flush=True)
        res, n_img = eval_conditions(model, tgt_loader, device, text_all, mean_t, mean_s, perm)

        lo, hi = max(0.05, vthr - 0.15), min(0.95, vthr + 0.15)
        win = (GRID >= lo - 1e-9) & (GRID <= hi + 1e-9)
        allm = np.ones_like(win, dtype=bool)

        row = {'seed': s, 'checkpoint': ck.name, 'source_threshold': vthr, 'n_images': n_img,
               'conditions': {}}
        for c in CONDITIONS:
            iou, pre, rec = res[c]
            row['conditions'][c] = {
                'source': at(iou, pre, rec, vthr),
                'window': pick(iou, pre, rec, win),
                'oracle': pick(iou, pre, rec, allm),
                'fixed05': at(iou, pre, rec, 0.5),
            }
        out['runs'].append(row)

        print('  %-12s %-8s %-8s %-8s' % ('condition', 'source', 'window', 'oracle'), flush=True)
        for c in CONDITIONS:
            d = row['conditions'][c]
            print('  %-12s %-8.2f %-8.2f %-8.2f' % (c, d['source']['iou'] * 100,
                                                    d['window']['iou'] * 100,
                                                    d['oracle']['iou'] * 100), flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')

    print('saved ' + str(out_path), flush=True)


if __name__ == '__main__':
    main()
