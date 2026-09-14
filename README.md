# CMCE: text-conditioned cross-domain building extraction

Code, trained weights and text data for

> Yongtao Deng, Dajiang Lei, Liping Zhang, Jiaxin Li, Yidong Peng, and Weisheng Li.
> **Text-Conditioned Cross-Domain Generalization in Remote Sensing Building Extraction: A Mechanistic Evaluation of Semantic Anchor Refinement.**
> *Expert Systems with Applications*, 2026 (accepted; DOI to be added).

CMCE attaches a text-conditioned refinement branch to a U-Net. Each image comes with a
Janus-Pro-1B description embedding, which is decomposed into semantic anchors (SAD) and used by a
bidirectional closed-loop refinement module (BCLR) at four encoder scales. Models are trained on
WHU Building only and applied to Inria and Massachusetts without fine-tuning.

The paper reports two findings:

- **A transfer effect.** With the decision threshold fixed on WHU validation, so that no target
  label is used, the complete branch outperforms a matched visual-only U-Net in all five seeds on
  both targets (mean +4.40 IoU on Inria, +4.88 on Massachusetts).
- **What the trained branch computes.** The text key/value sequence has length one, so the
  attention weights are identically one and the query/key projections receive no gradient.
  Anchor interventions change at most 0.014% of predicted pixels, and the alignment estimator is
  saturated, consistent with a refinement loop that reaches a fixed point after one step. A branch
  with the same 28,761,129 parameters fed one fixed random vector instead of text has a lower mean
  IoU than CMCE (paired p = 0.094 on Inria, 0.415 on Massachusetts).

## Results

IoU (%) on the target test split, mean ± sample standard deviation over seeds 0, 42, 123, 456
and 666, with the threshold selected on WHU validation:

| WHU → | None (visual-only) | Simple fusion | CMCE |
|---|---|---|---|
| Inria (2,997 patches) | 46.67 ± 3.74 | 49.51 ± 1.87 | 51.07 ± 2.56 |
| Massachusetts (40 images) | 24.91 ± 3.09 | 28.61 ± 1.97 | 29.78 ± 1.73 |

The fixed-random control reaches 45.82 ± 3.98 on Inria and 27.65 ± 4.92 on Massachusetts.

All 30 per-seed values behind the table are reproduced exactly by the released weights with
`tools/evaluate_transfer.py`. The outputs are in [`reference_outputs/`](reference_outputs), and
[PROVENANCE.md](PROVENANCE.md) lists the release checks.

> **Training text of the CMCE weights.** The five CMCE checkpoints were trained on the Prompt-A
> description embeddings (`text_features/`) and are evaluated with the unified descriptions; None
> and Simple use the unified set throughout. The accepted manuscript describes the unified set as
> the training input of all models. Evaluated with their training-matched Prompt-A features, the
> CMCE checkpoints give 50.62 ± 2.85 on Inria and 30.49 ± 2.00 on Massachusetts, above None in
> every seed on both targets. See [docs/TEXT_PIPELINE.md](docs/TEXT_PIPELINE.md).

## Contents

    alignment/          CMCE (cmce.py) and the Simple, Dynamic and Generative fusion baselines
    models/             U-Net backbone and the segmentation wrapper with its trainer
    utils/              data loading, transforms, losses
    configs/            canonical configuration (WHU) and its Inria and Massachusetts variants
    train.py            trains None, Simple, Dynamic, Generative (and CMCE) on a chosen text set
    ablation_cmce.py    the CMCE training script that produced the released CMCE weights
    test.py             in-domain evaluation behind the paper's in-domain context tables
    tools/              transfer evaluation, preflight and text-free checks, code-version check,
                        download verification
    experiments/        mechanism probe, text substitution, anchor collapse, convergence trace,
                        fixed-random control
    data/               expected data layout and exact split lists
    docs/               how the descriptions and embeddings were produced
    checksums/          SHA-256 checksums of the downloadable weights and text data
    reference_outputs/  outputs of tools/evaluate_transfer.py on the released weights

`alignment/daca.py` and the Qwen text option are left over from earlier experiments and are not
used in the paper.

## Installation

    pip install -r requirements.txt

Install the PyTorch build that matches your CUDA version first if needed. The code has been run
with Python 3.10 and 3.11 and PyTorch 2.4.0 to 2.8.0.

## Downloads

The trained weights and the text data are hosted on Google Drive.

| Item | Files | Size | Link |
|---|---|---|---|
| Trained weights | 15 files, `{cmce,none,simple}_whu_seed{0,42,123,456,666}.pth` | 1.15 GB | *link to be added* |
| Text data: unified descriptions and embeddings | `cmce_text_unified.zip` | 227 MB | *link to be added* |
| Text data: Prompt-A embeddings | `cmce_text_promptA_features.zip` | 96 MB | *link to be added* |

Check the downloads against the checksums published in this repository:

    python tools/verify_downloads.py --sums checksums/weights_SHA256SUMS.txt --dir weights
    python tools/verify_downloads.py --sums checksums/text_data_SHA256SUMS.txt --dir <folder with the two zip files>

## Data and text features

Download WHU Building, Inria and Massachusetts from their providers, arrange them as described in
[data/README.md](data/README.md), and unzip the two text-data archives into `data/`.

## Pretrained weights

Download the weight files (see [Downloads](#downloads)) into `weights/`. Each file holds the model
parameters (no optimizer state), the alignment type, the seed, the WHU validation threshold and IoU,
the best epoch, the training configuration and the SHA-256 of the original checkpoint.

| Files | Model | Trained with | Training text | Size each |
|---|---|---|---|---|
| `cmce_whu_seed{0,42,123,456,666}.pth` | CMCE | `ablation_cmce.py --ablation full` | Prompt-A `text_features` | 110 MB |
| `none_whu_seed{0,42,123,456,666}.pth` | U-Net without text | `train.py --alignment none --unified-prompt` | none | 51 MB |
| `simple_whu_seed{0,42,123,456,666}.pth` | Simple fusion | `train.py --alignment simple --unified-prompt` | unified | 68 MB |

## Evaluate transfer

    python tools/evaluate_transfer.py --dataset inria \
        --checkpoints weights/cmce_whu_seed0.pth weights/cmce_whu_seed42.pth \
                      weights/cmce_whu_seed123.pth weights/cmce_whu_seed456.pth \
                      weights/cmce_whu_seed666.pth \
        --out results/transfer_inria_cmce.json

Use `--dataset massachusetts` for the second target and the `none_*` or `simple_*` files for the
baselines. One pass reports four threshold protocols: `source` (the WHU validation threshold, the
paper's primary protocol), `window` and `oracle` (both use target labels, sensitivity analyses
only) and `fixed05`. Add `--text-dir text_features` to evaluate CMCE with its training-matched
Prompt-A features.

## Train

Check the text features first. Without this check, a missing directory makes CMCE train silently
without text:

    python tools/preflight_check.py --entry train_unified
    python tools/preflight_check.py --entry ablation_cmce

Visual-only and Simple baselines (unified text):

    python train.py --config configs/cmce_v5_canonical.yaml --alignment none --unified-prompt --seed 42
    python train.py --config configs/cmce_v5_canonical.yaml --alignment simple --unified-prompt --seed 42

CMCE, trained the way the released CMCE weights were (Prompt-A features):

    python ablation_cmce.py --config configs/cmce_v5_canonical.yaml --ablation full --seed 42 --gpu 0

Checkpoints are written below `checkpoints/whu_building/`. Confirm that a CMCE run really used text:

    python tools/audit_textless_checkpoint.py --text-dir text_features \
        --checkpoints checkpoints/whu_building/ablation/ablation_full_cmce_seed42/best_model.pth

Pass `--alignment` when evaluating your own checkpoints with `tools/evaluate_transfer.py`. Training
is not bit-for-bit deterministic across GPUs and library versions, so retrained models differ from
the released ones seed by seed.

`train.py --alignment cmce --unified-prompt` trains CMCE on the unified descriptions instead, which
is not how the released CMCE weights were produced. The other `--ablation` choices of
`ablation_cmce.py` are development variants; only `full` corresponds to released weights.

## Mechanistic evaluation and controls

    python experiments/mechanism_probe.py --checkpoint weights/cmce_whu_seed42.pth
    python experiments/text_substitution_control.py                            # unified descriptions
    python experiments/text_substitution_control.py --text-dir text_features   # Prompt-A features
    python experiments/anchor_collapse.py
    python experiments/convergence_trace.py --seeds 42,123,456

Fixed-random control, for each seed:

    python experiments/fixed_random_control/run_nonsemantic_capacity_control.py --seed 42 --gpu 0
    python experiments/fixed_random_control/evaluate_nonsemantic_capacity.py \
        --checkpoint results/fixed_random_control/seed42/best_model.pth \
        --output results/fixed_random_control/seed42/evaluation_inria_mass.json
    python experiments/fixed_random_control/validate_nonsemantic_capacity_results.py \
        --run-dir results/fixed_random_control/seed42

and then across seeds:

    python experiments/fixed_random_control/summarize_nonsemantic_capacity.py \
        --eval-json results/fixed_random_control/seed0/evaluation_inria_mass.json \
                    results/fixed_random_control/seed42/evaluation_inria_mass.json ... \
        --output results/fixed_random_control/summary.json

Every script accepts `--data-root` when the data are not in `data/`.

## Code version

`python tools/check_code_version.py` confirms that the model and training code match the version
that produced the paper's results.

## Citation

    @article{deng2026textconditioned,
      title   = {Text-Conditioned Cross-Domain Generalization in Remote Sensing Building Extraction:
                 A Mechanistic Evaluation of Semantic Anchor Refinement},
      author  = {Deng, Yongtao and Lei, Dajiang and Zhang, Liping and Li, Jiaxin and Peng, Yidong and Li, Weisheng},
      journal = {Expert Systems with Applications},
      year    = {2026},
      note    = {Accepted. Volume, pages and DOI to be added.}
    }

## License

MIT, see [LICENSE](LICENSE). The datasets and Janus-Pro-1B are subject to their own licences.
