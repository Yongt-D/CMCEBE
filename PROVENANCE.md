# Provenance and release checks

Where the released code, weights and text data come from, and how they were checked before release
(2026-09-14).

## Code

- The 17 model and training files checked by `tools/check_code_version.py` are identical, after
  removing carriage returns, to the version pinned on 2026-06-29, before the paper's experiments
  were run.
- The authors' working copy later gained one change: an optional `start_epoch` argument to
  `CleanedUnifiedTrainer.train`, added on 2026-09-02 so that an interrupted run of the fixed-random
  control (seed 0) could resume from its saved checkpoints; that run's protocol records its last
  resume, at epoch 80. With the default value the computation is unchanged. This release ships the
  pinned version, so the resume option was removed from `run_nonsemantic_capacity_control.py`. The
  model-wrapper hash recorded by the other four control seeds equals the pinned file (stored with
  Windows line endings), and all other released results predate the change.
- `tools/evaluate_transfer.py` and the scripts in `experiments/` are the analysis scripts used for
  the paper, with local paths turned into arguments, threshold protocols renamed (honest -> source,
  paper -> window, half -> fixed05), and a second analysis on superseded checkpoints removed from the
  anchor-collapse script. The guards in `tools/` (preflight check, text-free audit, code-version
  check) were written during the experiments and simplified for release.

## Weights

Each released file was exported from an original checkpoint, whose SHA-256 it stores as
`source_checkpoint_sha256`. Only the model parameters and a few metadata fields were kept (the
optimizer and scheduler states and the training histories were dropped), and every parameter
tensor was compared with the original after export.

| File | SHA-256 (first 16 hex) | Epoch stored | WHU validation threshold |
|---|---|---|---|
| `cmce_whu_seed0.pth` | `7900d506f9d41cfd` | 85 | 0.60 |
| `cmce_whu_seed42.pth` | `6ec8d8af4c62605a` | 88 | 0.60 |
| `cmce_whu_seed123.pth` | `378b5f4a2d1301fc` | 46 | 0.58 |
| `cmce_whu_seed456.pth` | `d201d1cc41c0f0a2` | 89 | 0.60 |
| `cmce_whu_seed666.pth` | `84df291fb803a152` | 79 | 0.54 |
| `none_whu_seed0.pth` | `aecc0462e2efdb10` | 89 | 0.60 |
| `none_whu_seed42.pth` | `c886c645d8a1bf38` | 93 | 0.60 |
| `none_whu_seed123.pth` | `12f0224d93a3eb6d` | 89 | 0.56 |
| `none_whu_seed456.pth` | `473b55f43a12c367` | 79 | 0.60 |
| `none_whu_seed666.pth` | `f15cbe0e742a249b` | 53 | 0.60 |
| `simple_whu_seed0.pth` | `e1c050128a23a878` | 88 | 0.60 |
| `simple_whu_seed42.pth` | `e09162393b083715` | 85 | 0.58 |
| `simple_whu_seed123.pth` | `57bbe5be7b80633f` | 84 | 0.60 |
| `simple_whu_seed456.pth` | `6f77bfc116832d44` | 86 | 0.60 |
| `simple_whu_seed666.pth` | `2728edf4b32bb4bd` | 84 | 0.60 |

Full hashes are in `checksums/weights_SHA256SUMS.txt`.

The CMCE checkpoints were written by `ablation_cmce.py`, which reads `text_features/` (Prompt A);
the None and Simple checkpoints were written by `train.py --unified-prompt`. The checkpoints show
this themselves: `train.py` stores the configuration path and the test IoU in every checkpoint it
writes, and both are present in the None and Simple checkpoints and absent from the CMCE ones.

## Text data

`cmce_text_unified.zip` and `cmce_text_promptA_features.zip` are byte copies of the authors'
description and feature directories, and the release checks below ran on these files. Each archive
contains `MANIFEST.json` with the SHA-256 of every file. The two sets are different tensors for the
same images; [docs/TEXT_PIPELINE.md](docs/TEXT_PIPELINE.md) explains which model uses which.

## Release checks

Run with Python 3.10, PyTorch 2.8.0 and an RTX 5080, using the released code, weights and text data.

| Check | Result |
|---|---|
| Transfer tables with `tools/evaluate_transfer.py` | the 30 per-seed source-only values (None, Simple and CMCE; Inria and Massachusetts; five seeds) equal the paper's two transfer tables; on Inria, the means and standard deviations under all four threshold protocols equal the paper's threshold-sensitivity table |
| CMCE with training-matched Prompt-A features (`--text-dir text_features`) | Inria 50.62 ± 2.85, equal seed by seed to the "own description" column of the paper's legacy-directory table; Massachusetts 30.49 ± 2.00, not reported in the paper; above None in all five seeds on both targets |
| `experiments/mechanism_probe.py`, seed 42, 24 Inria test images | key/value length 1 and attention weights exactly 1 at all four scales; zero gradient for W_Q, W_K and the alignment estimator; 2 iterations in evaluation mode and 3 in training mode; zeroing the anchors changes 0.011% of pixels |
| `experiments/convergence_trace.py`, seed 42, 8 WHU test images | alignment score 1.0 at iterations 1 and 2 for every image; text-state change at most 1.8e-8 |
| Fixed-random control, dry run | fixed-vector hash and parameter count (28,761,129) equal those recorded by the original runs |
| Fixed-random control evaluator on the original seed-123 checkpoint | Massachusetts 27.82%, as recorded |
| `tools/audit_textless_checkpoint.py` on `cmce_whu_seed42.pth`, 40 WHU test images | IoU 0.837 with text and 0.054 without: trained with text |
| `tools/check_code_version.py` | all 17 files match |

The anchor-collapse and text-substitution scripts were not rerun in full for the release; they
differ from the scripts that produced the paper's tables only in paths, argument names, protocol
labels and the removed second anchor analysis.

## Not included

- Development experiments that do not back a result in the paper: earlier ablation batches, the
  DACA and Qwen experiments, diagnostics.
- The Janus-Pro-1B generation script. Its settings are documented in
  [docs/TEXT_PIPELINE.md](docs/TEXT_PIPELINE.md). The exact wording of the unified prompt was not
  preserved, so the released strings and tensors are the reference.
- The Prompt-A description strings (their embeddings are released).
- Weights of the ablation sweeps and of the controls, which can be retrained with the scripts here.
- The profiling scripts behind the efficiency table and the runs behind the multi-scale and
  deformable-attention tables, which the paper reports as single runs of an earlier configuration.
