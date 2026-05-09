# Cross-Modal Co-Evolution: A Bidirectional Vision-Language Refinement Framework for Remote Sensing Image Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Journal](https://img.shields.io/badge/Submitted%20to-IEEE%20TMM-blue)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6046)

> **Note:** This is the official implementation of the paper submitted to **IEEE Transactions on Multimedia (TMM)**.

## 🚀 News
- **[2026-05-xx]**: The paper has been submitted to *IEEE Transactions on Multimedia*. The code will be released upon acceptance.
- **[2026-02-02]**: Repository created.

## 📄 Abstract
Effective cross-modal alignment between visual features and linguistic semantics is a fundamental challenge in multimodal learning. Existing vision-language fusion approaches predominantly adopt a unidirectional paradigm, where textual information enhances visual representations without reciprocal refinement of the text modality. This asymmetry prevents the full exploitation of complementary cross-modal information. We propose **CMCE (Cross-Modal Co-Evolution)**, a novel bidirectional cross-modal refinement framework that enables mutual enhancement between visual and textual modalities through iterative co-evolution. CMCE introduces three key contributions: (1) a *Semantic Anchor Decomposition* mechanism that factorizes holistic text embeddings into multiple specialized anchors capturing complementary visual aspects of the scene; (2) a *Bidirectional Closed-Loop Refinement* process that iteratively refines both visual features and text representations through cross-modal feedback; and (3) a *Convergence-Aware Iterative Mechanism* with stability properties grounded in Lyapunov-inspired analysis. Applied to building extraction from remote sensing imagery, CMCE achieves state-of-the-art performance on three benchmark datasets: 90.49% IoU on WHU Building, 80.56% on Inria, and 69.91% on Massachusetts.

## 🔑 Highlights
- **Bidirectional refinement**: visual and text representations co-evolve through cross-modal feedback rather than text-to-vision only.
- **Semantic Anchor Decomposition**: holistic text embeddings are factorized into multiple specialized anchors.
- **Convergence-aware design**: refinement dynamics are analyzed under a Lyapunov-inspired framework with empirical validation.
- **Strong empirical results**: SOTA across three building extraction benchmarks; 50+ IoU points above zero-shot CLIPSeg.

## ⏳ Code Release Plan
Thank you for your interest in our work!

The source code is currently being organized and cleaned. We plan to release the full training and evaluation code **upon paper acceptance**, including:

- [ ] Data preprocessing utilities (WHU / Inria / Massachusetts)
- [ ] Model architecture (CMCE alignment module + U-Net backbone)
- [ ] Training and evaluation scripts
- [ ] Pre-trained checkpoints
- [ ] Per-anchor attention visualization tools

Please **Star** ⭐ this repository to receive notifications when the code is released.

## 📖 Citation
If you find this work useful for your research, please consider citing our paper (BibTeX entry will be updated upon acceptance):

```bibtex
@article{CMCEBE,
  title  = {Cross-Modal Co-Evolution: A Bidirectional Vision-Language Refinement Framework for Remote Sensing Image Segmentation},
  author = {Deng, Yongtao and Lei, Dajiang and Zhang, Liping and Li, Jiaxin and Peng, Yidong and Li, Weisheng},
  journal = {Submitted to IEEE Transactions on Multimedia},
  year   = {2026}
}
```


