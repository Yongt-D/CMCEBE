# Data

The datasets are not redistributed here. Download them from their providers, arrange them as
below, and add the text data released with this repository.

- WHU Building (aerial): http://gpcv.whu.edu.cn/data/building_dataset.html
- Inria Aerial Image Labeling: https://project.inria.fr/aerialimagelabeling/
- Massachusetts Buildings: https://www.cs.toronto.edu/~vmnih/data/

## Layout

    data/
      whu_building/{train,val,test}/
        images/                   512 x 512 RGB tiles (.tif)
        labels/                   binary building masks (.tif)
        unified_janus_texts/      <stem>.txt   from cmce_text_unified.zip
        unified_janus_features/   <stem>.pt    from cmce_text_unified.zip
        text_features/            <stem>.pt    from cmce_text_promptA_features.zip
      inria/{train,val,test}/            .tif images, 0/255 masks
      massachusetts/{train,val,test}/    .png images, 0/255 masks

`utils/unified_data_manager.py` pairs each image with the label and the text feature that share
its file stem, and binarizes masks at 127.

## Splits

The exact membership of every split is listed in `data/splits/<dataset>_<split>.txt`, one image
stem per line.

| Dataset | Train | Val | Test |
|---|---|---|---|
| WHU Building | 4,736 | 1,036 | 2,416 |
| Inria | 10,125 | 1,458 | 2,997 |
| Massachusetts | 485 | 16 | 40 |

- **WHU Building**: the official 512 x 512 tiles and split.
- **Inria**: the split is made at tile level over the 180 labelled 5,000 x 5,000 tiles of Austin,
  Chicago, Kitsap, Tyrol-W and Vienna (125 / 18 / 37 tiles), so no tile contributes to more than
  one split. Each tile is cut into a non-overlapping 9 x 9 grid of 512 x 512 patches (stride 512,
  row-major from the upper-left corner, patch index 1-81 in the file name `<city><tile>_<index>`);
  the right and bottom 392-pixel margins are unused. Tiles per split: `data/splits/inria_tiles.json`.
- **Massachusetts**: 512 x 512 patches from the upper-left 1,024 x 1,024 region of each
  1,500 x 1,500 tile (offsets 0 and 512, file name `<tile>_patch_<a>_<b>`). Some training patches
  are absent; the stem lists and `data/splits/massachusetts_tiles.json` are authoritative.

Native ground sample distances are kept; nothing is resampled to a common resolution.

## Text data

| Archive | Size | Contents |
|---|---|---|
| `cmce_text_unified.zip` | 227 MB | `unified_janus_texts/` and `unified_janus_features/` for every split of the three datasets |
| `cmce_text_promptA_features.zip` | 96 MB | Prompt-A `text_features/` for WHU train/val/test, Inria test and Massachusetts train/val/test |

Download both archives from the Google Drive folder linked in the [main README](../README.md#downloads)
and unzip them into `data/`; paths inside start with `<dataset>/<split>/`.
`checksums/text_data_SHA256SUMS.txt` lists the archive checksums, and each archive carries
`MANIFEST.json` with the SHA-256 of every file. [docs/TEXT_PIPELINE.md](../docs/TEXT_PIPELINE.md) explains how the descriptions and
embeddings were produced and which models use which set.

| Task | Text directories needed |
|---|---|
| Evaluate the released weights (`tools/evaluate_transfer.py`) | target test `unified_janus_features/` |
| Train None or Simple (`train.py --unified-prompt`) | WHU train/val/test `unified_janus_features/` |
| Train CMCE as the released weights were trained (`ablation_cmce.py --ablation full`) | WHU train/val/test `text_features/` |
| Legacy-directory text controls (`--text-dir text_features`) | Inria test and WHU train `text_features/` |

Run `python tools/preflight_check.py` before training. Without it, a missing text directory makes
CMCE train silently without text.
