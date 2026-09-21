# SignSparK-BT

Back-translation for sign language production: takes SMPL+MANO pose features
and produces spoken language text. Use it to score generated sign sequences by
translating the poses back to text and comparing it against the source sentence.

This is the evaluation companion to [SignSparK](https://github.com/JianHe0628/SignSparK)
and it reads from the same LMDB datasets.

---

## Quickstart

From a clean checkout to a score, in four steps.

```bash
# 1. install
conda env create -f environment.yml && conda activate signspark-bt
pip install -e .

# 2. download pose data (from the SignSparK repo) and point it to the download location
python tools/download_data.py --datasets PHOENIX-2014T --dest ./data
export DATA_ROOT=$(pwd)/data

# 3. download our back-translation models
python tools/download_models.py --datasets PHOENIX-2014T --dest ./checkpoints

# 4. check it works: this should print BLEU-4 15.00
signspark-bt score checkpoints/PHOENIX-2014T --split test
```

Please make sure you have two things pointed at the right place:

| | must be | why |
|---|---|---|
| `DATA_ROOT` | the directory **containing** `lmdb/` | configs read `${DATA_ROOT}/lmdb/{train,dev,test}` |
| the model directory | holds `config.yaml`, `best.ckpt`, `gls.vocab`, `txt.vocab` | `score` resolves all four from it |

`DATA_ROOT` is only needed for `train`, `test`, and `score --split`. Scoring a
sample dump or your own features needs the model directory alone.

---

## Install

Conda, for a pinned and reproducible environment:

```bash
conda env create -f environment.yml
conda activate signspark-bt
```

Or pip:

```bash
pip install -r requirements.txt   # pinned, equivalent to environment.yml
pip install -e .
```

Our provided requirements are targeted for CUDA 12.8 wheels, needed for Blackwell / RTX 50-series. For a
different CUDA version change the index URL and drop the `+cu128` tag.

To install just the package and let pip resolve loose dependency bounds:

```bash
pip install -e ".[zh,logging]"
```

| Extra | Needed for |
|---|---|
| `zh` | CSL-Daily (Chinese word segmentation via jieba) |
| `logging` | Weights & Biases experiment tracking |
| `pseudogloss` | generating How2Sign pseudo-glosses with spaCy |
| `dev` | running the tests |

Requires Python 3.9+, PyTorch and LMDB.

## Data

The pose data is distributed by SignSparK on the HuggingFace Hub at
[LionelLow/SignSparK_data](https://huggingface.co/datasets/LionelLow/SignSparK_data):

```bash
python tools/download_data.py --datasets CSL-Daily How2Sign PHOENIX-2014T --dest ./data
export DATA_ROOT=$(pwd)/data
```

(`tools/download_data.py` lives in the SignSparK repo.) The expected layout is:

```
$DATA_ROOT/lmdb/{train,dev,test}/<Dataset>_reopt_<split>.lmdb
```

Each clip provides a translation, a gloss, and per-frame SMPL-X 6D rotations.
Our backtranslation model consumes a 240-dimensional vector per frame:

| Stream | Source | Dims |
|---|---|---|
| Body | `body_features[:, 66:]` (last 10 joints, legs dropped) | 60 |
| Left hand | `left_features[:, :90]` | 90 |
| Right hand | `right_features[:, :90]` | 90 |

Copy `.env.example` to `.env` and fill in `DATA_ROOT` and, if you use W&B,
`WANDB_API_KEY` and `WANDB_ENTITY`.

### Pseudo-glosses for How2Sign

As How2Sign is not gloss annotated, we generate pseudo-glosses as a naive fix (detailed in supplementary):

```bash
python -m spacy download en_core_web_lg   # 3.8.0
python tools/make_pseudo_glosses.py --dataset How2Sign
```

Output is sensitive to the spaCy model version, so pin `en_core_web_lg==3.8.0`
if you need to reproduce a specific run.

## Usage

```bash
# Train
signspark-bt train configs/phoenix14t.yaml

# Evaluate a checkpoint on the dev and test splits
signspark-bt test configs/phoenix14t.yaml --ckpt checkpoints/bt_phoenix14t/best.ckpt
```

Training writes to the config's `model_dir` (`checkpoints/bt_phoenix14t`), while
downloaded models land in `checkpoints/<Dataset>` (`checkpoints/PHOENIX-2014T`).
Both layouts work anywhere a model directory is expected.

Any config value can be overridden on the command line:

```bash
signspark-bt train configs/how2sign.yaml --training.batch_size 64 --training.learning_rate 1e-4
```

The first run indexes every clip in the LMDB, which is I/O bound. The index is
cached under `$DATA_ROOT/.bt_index_cache`, so later runs start immediately.

## Scoring sign language production

Additionally, we provide a `score` script to easily back-translate poses and
report BLEU, chrF and ROUGE. Given a SignSparK sample .npy dump, the script
scores the ground-truth and the generated poses in one pass and reports the
drop between them:

```bash
signspark-bt score checkpoints/<model> \
    --body-npy  .../body/<run>/seed102_clampstep0_<dataset>.npy \
    --hand-npy  .../hand/<run>/seed102_clampstep0_<dataset>.npy
```

| Dataset | | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | chrF | ROUGE |
|---|---|---:|---:|---:|---:|---:|---:|
| **PHOENIX-2014T** | ground truth | 38.08 | 25.43 | 18.80 | 15.00 | 37.34 | 38.18 |
| 642 clips | SignSparK KF2P | 36.24 | 23.52 | 17.07 | 13.38 | 35.19 | 36.20 |
| | *drop* | *−4.8%* | *−7.5%* | *−9.2%* | *−10.8%* | *−5.7%* | *−5.2%* |
| **CSL-Daily** | ground truth | 26.00 | 15.11 | 9.85 | 6.98 | 7.50 | 26.88 |
| 1162 clips | SignSparK KF2P | 24.79 | 14.06 | 8.99 | 6.27 | 6.95 | 25.33 |
| | *drop* | *−4.7%* | *−6.9%* | *−8.8%* | *−10.2%* | *−7.3%* | *−5.8%* |
| **How2Sign** | ground truth | 23.07 | 10.33 | 5.73 | 3.53 | 19.81 | 23.58 |
| 2183 clips | SignSparK KF2P | 22.26 | 9.69 | 5.39 | 3.34 | 19.36 | 22.59 |
| | *drop* | *−3.5%* | *−6.2%* | *−6.0%* | *−5.5%* | *−2.2%* | *−4.2%* |

CSL-Daily needs the `zh` extra: its text is segmented with jieba and scored per
character, so its chrF is not comparable with the other two.

Note that the ground-truth values are the upper-bound of the back-translation model. The body and hand dumps should come from the same `sample_all.py` run in the SignSparK repo. Also note that the face stream is unused: the model only takes body (60) plus both hands (90 each).


> **Note:** Scores reported here differ from the main paper. Both the ground-truth and predicted pose scores here have been recomputed using a back-translation model retrained to match the reoptimized LMDBs released at [LionelLow/SignSparK_data](https://huggingface.co/datasets/LionelLow/SignSparK_data). The back-translation drop is also smaller than in the paper, since the released checkpoints are trained on 15× more data of higher quality.


## Using a trained model from Python

```python
import torch
from signspark_bt import make_back_translation_model, back_translate

model = make_back_translation_model("checkpoints/PHOENIX-2014T")

# B x T x 240 SMPL+MANO features: body (60) + left hand (90) + right hand (90)
poses = torch.rand(8, 64, 240)

sentences = back_translate(model=model, poses=poses)
```

`poses` may also be a list of `T x 240` tensors of differing length. Empty or
`None` entries come back as empty strings at their original positions, so the
output always lines up with the input. Beam size and alpha are read from each model's `config.yaml`, so you do not
need to set them by hand.

## Download the Pre-trained models

The back-translation models are on the HuggingFace Hub at
[LionelLow/SignSparK_BT](https://huggingface.co/LionelLow/SignSparK_BT):

```bash
python tools/download_models.py --datasets PHOENIX-2014T CSL-Daily How2Sign --dest ./checkpoints
```

After download, each dataset directory should hold `config.yaml`, `best.ckpt` and two vocabulary
files. See [Scoring sign language production](#scoring-sign-language-production) to find our **updated** back translation scores.

> **Note:** Our original back-translation scores in the paper was trained on the raw, unoptimization data. This has since been superseded by this official release to match our improved data extractions.

## Configuration

Configs live in `configs/`. `${VAR}` and `~` are expanded anywhere in the file.

The fields most worth knowing:

| Key | Meaning |
|---|---|
| `data.base_path` | directory holding `train/`, `dev/`, `test/` |
| `data.dataset` | matched against the LMDB filename, e.g. `PHOENIX-2014T` |
| `data.version` | selects dataset-specific scoring (PHOENIX gloss clean-up, CSL character-level BLEU) |
| `data.tokenizer` | `whitespace`, or `jieba` for Chinese |
| `data.normalize_punctuation` | detach the sentence-final period into its own token (Latin script); off for CSL-Daily |
| `data.max_sgn_length` | frames per clip; longer clips are truncated (default 300) |
| `data.pseudo_gloss_dir` | where to find generated glosses, if any |
| `training.recognition_loss_weight` | set to `0.0` to train translation only |
| `training.num_workers` | DataLoader workers |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite covers CTC decoding, collation and masking, vocabularies, metrics and
pseudo-gloss generation. It needs no data and no GPU.

## Citation

If you use this code, the released back-translation models, or the SignSparK
data, please cite:

```bibtex
@inproceedings{low2026signspark,
  title={SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning},
  author={Low, Jianhe and Symeonidis-Herzig, Alexandre and Ivashechkin, Maksym and Sincan, Ozge Mercanoglu and Bowden, Richard},
  booktitle={European Conference on Computer Vision},
  pages={648--670},
  year={2026},
  organization={Springer}
}
```

## Licence

Apache 2.0; see `LICENSE`. This code derives from
[JoeyNMT](https://github.com/joeynmt/joeynmt) and
[Sign Language Transformers](https://github.com/neccam/slt),
and vendors corpus BLEU and chrF from
[sacreBLEU](https://github.com/mjpost/sacrebleu) and ROUGE-L from
[coco-caption](https://github.com/tylin/coco-caption). We thank them for open-sourcing their code and making this codebase possible.

The underlying datasets (CSL-Daily, How2Sign, RWTH-PHOENIX-Weather 2014T) carry
their own licences, which govern any use of the data and of models trained on
it.
