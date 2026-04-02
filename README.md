# ARC React

Minimal ARC training repo.

## Setup

```bash
bash setup.sh
source .venv/bin/activate
```

`setup.sh` auto-selects a PyTorch wheel based on the host:
- NVIDIA + CUDA 12.8 -> `cu128`
- NVIDIA + CUDA 12.6 -> `cu126`
- NVIDIA + CUDA 11.8 -> `cu118`
- no NVIDIA GPU detected -> CPU wheel

You can override this manually if needed:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash setup.sh
```

## Prepare Data

Plain ARC-AGI-1 data:

```bash
python scripts/prepare_data.py --workspace .
```

TRM-style augmented ARC-AGI-1 data:

```bash
python scripts/prepare_data.py --workspace .
```

The same command deletes local `ARC-AGI` / `ARC-AGI-2` clones after a successful run unless you add `--keep-repo-clones`.

Prepare both ARC-AGI-1 and ARC-AGI-2 with the same augmentation style:

```bash
python scripts/prepare_data.py --workspace . --include-arc-agi-2
```

This creates:

- `data_arc_agi_1/` for ARC-AGI-1
- `data_arc_agi_2/` for ARC-AGI-2
- `augmented_train/` inside each data root when `--num-aug > 0`

## Train

Quick smoke test:

```bash
python3 train.py --smoke-test --device cpu --batch-size 1
```

Small local run:

```bash
python3 train.py --device cpu --batch-size 1 --epochs 3 --d-model 64 --nhead 4 --num-encoder-layers 2 --num-query-layers 1 --num-pair-rule-tokens 2 --num-global-tokens 2 --num-rule-tokens 2 --kmax 4 --num-workers 0
```

Train on the augmented ARC-AGI-1 split:

```bash
python3 train.py --data-root data_arc_agi_1 --train-split-name augmented_train
```

Train on ARC-AGI-2:

```bash
python3 train.py --data-root data_arc_agi_2
```

Train on augmented ARC-AGI-2:

```bash
python3 train.py --data-root data_arc_agi_2 --train-split-name augmented_train
```

On an NVIDIA machine with a CUDA-verified setup, switch to `--device cuda`.

## Analyze

```bash
python3 scripts/analysis.py --results-dir results
```
