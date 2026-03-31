# ARC React

Minimal ARC training repo.

## Setup

```bash
bash setup.sh
source .venv/bin/activate
```

## Train

Quick smoke test:

```bash
python3 train.py --smoke-test --device cpu --batch-size 1
```

Small local run:

```bash
python3 train.py --device cpu --batch-size 1 --epochs 3 --d-model 64 --nhead 4 --num-encoder-layers 2 --num-query-layers 1 --num-pair-rule-tokens 2 --num-global-tokens 2 --num-rule-tokens 2 --kmax 4 --num-workers 0
```

On an NVIDIA machine, switch to `--device cuda`.

## Analyze

```bash
python3 analysis.py --results-dir results
```
