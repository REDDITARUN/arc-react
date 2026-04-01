# Trained Model Analysis

- run_dir: `results/latest_run_20260401_120412`
- checkpoint: `model_best.pt`

## Same-size vs Changed-size

### evaluation
- same-size: count=270 solve_rate=0.00% cell_acc=81.50%
- changed-size: count=130 solve_rate=0.77% cell_acc=55.59%

### train
- same-size: count=262 solve_rate=0.00% cell_acc=84.01%
- changed-size: count=138 solve_rate=3.62% cell_acc=61.70%

## Near-miss

- evaluation: <= 3 error cells on 2.50% of tasks and 2.26% of failed tasks
- train: <= 3 error cells on 9.75% of tasks and 8.61% of failed tasks

## Token Diversity

- pair_rule_tokens: mean_norm=11.4712 mean_offdiag_cosine=0.6688 max_offdiag_cosine=0.9347
- global_tokens: mean_norm=11.1789 mean_offdiag_cosine=0.1697 max_offdiag_cosine=0.5806
- rule_tokens: mean_norm=11.4869 mean_offdiag_cosine=0.7856 max_offdiag_cosine=0.9491

## Caveat

- This model is evaluated with the ground-truth `query_output_mask`, so changed-size results reflect filling a known target footprint, not discovering the target shape from scratch.
