"""
Training entrypoint for ARCModel.

This script:
- loads ARC task batches from data_loader.py
- trains model.ARCModel end to end
- writes each run into results/latest_run_<timestamp>
- stores config, checkpoints, logs, epoch stats, and plots
"""

import argparse
import csv
import json
import os
import random

from datetime import datetime
from pathlib import Path

MPLCONFIGDIR = Path(".mplconfig").resolve()
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader, Subset

from data_loader import ARCTaskDataset, arc_collate_fn
from model import ARCModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ARCModel.")
    parser.add_argument("--data-root", default="data", help="Prepared data root.")
    parser.add_argument("--results-dir", default="results", help="Parent directory for run outputs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--kmax", type=int, default=10, help="Max number of example slots.")
    parser.add_argument("--pad-size", type=int, default=30, help="Grid pad size.")
    parser.add_argument("--device", default="cpu", help="Training device.")
    parser.add_argument("--tiny-overfit", type=int, default=0, help="Use only the first N train tasks.")
    parser.add_argument("--eval-limit", type=int, default=0, help="Use only the first N eval tasks.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one batch through forward/backward and exit.")
    parser.add_argument("--d-model", type=int, default=128, help="Transformer hidden size.")
    parser.add_argument("--nhead", type=int, default=8, help="Attention heads.")
    parser.add_argument("--num-encoder-layers", type=int, default=3, help="Grid encoder layers.")
    parser.add_argument("--num-query-layers", type=int, default=2, help="Query refinement layers.")
    parser.add_argument("--num-pair-rule-tokens", type=int, default=4, help="Pair rule token count.")
    parser.add_argument("--num-global-tokens", type=int, default=4, help="Global token count.")
    parser.add_argument("--num-rule-tokens", type=int, default=4, help="Final rule token count.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
    parser.add_argument("--pcn-num-steps", type=int, default=3, help="Number of PCN refinement steps.")
    parser.add_argument("--pcn-step-size", type=float, default=0.5, help="PCN update step size.")
    parser.add_argument("--pcn-energy-weight", type=float, default=0.1, help="Weight for PCN energy in total loss.")
    parser.add_argument("--max-objects", type=int, default=16, help="Max objects per grid.")
    parser.add_argument("--object-shape-pool", type=int, default=5, help="Adaptive pooled shape size.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def maybe_subset(dataset, limit: int):
    if limit <= 0:
        return dataset
    limit = min(limit, len(dataset))
    return Subset(dataset, list(range(limit)))


def build_loaders(args: argparse.Namespace):
    train_dataset = ARCTaskDataset(
        data_root=args.data_root,
        split="train",
        kmax=args.kmax,
        pad_size=args.pad_size,
    )
    eval_dataset = ARCTaskDataset(
        data_root=args.data_root,
        split="evaluation",
        kmax=args.kmax,
        pad_size=args.pad_size,
    )

    train_dataset = maybe_subset(train_dataset, args.tiny_overfit)
    eval_dataset = maybe_subset(eval_dataset, args.eval_limit)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=arc_collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=arc_collate_fn,
    )
    return train_loader, eval_loader


def create_run_dir(results_root: Path) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / f"latest_run_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = results_root / f"latest_run_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def log_message(log_path: Path, message: str) -> None:
    print(message)
    with open(log_path, "a") as handle:
        handle.write(message + "\n")


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def save_history_csv(path: Path, history: list) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_ce_loss",
        "train_pcn_energy",
        "train_cell_acc",
        "train_grid_acc",
        "eval_loss",
        "eval_ce_loss",
        "eval_pcn_energy",
        "eval_cell_acc",
        "eval_grid_acc",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def plot_metric(history: list, metric_key: str, train_label: str, eval_label: str, title: str, ylabel: str, output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_values = [row[f"train_{metric_key}"] for row in history]
    eval_values = [row[f"eval_{metric_key}"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_values, marker="o", label=train_label)
    plt.plot(epochs, eval_values, marker="o", label=eval_label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_overview(history: list, output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(3, 1, figsize=(9, 13))

    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["eval_loss"] for row in history], marker="o", label="eval")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [100.0 * row["train_grid_acc"] for row in history], marker="o", label="train")
    axes[1].plot(epochs, [100.0 * row["eval_grid_acc"] for row in history], marker="o", label="eval")
    axes[1].set_title("Exact Solve Rate")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Percent")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, [100.0 * row["train_cell_acc"] for row in history], marker="o", label="train")
    axes[2].plot(epochs, [100.0 * row["eval_cell_acc"] for row in history], marker="o", label="eval")
    axes[2].set_title("Cell Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Percent")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_model(args: argparse.Namespace) -> ARCModel:
    return ARCModel(
        d_model=args.d_model,
        grid_size=args.pad_size,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_query_layers=args.num_query_layers,
        num_pair_rule_tokens=args.num_pair_rule_tokens,
        num_global_tokens=args.num_global_tokens,
        num_rule_tokens=args.num_rule_tokens,
        dropout=args.dropout,
        pcn_num_steps=args.pcn_num_steps,
        pcn_step_size=args.pcn_step_size,
        pcn_energy_weight=args.pcn_energy_weight,
        max_objects=args.max_objects,
        object_shape_pool=args.object_shape_pool,
    )


def format_parameter_count(count: int) -> str:
    if count >= 1_000_000:
        return f"{count:,} ({count / 1_000_000:.2f}M)"
    if count >= 1_000:
        return f"{count:,} ({count / 1_000:.2f}K)"
    return str(count)


def print_model_parameter_summary(model: ARCModel) -> None:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"model_total_params={format_parameter_count(total_params)}")
    print(f"model_trainable_params={format_parameter_count(trainable_params)}")


def run_smoke_test(model: ARCModel, loader, device: torch.device) -> None:
    model.train()
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    output = model.training_step(batch)
    output["loss"].backward()

    print("Smoke Test")
    print("example_inputs", tuple(batch["example_inputs"].shape))
    print("example_outputs", tuple(batch["example_outputs"].shape))
    print("query_input", tuple(batch["query_input"].shape))
    print("query_output", tuple(batch["query_output"].shape))
    print("logits", tuple(output["logits"].shape))
    print("loss", float(output["loss"].detach()))
    print("cell_acc", float(output["cell_acc"].detach()))
    print("grid_acc", float(output["grid_acc"].detach()))


def train_one_epoch(model: ARCModel, loader, optimizer, device: torch.device) -> dict:
    model.train()

    total_examples = 0
    total_loss = 0.0
    total_cell_acc = 0.0
    total_grid_acc = 0.0
    total_ce_loss = 0.0
    total_pcn_energy = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        batch_size = int(batch["query_output"].shape[0])

        optimizer.zero_grad(set_to_none=True)
        output = model.training_step(batch)
        loss = output["loss"]
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_ce_loss += float(output["ce_loss"].detach()) * batch_size
        total_pcn_energy += float(output["pcn_energy"].detach()) * batch_size
        total_cell_acc += float(output["cell_acc"].detach()) * batch_size
        total_grid_acc += float(output["grid_acc"].detach()) * batch_size

    denominator = max(total_examples, 1)
    return {
        "loss": total_loss / denominator,
        "ce_loss": total_ce_loss / denominator,
        "pcn_energy": total_pcn_energy / denominator,
        "cell_acc": total_cell_acc / denominator,
        "grid_acc": total_grid_acc / denominator,
        "num_examples": total_examples,
    }


@torch.no_grad()
def evaluate(model: ARCModel, loader, device: torch.device) -> dict:
    model.eval()

    total_examples = 0
    total_loss = 0.0
    total_cell_acc = 0.0
    total_grid_acc = 0.0
    total_ce_loss = 0.0
    total_pcn_energy = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        batch_size = int(batch["query_output"].shape[0])
        output = model.training_step(batch)

        total_examples += batch_size
        total_loss += float(output["loss"].detach()) * batch_size
        total_ce_loss += float(output["ce_loss"].detach()) * batch_size
        total_pcn_energy += float(output["pcn_energy"].detach()) * batch_size
        total_cell_acc += float(output["cell_acc"].detach()) * batch_size
        total_grid_acc += float(output["grid_acc"].detach()) * batch_size

    denominator = max(total_examples, 1)
    return {
        "loss": total_loss / denominator,
        "ce_loss": total_ce_loss / denominator,
        "pcn_energy": total_pcn_energy / denominator,
        "cell_acc": total_cell_acc / denominator,
        "grid_acc": total_grid_acc / denominator,
        "num_examples": total_examples,
    }


def build_summary(args: argparse.Namespace, history: list, best_epoch: dict, run_dir: Path) -> dict:
    final_epoch = history[-1]
    return {
        "run_dir": str(run_dir),
        "timestamp": run_dir.name.replace("latest_run_", ""),
        "epochs": args.epochs,
        "data_root": args.data_root,
        "device": args.device,
        "final": final_epoch,
        "best_eval_loss": {
            "epoch": best_epoch["epoch"],
            "eval_loss": best_epoch["eval_loss"],
            "eval_cell_acc": best_epoch["eval_cell_acc"],
            "eval_grid_acc": best_epoch["eval_grid_acc"],
        },
        "best_eval_grid_acc": max(row["eval_grid_acc"] for row in history),
        "best_eval_cell_acc": max(row["eval_cell_acc"] for row in history),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    train_loader, eval_loader = build_loaders(args)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print_model_parameter_summary(model)

    if args.smoke_test:
        run_smoke_test(model, train_loader, device)
        return

    run_dir = create_run_dir(Path(args.results_dir))
    log_path = run_dir / "train.log"

    config = vars(args).copy()
    config["run_dir"] = str(run_dir)
    save_json(run_dir / "config.json", config)
    log_message(log_path, f"run_dir={run_dir}")
    log_message(log_path, f"device={device}")
    log_message(log_path, f"model_total_params={format_parameter_count(sum(parameter.numel() for parameter in model.parameters()))}")
    log_message(log_path, f"model_trainable_params={format_parameter_count(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))}")

    history = []
    best_eval_loss = float("inf")
    best_epoch_record = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        eval_metrics = evaluate(model, eval_loader, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_pcn_energy": train_metrics["pcn_energy"],
            "train_cell_acc": train_metrics["cell_acc"],
            "train_grid_acc": train_metrics["grid_acc"],
            "eval_loss": eval_metrics["loss"],
            "eval_ce_loss": eval_metrics["ce_loss"],
            "eval_pcn_energy": eval_metrics["pcn_energy"],
            "eval_cell_acc": eval_metrics["cell_acc"],
            "eval_grid_acc": eval_metrics["grid_acc"],
        }
        history.append(epoch_record)

        if eval_metrics["loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["loss"]
            best_epoch_record = dict(epoch_record)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": epoch_record,
                    "config": config,
                },
                run_dir / "model_best.pt",
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": epoch_record,
                "config": config,
            },
            run_dir / "model_last.pt",
        )

        log_message(
            log_path,
            (
                f"epoch={epoch} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_ce_loss={train_metrics['ce_loss']:.4f} "
                f"train_pcn_energy={train_metrics['pcn_energy']:.4f} "
                f"train_grid_acc={100.0 * train_metrics['grid_acc']:.2f}% "
                f"train_cell_acc={100.0 * train_metrics['cell_acc']:.2f}% "
                f"eval_loss={eval_metrics['loss']:.4f} "
                f"eval_ce_loss={eval_metrics['ce_loss']:.4f} "
                f"eval_pcn_energy={eval_metrics['pcn_energy']:.4f} "
                f"eval_grid_acc={100.0 * eval_metrics['grid_acc']:.2f}% "
                f"eval_cell_acc={100.0 * eval_metrics['cell_acc']:.2f}%"
            ),
        )

    save_json(run_dir / "history.json", history)
    save_history_csv(run_dir / "history.csv", history)

    summary = build_summary(args, history, best_epoch_record, run_dir)
    save_json(run_dir / "summary.json", summary)

    plot_metric(
        history,
        metric_key="loss",
        train_label="train loss",
        eval_label="eval loss",
        title="Train vs Eval Loss",
        ylabel="Loss",
        output_path=run_dir / "loss_curve.png",
    )
    plot_metric(
        history,
        metric_key="grid_acc",
        train_label="train solve rate",
        eval_label="eval solve rate",
        title="Train vs Eval Exact Solve Rate",
        ylabel="Rate",
        output_path=run_dir / "solve_rate_curve.png",
    )
    plot_metric(
        history,
        metric_key="cell_acc",
        train_label="train cell acc",
        eval_label="eval cell acc",
        title="Train vs Eval Cell Accuracy",
        ylabel="Rate",
        output_path=run_dir / "cell_accuracy_curve.png",
    )
    plot_metric(
        history,
        metric_key="pcn_energy",
        train_label="train PCN energy",
        eval_label="eval PCN energy",
        title="Train vs Eval PCN Energy",
        ylabel="Energy",
        output_path=run_dir / "pcn_energy_curve.png",
    )

    plot_metric(
        history,
        metric_key="ce_loss",
        train_label="train CE loss",
        eval_label="eval CE loss",
        title="Train vs Eval CE Loss",
        ylabel="Loss",
        output_path=run_dir / "ce_loss_curve.png",
    )

    
    plot_overview(history, run_dir / "overview.png")

    log_message(log_path, f"saved_summary={run_dir / 'summary.json'}")
    log_message(log_path, f"saved_best_model={run_dir / 'model_best.pt'}")
    log_message(log_path, f"saved_last_model={run_dir / 'model_last.pt'}")


if __name__ == "__main__":
    main()
