"""
Aggregate training run summaries from results/ and plot cross-run comparisons.
"""

import argparse
import csv
import json
import os

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MPLCONFIGDIR = REPO_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ARC training runs.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Parent results directory (relative paths are resolved from the repo root).",
    )
    return parser.parse_args()


def resolve_results_dir(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_json(path: Path):
    with open(path, "r") as handle:
        return json.load(handle)


def safe_float(value, default=0.0):
    if value is None:
        return default
    return float(value)


def short_run_label(run_name: str) -> str:
    return run_name.replace("latest_run_", "")


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    return plt


def collect_runs(results_dir: Path) -> list:
    runs = []

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith("latest_run_"):
            continue

        summary_path = child / "summary.json"
        history_path = child / "history.json"
        if not summary_path.exists() or not history_path.exists():
            continue

        summary = load_json(summary_path)
        history = load_json(history_path)

        final_metrics = summary.get("final", {})
        best_eval_loss = summary.get("best_eval_loss", {})

        run = {
            "run_name": child.name,
            "run_label": short_run_label(child.name),
            "timestamp": summary.get("timestamp", short_run_label(child.name)),
            "epochs": summary.get("epochs", len(history)),
            "final_train_loss": safe_float(final_metrics.get("train_loss")),
            "final_eval_loss": safe_float(final_metrics.get("eval_loss")),
            "final_train_ce_loss": safe_float(final_metrics.get("train_ce_loss")),
            "final_eval_ce_loss": safe_float(final_metrics.get("eval_ce_loss")),
            "final_train_pcn_energy": safe_float(final_metrics.get("train_pcn_energy")),
            "final_eval_pcn_energy": safe_float(final_metrics.get("eval_pcn_energy")),
            "final_train_solved_rate": safe_float(final_metrics.get("train_grid_acc")),
            "final_eval_solved_rate": safe_float(final_metrics.get("eval_grid_acc")),
            "final_train_match_rate": safe_float(final_metrics.get("train_cell_acc")),
            "final_eval_match_rate": safe_float(final_metrics.get("eval_cell_acc")),
            "best_eval_loss": safe_float(best_eval_loss.get("eval_loss")),
            "best_eval_loss_epoch": best_eval_loss.get("epoch"),
            "best_eval_solved_rate": safe_float(summary.get("best_eval_grid_acc")),
            "best_eval_match_rate": safe_float(summary.get("best_eval_cell_acc")),
        }
        run["final_train_solved_pct"] = 100.0 * run["final_train_solved_rate"]
        run["final_eval_solved_pct"] = 100.0 * run["final_eval_solved_rate"]
        run["final_train_match_pct"] = 100.0 * run["final_train_match_rate"]
        run["final_eval_match_pct"] = 100.0 * run["final_eval_match_rate"]
        run["best_eval_solved_pct"] = 100.0 * run["best_eval_solved_rate"]
        run["best_eval_match_pct"] = 100.0 * run["best_eval_match_rate"]
        runs.append(run)

    return runs


def save_summary_files(results_dir: Path, runs: list) -> None:
    fieldnames = [
        "run_name",
        "timestamp",
        "epochs",
        "final_train_loss",
        "final_eval_loss",
        "final_train_ce_loss",
        "final_eval_ce_loss",
        "final_train_pcn_energy",
        "final_eval_pcn_energy",
        "final_train_solved_pct",
        "final_eval_solved_pct",
        "best_eval_solved_pct",
        "final_train_match_pct",
        "final_eval_match_pct",
        "best_eval_match_pct",
        "best_eval_loss",
        "best_eval_loss_epoch",
    ]

    with open(results_dir / "run_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in runs:
            writer.writerow({key: row.get(key) for key in fieldnames})

    with open(results_dir / "run_summary.json", "w") as handle:
        json.dump(runs, handle, indent=2)

    lines = []
    lines.append("# Run Summary")
    lines.append("")
    lines.append("| Run | Epochs | Final Eval Solved % | Best Eval Solved % | Final Eval Match % | Best Eval Match % | Final Eval Loss | Final Eval PCN Energy |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for row in runs:
        lines.append(
            "| "
            + f"{row['run_label']} | "
            + f"{row['epochs']} | "
            + f"{row['final_eval_solved_pct']:.2f} | "
            + f"{row['best_eval_solved_pct']:.2f} | "
            + f"{row['final_eval_match_pct']:.2f} | "
            + f"{row['best_eval_match_pct']:.2f} | "
            + f"{row['final_eval_loss']:.4f} | "
            + f"{row['final_eval_pcn_energy']:.4f} |"
        )

    with open(results_dir / "run_summary.md", "w") as handle:
        handle.write("\n".join(lines) + "\n")


def apply_plot_style(plt) -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "#faf7f2",
            "axes.facecolor": "#fffdf9",
            "axes.edgecolor": "#d8d2c6",
            "axes.labelcolor": "#2f2a24",
            "axes.titlecolor": "#1f1b16",
            "xtick.color": "#4a433b",
            "ytick.color": "#4a433b",
            "grid.color": "#ddd6ca",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
        }
    )


def plot_percentage_graph(results_dir: Path, runs: list, final_key: str, best_key: str, title: str, output_name: str) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        return

    apply_plot_style(plt)

    labels = [run["run_label"] for run in runs]
    x = list(range(len(runs)))
    final_values = [run[final_key] for run in runs]
    best_values = [run[best_key] for run in runs]

    fig, ax = plt.subplots(figsize=(max(9, len(runs) * 1.6), 5.8))

    ax.plot(
        x,
        final_values,
        color="#1f6f78",
        linewidth=2.4,
        marker="o",
        markersize=7,
        markerfacecolor="#fffdf9",
        markeredgewidth=2,
        label="Final Eval",
    )
    ax.plot(
        x,
        best_values,
        color="#d97b29",
        linewidth=2.0,
        marker="o",
        markersize=6,
        markerfacecolor="#fffdf9",
        markeredgewidth=1.8,
        label="Best Eval",
    )

    ax.set_title(title, pad=14)
    ax.set_xlabel("Run", labelpad=12)
    ax.set_ylabel("Percentage", labelpad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.65)
    ax.grid(False, axis="x")

    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)
    ax.spines["left"].set_color("#c9c1b4")
    ax.spines["bottom"].set_color("#c9c1b4")

    ax.legend(frameon=False, loc="upper left")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.27)
    fig.savefig(results_dir / output_name, dpi=180)
    plt.close(fig)


def plot_single_series_graph(results_dir: Path, runs: list, key: str, title: str, ylabel: str, output_name: str) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        return

    apply_plot_style(plt)

    labels = [run["run_label"] for run in runs]
    x = list(range(len(runs)))
    values = [run[key] for run in runs]

    fig, ax = plt.subplots(figsize=(max(9, len(runs) * 1.6), 5.8))
    ax.plot(
        x,
        values,
        color="#6f3cc3",
        linewidth=2.2,
        marker="o",
        markersize=7,
        markerfacecolor="#fffdf9",
        markeredgewidth=2,
    )
    ax.set_title(title, pad=14)
    ax.set_xlabel("Run", labelpad=12)
    ax.set_ylabel(ylabel, labelpad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(True, axis="y", alpha=0.65)
    ax.grid(False, axis="x")

    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)
    ax.spines["left"].set_color("#c9c1b4")
    ax.spines["bottom"].set_color("#c9c1b4")

    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.27)
    fig.savefig(results_dir / output_name, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = resolve_results_dir(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(results_dir)
    if not runs:
        print(f"No completed runs found in {results_dir}")
        return

    save_summary_files(results_dir, runs)

    matplotlib_available = maybe_import_matplotlib() is not None
    plot_percentage_graph(
        results_dir,
        runs,
        final_key="final_eval_match_pct",
        best_key="best_eval_match_pct",
        title="Evaluation Match Percentage Across Runs",
        output_name="eval_match_percentage_across_runs.png",
    )
    plot_percentage_graph(
        results_dir,
        runs,
        final_key="final_eval_solved_pct",
        best_key="best_eval_solved_pct",
        title="Evaluation Solved Percentage Across Runs",
        output_name="eval_solved_percentage_across_runs.png",
    )
    plot_single_series_graph(
        results_dir,
        runs,
        key="final_eval_loss",
        title="Final Evaluation Loss Across Runs",
        ylabel="Loss",
        output_name="eval_loss_across_runs.png",
    )
    if any(run["final_eval_pcn_energy"] != 0.0 for run in runs):
        plot_single_series_graph(
            results_dir,
            runs,
            key="final_eval_pcn_energy",
            title="Final Evaluation PCN Energy Across Runs",
            ylabel="Energy",
            output_name="eval_pcn_energy_across_runs.png",
        )

    if not matplotlib_available:
        print("matplotlib not installed; wrote summary files and skipped plot generation.")
        return

    print(f"Saved run analysis to {results_dir}")


if __name__ == "__main__":
    main()
