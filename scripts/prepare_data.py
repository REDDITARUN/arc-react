#!/usr/bin/env python3
"""
Prepare ARC-AGI-1 and optional ARC-AGI-2 data.

This keeps the original canonical data, padded inspection data, and an optional
TRM-style augmentation split for training. The augmentation is intentionally
label-preserving and whole-task:
- dihedral grid transforms
- color remapping that keeps black fixed
- optional translation offsets stored as metadata and applied during padding
"""

import argparse
import json
import os
import random
import shutil
import subprocess

from collections import Counter
from pathlib import Path


CONFIG = {
    "arc1_repo_url": "https://github.com/fchollet/ARC-AGI",
    "arc1_repo_dir_name": "ARC-AGI",
    "arc2_repo_url": "https://github.com/arcprize/ARC-AGI-2.git",
    "arc2_repo_dir_name": "ARC-AGI-2",
    "arc1_output_dir_name": "data_arc_agi_1",
    "arc2_output_dir_name": "data_arc_agi_2",
    "canonical_train_dir_name": "canonical_train",
    "canonical_evaluation_dir_name": "canonical_evaluation",
    "augmented_train_dir_name": "augmented_train",
    "padded_dir_name": "padded_processed",
    "reports_dir_name": "reports",
    "manifests_dir_name": "manifests",
    "pad_size": 30,
    "cleanup_clone": True,
    "sample_preview_tasks": 2,
    "num_aug": 1000,
    "include_arc_agi_2": False,
    "enable_translation_aug": True,
    "seed": 42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ARC-AGI data with optional TRM-style augmentation.")
    parser.add_argument("--workspace", default=".", help="Workspace folder that contains repos and outputs.")
    parser.add_argument(
        "--keep-repo-clones",
        action="store_true",
        help="Keep ARC-AGI and ARC-AGI-2 clone directories after preparation (default: remove them).",
    )
    parser.add_argument("--include-arc-agi-2", action="store_true", help="Also prepare ARC-AGI-2 into a separate data root.")
    parser.add_argument("--num-aug", type=int, default=1000, help="How many augmented copies to create per training task.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic augmentation.")
    parser.add_argument("--disable-translation-aug", action="store_true", help="Disable translation offsets for augmented tasks.")
    return parser.parse_args()


def make_runtime_config(args: argparse.Namespace) -> dict:
    config = dict(CONFIG)
    config["cleanup_clone"] = not args.keep_repo_clones
    config["include_arc_agi_2"] = args.include_arc_agi_2
    config["num_aug"] = args.num_aug
    config["seed"] = args.seed
    config["enable_translation_aug"] = not args.disable_translation_aug
    return config


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: object) -> None:
    ensure_directory(path.parent)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def save_text(path: Path, text: str) -> None:
    ensure_directory(path.parent)
    with open(path, "w") as handle:
        handle.write(text)


def run_command(command: list, cwd: Path = None) -> None:
    print("Running command:")
    print("  " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def build_table(headers: list, rows: list, title: str = "") -> str:
    widths = [len(str(value)) for value in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = []
    if title:
        lines.append(title)
    lines.append(border)
    lines.append("|" + "|".join(f" {str(value).ljust(widths[index])} " for index, value in enumerate(headers)) + "|")
    lines.append(border)
    for row in rows:
        lines.append("|" + "|".join(f" {str(value).ljust(widths[index])} " for index, value in enumerate(row)) + "|")
    lines.append(border)
    return "\n".join(lines)


def print_section(title: str) -> None:
    print("")
    print(title)
    print("=" * len(title))


def print_config(config: dict) -> None:
    rows = [[key, config[key]] for key in sorted(config)]
    print(build_table(["key", "value"], rows, "Pipeline Config"))


def clone_or_reuse_repo(workspace: Path, repo_url: str, repo_dir_name: str) -> Path:
    repo_path = workspace / repo_dir_name
    if repo_path.exists():
        print(f"Reusing existing repo: {repo_path}")
        return repo_path
    print(f"Cloning repo: {repo_url}")
    run_command(["git", "clone", repo_url, repo_dir_name], cwd=workspace)
    return repo_path


def load_json(path: Path) -> object:
    with open(path, "r") as handle:
        return json.load(handle)


def load_split_tasks(arc_repo: Path, split_name: str) -> dict:
    split_dir = arc_repo / "data" / split_name
    return {
        file_name.replace(".json", ""): load_json(split_dir / file_name)
        for file_name in sorted(entry for entry in os.listdir(split_dir) if entry.endswith(".json"))
    }


def grid_shape(grid: list) -> list:
    return [len(grid), len(grid[0])]


def grid_area(grid: list) -> int:
    return len(grid) * len(grid[0])


def size_relation(pair: dict) -> str:
    if grid_shape(pair["input"]) == grid_shape(pair["output"]):
        return "same"
    return "changed"


def count_colors(grid: list) -> int:
    colors = set()
    for row in grid:
        for value in row:
            colors.add(value)
    return len(colors)


def count_total_queries(tasks: dict) -> int:
    return sum(len(task["test"]) for task in tasks.values())


def task_record(task_id: str, task_data: dict, source: str) -> dict:
    return {
        "task_id": task_id,
        "base_task_id": task_id,
        "source": source,
        "train": task_data["train"],
        "test": task_data["test"],
    }


def save_task_split(tasks: dict, output_dir: Path, source: str) -> None:
    reset_directory(output_dir)
    for task_id in sorted(tasks):
        save_json(output_dir / f"{task_id}.json", task_record(task_id, tasks[task_id], source))


def pad_grid(grid: list, pad_size: int) -> tuple:
    padded = []
    mask = []
    for row_index in range(pad_size):
        padded_row = []
        mask_row = []
        for col_index in range(pad_size):
            if row_index < len(grid) and col_index < len(grid[0]):
                padded_row.append(grid[row_index][col_index])
                mask_row.append(1)
            else:
                padded_row.append(0)
                mask_row.append(0)
        padded.append(padded_row)
        mask.append(mask_row)
    return padded, mask


def zero_grid(size: int) -> list:
    return [[0 for _ in range(size)] for _ in range(size)]


def padded_pair(pair: dict, pair_index: int, pad_size: int) -> dict:
    input_padded, input_mask = pad_grid(pair["input"], pad_size)
    if "output" in pair:
        output_padded, output_mask = pad_grid(pair["output"], pad_size)
        output_shape = grid_shape(pair["output"])
    else:
        output_padded = zero_grid(pad_size)
        output_mask = zero_grid(pad_size)
        output_shape = None
    return {
        "pair_index": pair_index,
        "input_shape": grid_shape(pair["input"]),
        "output_shape": output_shape,
        "input": input_padded,
        "output": output_padded,
        "input_mask": input_mask,
        "output_mask": output_mask,
    }


def padded_task_record(task_id: str, task_data: dict, source: str, pad_size: int) -> dict:
    record = {"task_id": task_id, "source": source, "train": [], "test": []}
    for split_name in ["train", "test"]:
        for pair_index, pair in enumerate(task_data[split_name]):
            record[split_name].append(padded_pair(pair, pair_index, pad_size))
    return record


def save_padded_split(tasks: dict, output_dir: Path, source: str, pad_size: int) -> None:
    reset_directory(output_dir)
    for task_id in sorted(tasks):
        save_json(output_dir / f"{task_id}.json", padded_task_record(task_id, tasks[task_id], source, pad_size))


def limited_counter_rows(counter: Counter, limit: int = 12) -> list:
    return [[key, counter[key]] for key in sorted(counter, key=lambda value: str(value))[:limit]]


def analyze_training_tasks(tasks: dict) -> tuple:
    examples_per_task = Counter()
    queries_per_task = Counter()
    input_size_dist = Counter()
    output_size_dist = Counter()
    size_relation_dist = Counter()
    input_color_dist = Counter()

    for task_id in sorted(tasks):
        task = tasks[task_id]
        examples_per_task[len(task["train"])] += 1
        queries_per_task[len(task["test"])] += 1
        for pair in task["train"]:
            input_shape = grid_shape(pair["input"])
            output_shape = grid_shape(pair["output"])
            input_size_dist[f"{input_shape[0]}x{input_shape[1]}"] += 1
            output_size_dist[f"{output_shape[0]}x{output_shape[1]}"] += 1
            size_relation_dist[size_relation(pair)] += 1
            input_color_dist[count_colors(pair["input"])] += 1

    summary_rows = [
        ["num training tasks", len(tasks)],
        ["num training queries", count_total_queries(tasks)],
        ["distinct train example counts", len(examples_per_task)],
        ["distinct query counts", len(queries_per_task)],
        ["distinct input sizes", len(input_size_dist)],
        ["distinct output sizes", len(output_size_dist)],
        ["distinct input color counts", len(input_color_dist)],
    ]

    sections = [
        build_table(["metric", "value"], summary_rows, "Original Analysis Summary"),
        build_table(["value", "count"], limited_counter_rows(examples_per_task), "Train Examples Per Task"),
        build_table(["value", "count"], limited_counter_rows(queries_per_task), "Queries Per Task"),
        build_table(["value", "count"], limited_counter_rows(size_relation_dist), "Input/Output Same Vs Changed"),
        build_table(["value", "count"], limited_counter_rows(input_size_dist), "Top Input Grid Sizes"),
        build_table(["value", "count"], limited_counter_rows(output_size_dist), "Top Output Grid Sizes"),
        build_table(["value", "count"], limited_counter_rows(input_color_dist), "Input Color Count Distribution"),
    ]

    summary = {
        "num_training_tasks": len(tasks),
        "num_training_queries": count_total_queries(tasks),
        "examples_per_task": dict(examples_per_task),
        "queries_per_task": dict(queries_per_task),
        "input_size_dist": dict(input_size_dist),
        "output_size_dist": dict(output_size_dist),
        "size_relation_dist": dict(size_relation_dist),
        "input_color_dist": dict(input_color_dist),
    }
    return summary, "\n\n".join(sections)


def analyze_padding(train_tasks: dict, evaluation_tasks: dict, pad_size: int) -> tuple:
    total_real_cells = 0
    total_padded_cells = 0
    for split_tasks in [train_tasks, evaluation_tasks]:
        for task in split_tasks.values():
            for split_name in ["train", "test"]:
                for pair in task[split_name]:
                    input_area = grid_area(pair["input"])
                    total_real_cells += input_area
                    total_padded_cells += (pad_size * pad_size) - input_area
                    if "output" in pair:
                        output_area = grid_area(pair["output"])
                        total_real_cells += output_area
                        total_padded_cells += (pad_size * pad_size) - output_area

    total_cells = total_real_cells + total_padded_cells
    padding_ratio = total_padded_cells / total_cells if total_cells > 0 else 0.0
    rows = [
        ["total train tasks", len(train_tasks)],
        ["total train queries", count_total_queries(train_tasks)],
        ["total evaluation tasks", len(evaluation_tasks)],
        ["total evaluation queries", count_total_queries(evaluation_tasks)],
        ["real cells", total_real_cells],
        ["padded cells", total_padded_cells],
        ["padding ratio", f"{padding_ratio:.4f}"],
    ]
    summary = {
        "total_train_tasks": len(train_tasks),
        "total_train_queries": count_total_queries(train_tasks),
        "total_evaluation_tasks": len(evaluation_tasks),
        "total_evaluation_queries": count_total_queries(evaluation_tasks),
        "real_cells": total_real_cells,
        "padded_cells": total_padded_cells,
        "padding_ratio": padding_ratio,
    }
    return summary, build_table(["metric", "value"], rows, "Padded Analysis Summary")


def preview_examples(train_tasks: dict, evaluation_tasks: dict, pad_size: int, sample_count: int) -> str:
    train_ids = sorted(train_tasks)
    eval_ids = sorted(evaluation_tasks)
    rows = []

    for task_id in train_ids[:sample_count]:
        raw_pair = train_tasks[task_id]["train"][0]
        padded = padded_pair(raw_pair, 0, pad_size)
        rows.append(["train_raw", task_id, f"{len(raw_pair['input'])}x{len(raw_pair['input'][0])}", f"{len(raw_pair['output'])}x{len(raw_pair['output'][0])}", "-"])
        rows.append([
            "train_padded",
            task_id,
            f"{padded['input_shape'][0]}x{padded['input_shape'][1]} -> {pad_size}x{pad_size}",
            f"{padded['output_shape'][0]}x{padded['output_shape'][1]} -> {pad_size}x{pad_size}",
            f"in_mask={sum(sum(row) for row in padded['input_mask'])}, out_mask={sum(sum(row) for row in padded['output_mask'])}",
        ])

    for task_id in eval_ids[:sample_count]:
        raw_pair = evaluation_tasks[task_id]["test"][0]
        padded = padded_pair(raw_pair, 0, pad_size)
        output_shape = "unknown"
        if "output" in raw_pair:
            output_shape = f"{len(raw_pair['output'])}x{len(raw_pair['output'][0])}"
        padded_output = "unknown -> 30x30_zero_mask"
        if padded["output_shape"] is not None:
            padded_output = f"{padded['output_shape'][0]}x{padded['output_shape'][1]} -> {pad_size}x{pad_size}"
        rows.append(["eval_raw", task_id, f"{len(raw_pair['input'])}x{len(raw_pair['input'][0])}", output_shape, "-"])
        rows.append([
            "eval_padded",
            task_id,
            f"{padded['input_shape'][0]}x{padded['input_shape'][1]} -> {pad_size}x{pad_size}",
            padded_output,
            f"in_mask={sum(sum(row) for row in padded['input_mask'])}, out_mask={sum(sum(row) for row in padded['output_mask'])}",
        ])

    return build_table(["kind", "task_id", "input", "output", "note"], rows, "Before And After Padded Examples")


def dihedral_transform(grid: list, tid: int) -> list:
    if tid == 0:
        return [row[:] for row in grid]
    if tid == 1:
        return [list(row) for row in zip(*grid[::-1])]
    if tid == 2:
        return [row[::-1] for row in grid[::-1]]
    if tid == 3:
        return [list(row) for row in zip(*grid)][::-1]
    if tid == 4:
        return [row[::-1] for row in grid]
    if tid == 5:
        return grid[::-1]
    if tid == 6:
        return [list(row) for row in zip(*grid)]
    if tid == 7:
        return [row[::-1] for row in [list(row) for row in zip(*grid)]][::-1]
    raise ValueError(f"Unsupported dihedral transform id: {tid}")


def color_permutation(rng: random.Random) -> list:
    values = list(range(1, 10))
    rng.shuffle(values)
    mapping = [0] * 10
    mapping[0] = 0
    for color in range(1, 10):
        mapping[color] = values[color - 1]
    return mapping


def remap_grid_colors(grid: list, mapping: list) -> list:
    return [[mapping[value] for value in row] for row in grid]


def transform_grid(grid: list, tid: int, mapping: list) -> list:
    return dihedral_transform(remap_grid_colors(grid, mapping), tid)


def choose_translation_offset(pair: dict, pad_size: int, rng: random.Random) -> list:
    input_shape = grid_shape(pair["input"])
    max_height = input_shape[0]
    max_width = input_shape[1]
    if "output" in pair:
        output_shape = grid_shape(pair["output"])
        max_height = max(max_height, output_shape[0])
        max_width = max(max_width, output_shape[1])
    top = rng.randint(0, pad_size - max_height)
    left = rng.randint(0, pad_size - max_width)
    return [top, left]


def transform_pair(pair: dict, tid: int, mapping: list, pad_size: int, rng: random.Random, enable_translation_aug: bool) -> dict:
    transformed = {"input": transform_grid(pair["input"], tid, mapping)}
    if "output" in pair:
        transformed["output"] = transform_grid(pair["output"], tid, mapping)
    if enable_translation_aug:
        transformed["pad_offset"] = choose_translation_offset(transformed, pad_size, rng)
    return transformed


def augment_task_variants(task_id: str, task_data: dict, num_aug: int, pad_size: int, seed: int, enable_translation_aug: bool) -> list:
    if num_aug <= 0:
        return []

    task_rng = random.Random(seed)
    seen_specs = set()
    variants = []
    max_attempts = max(32, num_aug * 8)
    attempts = 0

    while len(variants) < num_aug and attempts < max_attempts:
        attempts += 1
        tid = task_rng.randint(0, 7)
        mapping = color_permutation(task_rng)
        if tid == 0 and mapping == list(range(10)) and not enable_translation_aug:
            continue

        variant_seed = task_rng.randint(0, 10**9)
        spec_key = (tid, tuple(mapping), variant_seed if enable_translation_aug else 0)
        if spec_key in seen_specs:
            continue
        seen_specs.add(spec_key)

        rng = random.Random(variant_seed)
        variant = {
            "task_id": f"{task_id}__aug{len(variants):03d}",
            "base_task_id": task_id,
            "source": "trm_style_augmentation",
            "augmentation": {
                "dihedral_id": tid,
                "color_mapping": mapping,
                "translation_seed": variant_seed if enable_translation_aug else None,
                "translation_enabled": enable_translation_aug,
            },
            "train": [transform_pair(pair, tid, mapping, pad_size, rng, enable_translation_aug) for pair in task_data["train"]],
            "test": [transform_pair(pair, tid, mapping, pad_size, rng, enable_translation_aug) for pair in task_data["test"]],
        }
        variants.append(variant)

    return variants


def build_augmented_train_split(train_tasks: dict, output_dir: Path, num_aug: int, pad_size: int, seed: int, enable_translation_aug: bool) -> tuple:
    reset_directory(output_dir)
    total_original = 0
    total_augmented = 0
    preview_rows = []

    for task_offset, task_id in enumerate(sorted(train_tasks)):
        original_record = task_record(task_id, train_tasks[task_id], "original_train")
        save_json(output_dir / f"{task_id}__orig.json", original_record)
        total_original += 1

        variants = augment_task_variants(
            task_id=task_id,
            task_data=train_tasks[task_id],
            num_aug=num_aug,
            pad_size=pad_size,
            seed=seed + task_offset * 1009,
            enable_translation_aug=enable_translation_aug,
        )
        for variant in variants:
            save_json(output_dir / f"{variant['task_id']}.json", variant)
            total_augmented += 1

        if len(preview_rows) < 4:
            preview_rows.append([task_id, "orig", len(train_tasks[task_id]["train"]), len(train_tasks[task_id]["test"]), "-", "-", "-"])
            for variant in variants[:2]:
                aug = variant["augmentation"]
                preview_rows.append([
                    task_id,
                    variant["task_id"].split("__")[-1],
                    len(variant["train"]),
                    len(variant["test"]),
                    aug["dihedral_id"],
                    "".join(str(value) for value in aug["color_mapping"]),
                    aug["translation_enabled"],
                ])

    summary = {
        "original_train_tasks": len(train_tasks),
        "original_train_queries": count_total_queries(train_tasks),
        "num_aug_per_task": num_aug,
        "augmented_original_files": total_original,
        "augmented_variant_files": total_augmented,
        "total_augmented_train_files": total_original + total_augmented,
        "increase_vs_original_tasks": (total_original + total_augmented) / max(len(train_tasks), 1),
        "translation_enabled": enable_translation_aug,
    }

    preview_text = build_table(
        ["task_id", "kind", "train_pairs", "test_pairs", "dihedral_id", "color_map", "translation"],
        preview_rows,
        "Augmented Train Preview",
    )
    return summary, preview_text


def build_augmentation_report(summary: dict) -> str:
    rows = [
        ["original train tasks", summary["original_train_tasks"]],
        ["original train queries", summary["original_train_queries"]],
        ["num aug per task", summary["num_aug_per_task"]],
        ["original files kept", summary["augmented_original_files"]],
        ["augmented variant files", summary["augmented_variant_files"]],
        ["total augmented train files", summary["total_augmented_train_files"]],
        ["increase vs original tasks", f"{summary['increase_vs_original_tasks']:.2f}x"],
        ["translation enabled", summary["translation_enabled"]],
    ]
    return build_table(["metric", "value"], rows, "Augmentation Summary")


def git_commit_hash(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def dataset_paths(output_root: Path, config: dict) -> dict:
    paths = {
        "output_root": output_root,
        "canonical_train_root": output_root / config["canonical_train_dir_name"],
        "canonical_evaluation_root": output_root / config["canonical_evaluation_dir_name"],
        "augmented_train_root": output_root / config["augmented_train_dir_name"],
        "padded_root": output_root / config["padded_dir_name"],
        "reports_root": output_root / config["reports_dir_name"],
        "manifests_root": output_root / config["manifests_dir_name"],
    }
    paths["canonical_train_tasks"] = paths["canonical_train_root"] / "tasks"
    paths["canonical_evaluation_tasks"] = paths["canonical_evaluation_root"] / "tasks"
    paths["augmented_train_tasks"] = paths["augmented_train_root"] / "tasks"
    paths["padded_train_tasks"] = paths["padded_root"] / "train"
    paths["padded_evaluation_tasks"] = paths["padded_root"] / "evaluation"
    return paths


def ensure_dataset_layout(paths: dict) -> None:
    for key in [
        "output_root",
        "canonical_train_root",
        "canonical_evaluation_root",
        "augmented_train_root",
        "padded_root",
        "reports_root",
        "manifests_root",
        "canonical_train_tasks",
        "canonical_evaluation_tasks",
        "augmented_train_tasks",
        "padded_train_tasks",
        "padded_evaluation_tasks",
    ]:
        ensure_directory(paths[key])


def build_manifest(config: dict, repo_path: Path, dataset_name: str) -> dict:
    return {
        "dataset_name": dataset_name,
        "config": config,
        "repo_path": str(repo_path),
        "repo_commit": git_commit_hash(repo_path),
    }


def safe_to_cleanup(paths: dict) -> bool:
    required_files = [
        paths["manifests_root"] / "run_manifest.json",
        paths["manifests_root"] / "original_analysis_summary.json",
        paths["manifests_root"] / "padded_analysis_summary.json",
        paths["reports_root"] / "original_analysis.txt",
        paths["reports_root"] / "padded_analysis.txt",
    ]
    for file_path in required_files:
        if not file_path.exists() or file_path.stat().st_size == 0:
            return False
    return True


def prepare_single_dataset(dataset_name: str, repo_path: Path, output_root: Path, config: dict) -> None:
    paths = dataset_paths(output_root, config)
    ensure_dataset_layout(paths)

    print_section(f"Loading {dataset_name}")
    train_tasks = load_split_tasks(repo_path, "training")
    evaluation_tasks = load_split_tasks(repo_path, "evaluation")

    print_section(f"Saving Canonical Data ({dataset_name})")
    save_task_split(train_tasks, paths["canonical_train_tasks"], "original_train")
    save_task_split(evaluation_tasks, paths["canonical_evaluation_tasks"], "original_evaluation")

    print_section(f"Saving Padded Data ({dataset_name})")
    save_padded_split(train_tasks, paths["padded_train_tasks"], "padded_train", config["pad_size"])
    save_padded_split(evaluation_tasks, paths["padded_evaluation_tasks"], "padded_evaluation", config["pad_size"])

    print_section(f"Original Analysis ({dataset_name})")
    original_summary, original_text = analyze_training_tasks(train_tasks)
    save_json(paths["manifests_root"] / "original_analysis_summary.json", original_summary)
    save_text(paths["reports_root"] / "original_analysis.txt", original_text)
    print(original_text)

    print_section(f"Padded Analysis ({dataset_name})")
    padded_summary, padded_text = analyze_padding(train_tasks, evaluation_tasks, config["pad_size"])
    save_json(paths["manifests_root"] / "padded_analysis_summary.json", padded_summary)
    save_text(paths["reports_root"] / "padded_analysis.txt", padded_text)
    print(padded_text)

    print_section(f"Sample Preview ({dataset_name})")
    preview_text = preview_examples(train_tasks, evaluation_tasks, config["pad_size"], config["sample_preview_tasks"])
    save_text(paths["reports_root"] / "sample_preview.txt", preview_text)
    print(preview_text)

    manifest = build_manifest(config, repo_path, dataset_name)
    save_json(paths["manifests_root"] / "run_manifest.json", manifest)

    if config["num_aug"] > 0:
        print_section(f"Building Augmented Train ({dataset_name})")
        aug_summary, aug_preview = build_augmented_train_split(
            train_tasks=train_tasks,
            output_dir=paths["augmented_train_tasks"],
            num_aug=config["num_aug"],
            pad_size=config["pad_size"],
            seed=config["seed"],
            enable_translation_aug=config["enable_translation_aug"],
        )
        aug_report = build_augmentation_report(aug_summary)
        save_json(paths["manifests_root"] / "augmentation_summary.json", aug_summary)
        save_text(paths["reports_root"] / "augmentation_summary.txt", aug_report)
        save_text(paths["reports_root"] / "augmentation_preview.txt", aug_preview)
        print(aug_report)
        print("")
        print(aug_preview)
    else:
        reset_directory(paths["augmented_train_tasks"])

    print_section(f"Done ({dataset_name})")
    print(f"Outputs written to: {paths['output_root']}")


def main() -> None:
    args = parse_args()
    config = make_runtime_config(args)
    workspace = Path(args.workspace).resolve()

    print_config(config)

    arc1_repo = clone_or_reuse_repo(workspace, config["arc1_repo_url"], config["arc1_repo_dir_name"])
    prepare_single_dataset("ARC-AGI-1", arc1_repo, workspace / config["arc1_output_dir_name"], config)

    if config["include_arc_agi_2"]:
        arc2_repo = clone_or_reuse_repo(workspace, config["arc2_repo_url"], config["arc2_repo_dir_name"])
        prepare_single_dataset("ARC-AGI-2", arc2_repo, workspace / config["arc2_output_dir_name"], config)

    if config["cleanup_clone"]:
        cleanup_candidates = [arc1_repo]
        if config["include_arc_agi_2"]:
            cleanup_candidates.append(workspace / config["arc2_repo_dir_name"])
        for repo_path in cleanup_candidates:
            output_root = workspace / (config["arc1_output_dir_name"] if repo_path.name == config["arc1_repo_dir_name"] else config["arc2_output_dir_name"])
            if safe_to_cleanup(dataset_paths(output_root, config)):
                print(f"Deleting repo clone: {repo_path}")
                shutil.rmtree(repo_path)


if __name__ == "__main__":
    main()
