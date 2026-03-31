"""
Data loading for the first ARC baseline.

One dataset item is one ARC task.

For each task:
- task["train"] becomes the example set
- task["test"][0] becomes the query pair

We use a fixed example slot count Kmax.
If a task has fewer examples than Kmax, the remaining slots are zero padded.
example_slot_mask tells the model which example slots are real.
"""

import json
import os

import torch
from torch.utils.data import Dataset


DEFAULT_PAD_SIZE = 30
DEFAULT_KMAX = 10


def load_json(path: str) -> dict:
    with open(path, "r") as handle:
        data = json.load(handle)
    return data


def zero_grid(size: int) -> list:
    grid = []
    row_index = 0
    while row_index < size:
        row = []
        col_index = 0
        while col_index < size:
            row.append(0)
            col_index += 1
        grid.append(row)
        row_index += 1
    return grid


def grid_shape(grid: list) -> list:
    return [len(grid), len(grid[0])]


def validate_grid_size(grid: list, pad_size: int) -> None:
    height = len(grid)
    width = len(grid[0])
    if height > pad_size or width > pad_size:
        raise ValueError(
            f"Grid shape {(height, width)} exceeds pad_size={pad_size}"
        )


def pad_grid_and_mask(grid: list, pad_size: int) -> tuple:
    validate_grid_size(grid, pad_size)

    padded = []
    mask = []

    row_index = 0
    while row_index < pad_size:
        padded_row = []
        mask_row = []
        col_index = 0
        while col_index < pad_size:
            if row_index < len(grid) and col_index < len(grid[0]):
                padded_row.append(grid[row_index][col_index])
                mask_row.append(1)
            else:
                padded_row.append(0)
                mask_row.append(0)
            col_index += 1
        padded.append(padded_row)
        mask.append(mask_row)
        row_index += 1

    return padded, mask


def pair_to_tensors(pair: dict, pad_size: int) -> dict:
    input_grid = pair["input"]
    output_grid = pair["output"]

    padded_input, input_mask = pad_grid_and_mask(input_grid, pad_size)
    padded_output, output_mask = pad_grid_and_mask(output_grid, pad_size)

    result = {}
    result["input"] = torch.tensor(padded_input, dtype=torch.long)
    result["output"] = torch.tensor(padded_output, dtype=torch.long)
    result["input_mask"] = torch.tensor(input_mask, dtype=torch.float32)
    result["output_mask"] = torch.tensor(output_mask, dtype=torch.float32)
    result["input_shape"] = torch.tensor(grid_shape(input_grid), dtype=torch.long)
    result["output_shape"] = torch.tensor(grid_shape(output_grid), dtype=torch.long)
    return result


def empty_pair_tensors(pad_size: int) -> dict:
    zeros = zero_grid(pad_size)

    result = {}
    result["input"] = torch.tensor(zeros, dtype=torch.long)
    result["output"] = torch.tensor(zeros, dtype=torch.long)
    result["input_mask"] = torch.tensor(zeros, dtype=torch.float32)
    result["output_mask"] = torch.tensor(zeros, dtype=torch.float32)
    result["input_shape"] = torch.tensor([0, 0], dtype=torch.long)
    result["output_shape"] = torch.tensor([0, 0], dtype=torch.long)
    return result


def build_task_sample(task: dict, kmax: int, pad_size: int, task_id: str = "") -> dict:
    train_pairs = task["train"]
    query_pair = task["test"][0]

    example_inputs = []
    example_outputs = []
    example_input_masks = []
    example_output_masks = []
    example_input_shapes = []
    example_output_shapes = []
    example_slot_mask = []

    example_index = 0
    while example_index < kmax:
        if example_index < len(train_pairs):
            pair_tensors = pair_to_tensors(train_pairs[example_index], pad_size)
            example_slot_mask.append(1.0)
        else:
            pair_tensors = empty_pair_tensors(pad_size)
            example_slot_mask.append(0.0)

        example_inputs.append(pair_tensors["input"])
        example_outputs.append(pair_tensors["output"])
        example_input_masks.append(pair_tensors["input_mask"])
        example_output_masks.append(pair_tensors["output_mask"])
        example_input_shapes.append(pair_tensors["input_shape"])
        example_output_shapes.append(pair_tensors["output_shape"])
        example_index += 1

    query_tensors = pair_to_tensors(query_pair, pad_size)

    sample = {}
    sample["task_id"] = task.get("task_id", task_id)
    sample["num_examples"] = min(len(train_pairs), kmax)
    sample["example_inputs"] = torch.stack(example_inputs, dim=0)
    sample["example_outputs"] = torch.stack(example_outputs, dim=0)
    sample["example_input_masks"] = torch.stack(example_input_masks, dim=0)
    sample["example_output_masks"] = torch.stack(example_output_masks, dim=0)
    sample["example_input_shapes"] = torch.stack(example_input_shapes, dim=0)
    sample["example_output_shapes"] = torch.stack(example_output_shapes, dim=0)
    sample["query_input"] = query_tensors["input"]
    sample["query_output"] = query_tensors["output"]
    sample["query_input_mask"] = query_tensors["input_mask"]
    sample["query_output_mask"] = query_tensors["output_mask"]
    sample["query_input_shape"] = query_tensors["input_shape"]
    sample["query_output_shape"] = query_tensors["output_shape"]
    sample["example_slot_mask"] = torch.tensor(example_slot_mask, dtype=torch.float32)
    return sample


class ARCTaskDataset(Dataset):
    def __init__(
        self,
        data_root: str = "data",
        split: str = "train",
        kmax: int = DEFAULT_KMAX,
        pad_size: int = DEFAULT_PAD_SIZE,
    ) -> None:
        self.data_root = data_root
        self.split = split
        self.kmax = kmax
        self.pad_size = pad_size

        if split == "train":
            self.tasks_dir = os.path.join(data_root, "canonical_train", "tasks")
        elif split == "evaluation":
            self.tasks_dir = os.path.join(data_root, "canonical_evaluation", "tasks")
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.file_names = []
        for entry in os.listdir(self.tasks_dir):
            if entry.endswith(".json"):
                self.file_names.append(entry)
        self.file_names.sort()

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, index: int) -> dict:
        file_name = self.file_names[index]
        path = os.path.join(self.tasks_dir, file_name)
        task = load_json(path)
        task_id = file_name.replace(".json", "")
        sample = build_task_sample(task, self.kmax, self.pad_size, task_id=task_id)
        return sample


def arc_collate_fn(batch: list) -> dict:
    task_ids = []
    num_examples = []
    example_inputs = []
    example_outputs = []
    example_input_masks = []
    example_output_masks = []
    example_input_shapes = []
    example_output_shapes = []
    query_input = []
    query_output = []
    query_input_mask = []
    query_output_mask = []
    query_input_shape = []
    query_output_shape = []
    example_slot_mask = []

    batch_index = 0
    while batch_index < len(batch):
        sample = batch[batch_index]
        task_ids.append(sample["task_id"])
        num_examples.append(sample["num_examples"])
        example_inputs.append(sample["example_inputs"])
        example_outputs.append(sample["example_outputs"])
        example_input_masks.append(sample["example_input_masks"])
        example_output_masks.append(sample["example_output_masks"])
        example_input_shapes.append(sample["example_input_shapes"])
        example_output_shapes.append(sample["example_output_shapes"])
        query_input.append(sample["query_input"])
        query_output.append(sample["query_output"])
        query_input_mask.append(sample["query_input_mask"])
        query_output_mask.append(sample["query_output_mask"])
        query_input_shape.append(sample["query_input_shape"])
        query_output_shape.append(sample["query_output_shape"])
        example_slot_mask.append(sample["example_slot_mask"])
        batch_index += 1

    collated = {}
    collated["task_id"] = task_ids
    collated["num_examples"] = torch.tensor(num_examples, dtype=torch.long)
    collated["example_inputs"] = torch.stack(example_inputs, dim=0)
    collated["example_outputs"] = torch.stack(example_outputs, dim=0)
    collated["example_input_masks"] = torch.stack(example_input_masks, dim=0)
    collated["example_output_masks"] = torch.stack(example_output_masks, dim=0)
    collated["example_input_shapes"] = torch.stack(example_input_shapes, dim=0)
    collated["example_output_shapes"] = torch.stack(example_output_shapes, dim=0)
    collated["query_input"] = torch.stack(query_input, dim=0)
    collated["query_output"] = torch.stack(query_output, dim=0)
    collated["query_input_mask"] = torch.stack(query_input_mask, dim=0)
    collated["query_output_mask"] = torch.stack(query_output_mask, dim=0)
    collated["query_input_shape"] = torch.stack(query_input_shape, dim=0)
    collated["query_output_shape"] = torch.stack(query_output_shape, dim=0)
    collated["example_slot_mask"] = torch.stack(example_slot_mask, dim=0)
    return collated
