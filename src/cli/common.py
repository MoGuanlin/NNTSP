# src/cli/common.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import random
from typing import Any

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_device(device_arg: str) -> torch.device:
    if str(device_arg).startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def move_data_tensors_to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(device)
            elif value is not None:
                move_data_tensors_to_device(value, device)
        return data

    if isinstance(data, list):
        for idx, value in enumerate(data):
            if isinstance(value, torch.Tensor):
                data[idx] = value.to(device)
            elif value is not None:
                move_data_tensors_to_device(value, device)
        return data

    skip = {"num_faces"}
    for key in dir(data):
        if key.startswith("_") or key in skip:
            continue
        try:
            value = getattr(data, key, None)
        except Exception:
            continue
        if isinstance(value, torch.Tensor):
            try:
                setattr(data, key, value.to(device))
            except Exception:
                pass
        elif value is not None and (hasattr(value, "__dict__") or isinstance(value, (dict, list))):
            move_data_tensors_to_device(value, device)
    return data


def load_dataset(path: str):
    from src.dataprep.dataset import smart_load_dataset

    return smart_load_dataset(path)


def log_progress(prefix: str, message: str) -> None:
    print(f"{prefix} {message}", flush=True)


__all__ = [
    "load_dataset",
    "log_progress",
    "move_data_tensors_to_device",
    "parse_bool_arg",
    "resolve_device",
    "set_seed",
]
