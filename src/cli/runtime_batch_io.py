# src/cli/runtime_batch_io.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import io
from typing import Any

import torch


def serialize_torch_payload(payload: Any) -> bytes:
    """Serialize a payload into one byte blob for safer worker IPC."""
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def deserialize_torch_payload(payload: Any) -> Any:
    """Deserialize a bytes payload produced by worker IPC helpers."""
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        return payload

    raw = bytes(payload)
    try:
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(io.BytesIO(raw), map_location="cpu")


__all__ = ["deserialize_torch_payload", "serialize_torch_payload"]
