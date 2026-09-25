from __future__ import annotations

import os
import pathlib
from pathlib import Path

import torch


def torch_load_compatible(
    checkpoint_path: Path,
    map_location: torch.device | str | None = None,
    *,
    weights_only: bool = False,
):
    """Load torch checkpoint with cross-OS pathlib compatibility.

    Some checkpoints saved on Linux may pickle `pathlib.PosixPath`, which raises
    on Windows during unpickling, and vice versa. This loader retries with a
    temporary class alias to make loading robust across OSes.
    """

    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=weights_only)
    except NotImplementedError as exc:
        msg = str(exc)
        if "PosixPath" in msg and os.name == "nt":
            original = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            try:
                return torch.load(checkpoint_path, map_location=map_location, weights_only=weights_only)
            finally:
                pathlib.PosixPath = original
        if "WindowsPath" in msg and os.name != "nt":
            original = pathlib.WindowsPath
            pathlib.WindowsPath = pathlib.PosixPath
            try:
                return torch.load(checkpoint_path, map_location=map_location, weights_only=weights_only)
            finally:
                pathlib.WindowsPath = original
        raise
