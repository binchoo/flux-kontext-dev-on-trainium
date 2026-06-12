"""Hardware backend abstraction for FLUX.1-Kontext training on AWS Trainium.

Centralizes all hardware/device-specific logic so the training script does not
call ``torch.cuda.*`` directly. Supports:

- ``cuda`` : NVIDIA GPU (original, unchanged behaviour)
- ``xla``  : AWS Trainium / Inferentia via ``torch-neuronx`` (torch_xla)
- ``mps`` / ``cpu`` : fallbacks (local dev / static checks)

Design goals:
1. The CUDA path stays byte-for-byte identical (every helper forwards to the
   exact ``torch.cuda.*`` call it replaced when the backend is cuda).
2. Importable without torch_xla (all torch_xla imports are lazy/guarded), so the
   module loads cleanly on a macOS / CPU-only machine for static checks.
3. Single source of truth for backend selection.

Backend selection order:
    1. ``FLUX_BACKEND`` env var (explicit override: cuda|xla|mps|cpu)
    2. torch_xla importable AND a Neuron runtime env var set
       (``NEURON_RT_VISIBLE_CORES`` / ``NEURON_RT_NUM_CORES``) -> ``xla``
    3. ``torch.cuda.is_available()`` -> ``cuda``
    4. ``torch.backends.mps.is_available()`` -> ``mps``
    5. ``cpu``
"""

from __future__ import annotations

import contextlib
import importlib.util
import os

import torch


CUDA = "cuda"
XLA = "xla"
MPS = "mps"
CPU = "cpu"

_VALID_BACKENDS = {CUDA, XLA, MPS, CPU}
_backend: str | None = None


def _xla_available() -> bool:
    return importlib.util.find_spec("torch_xla") is not None


def _detect_backend() -> str:
    override = os.environ.get("FLUX_BACKEND", "").strip().lower()
    if override:
        if override not in _VALID_BACKENDS:
            raise ValueError(
                f"Invalid FLUX_BACKEND={override!r}. Expected one of {sorted(_VALID_BACKENDS)}."
            )
        return override

    if _xla_available() and (
        os.environ.get("NEURON_RT_VISIBLE_CORES") is not None
        or os.environ.get("NEURON_RT_NUM_CORES") is not None
    ):
        return XLA

    if torch.cuda.is_available():
        return CUDA

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return MPS

    return CPU


def get_backend() -> str:
    global _backend
    if _backend is None:
        _backend = _detect_backend()
    return _backend


def set_backend(name: str) -> None:
    name = name.strip().lower()
    if name not in _VALID_BACKENDS:
        raise ValueError(f"Invalid backend {name!r}. Expected one of {sorted(_VALID_BACKENDS)}.")
    global _backend
    _backend = name


def is_xla() -> bool:
    return get_backend() == XLA


def is_cuda() -> bool:
    return get_backend() == CUDA


def _xm():
    """Lazily import torch_xla.core.xla_model. Only call when backend is xla."""
    import torch_xla.core.xla_model as xm  # noqa: PLC0415

    return xm


def device(index: int | None = None) -> torch.device:
    """Canonical training device for the active backend."""
    backend = get_backend()
    if backend == XLA:
        return _xm().xla_device()
    if backend == CUDA:
        return torch.device("cuda" if index is None else f"cuda:{index}")
    if backend == MPS:
        return torch.device("mps")
    return torch.device("cpu")


def is_available() -> bool:
    """Replacement for torch.cuda.is_available() in feature gates.

    Returns True on cuda/xla/mps (i.e. an accelerator is present), False on cpu.
    """
    return get_backend() in (CUDA, XLA, MPS)


def empty_cache() -> None:
    """Free cached device memory. No-op on XLA (runtime-managed)."""
    if get_backend() == CUDA:
        torch.cuda.empty_cache()


def synchronize() -> None:
    if get_backend() == CUDA:
        torch.cuda.synchronize()
    elif get_backend() == XLA:
        _xm().mark_step()


def mark_step() -> None:
    """Execute the accumulated XLA graph. No-op on every other backend.

    Critical XLA training boundary — call once per optimizer step.
    """
    if get_backend() == XLA:
        _xm().mark_step()


def manual_seed_all(seed: int) -> None:
    backend = get_backend()
    if backend == CUDA:
        torch.cuda.manual_seed_all(seed)
    elif backend == XLA:
        _xm().set_rng_state(seed)


def device_context(index_or_device=None):
    """Context manager analogous to ``torch.cuda.device(idx)`` (null on non-cuda)."""
    if get_backend() == CUDA and index_or_device is not None:
        return torch.cuda.device(index_or_device)
    return contextlib.nullcontext()


def allow_tf32() -> None:
    """Enable TF32 matmul on CUDA. No-op elsewhere (Trainium handles precision via bf16)."""
    if get_backend() == CUDA:
        torch.backends.cuda.matmul.allow_tf32 = True


def no_grad_or_inference():
    """Return no_grad() on XLA, inference_mode() on CUDA.

    XLA's lazy-graph tracer rejects "inference tensors" (no version_counter)
    produced by inference_mode; no_grad is equivalent for frozen forward passes
    and yields trackable tensors. (This repo uses torch.no_grad() already, so
    this is provided for completeness / parity with the Qwen port.)
    """
    return torch.no_grad() if is_xla() else torch.inference_mode()
