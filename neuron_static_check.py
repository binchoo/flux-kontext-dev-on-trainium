#!/usr/bin/env python3
"""Static verification harness for the FLUX.1-Kontext Neuron (XLA) port.

Verifies the port WITHOUT Trainium hardware. Two tiers:

  Tier A (no torch) — always runs:
    - guard-coverage lint: train.py must not call torch.cuda.* directly (route
      through neuron_backend), with a small allowlist for legit feature gates.
    - artifact presence + parse (setup/run scripts, requirements, backend modules).
    - requirements-neuron.txt must exclude CUDA-only packages.
    - key port edits present (mark_step, build_accelerator, xla branches).

  Tier B (requires torch) — runs when importable:
    - backend selection: FLUX_BACKEND override + helper no-op/forward behaviour.
    - simulated XLA backend: stub torch_xla, assert is_xla()/device()/mark_step().

Exit 0 = all available checks passed. Tier B skipped (not failed) without torch.

Usage:  python neuron_static_check.py
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent

# Neuron-critical edited files.
CRITICAL_FILES = ["train.py", "neuron_backend.py", "neuron_accelerate.py"]

# Legit CUDA references in train.py the lint must NOT flag (feature gates, comments).
ALLOWED_CUDA_LINE_PATTERNS = [
    r"^\s*#",                                   # comments
    r"torch\.cuda\.is_available\(\)\s+or\s+torch\.backends\.mps",  # fp16-accelerator gate (has xla branch)
]

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

_failures: list[str] = []
_passes = 0


def _ok(msg: str) -> None:
    global _passes
    _passes += 1
    print(f"[{PASS}] {msg}")


def _bad(msg: str) -> None:
    _failures.append(msg)
    print(f"[{FAIL}] {msg}")


def _skip(msg: str) -> None:
    print(f"[{SKIP}] {msg}")


# --------------------------------------------------------------------------- #
# Tier A — no torch required
# --------------------------------------------------------------------------- #


def check_guard_coverage() -> None:
    bad_call = re.compile(r"torch\.cuda\.(empty_cache|synchronize|manual_seed_all)\b")
    allowed = [re.compile(p) for p in ALLOWED_CUDA_LINE_PATTERNS]
    offenders = []
    # neuron_backend.py is the abstraction layer — it INTENTIONALLY holds the
    # real torch.cuda.* calls (guarded by `if get_backend() == CUDA`). Lint only
    # the consumer files.
    for rel in [f for f in CRITICAL_FILES if f != "neuron_backend.py"]:
        path = REPO / rel
        if not path.exists():
            _bad(f"guard-coverage: missing file {rel}")
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if bad_call.search(code) and not any(a.search(line) for a in allowed):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    if offenders:
        _bad("guard-coverage: bare torch.cuda.* found (should route through backend):\n  "
             + "\n  ".join(offenders))
    else:
        _ok("guard-coverage: no bare torch.cuda.* mutation calls outside backend")


def check_port_edits_present() -> None:
    train = (REPO / "train.py").read_text()
    checks = {
        "import neuron_backend": "import neuron_backend as backend" in train,
        "build_accelerator wired": "build_accelerator(" in train,
        "mark_step inserted": "backend.mark_step()" in train,
        "get_sigmas xla branch": "if backend.is_xla():" in train and "argmax(dim=1)" in train,
        "8bit adam xla guard": "use_8bit_adam and backend.is_xla()" in train,
        "autocast device backend-aware": 'autocast_device = "xla"' in train,
        "validation skipped on xla": "(not backend.is_xla()) and global_step" in train,
    }
    missing = [k for k, v in checks.items() if not v]
    if missing:
        _bad(f"port-edits: missing expected edits in train.py: {missing}")
    else:
        _ok("port-edits: all expected XLA branches present in train.py")


def check_artifacts() -> None:
    artifacts = [
        "neuron_backend.py", "neuron_accelerate.py",
        "setup_neuron.sh", "run_neuron.sh", "requirements-neuron.txt",
        "neuron_static_check.py",
    ]
    for rel in artifacts:
        if (REPO / rel).exists():
            _ok(f"artifact present: {rel}")
        else:
            _bad(f"artifact missing: {rel}")


def check_no_cuda_only_in_reqs() -> None:
    text = (REPO / "requirements-neuron.txt").read_text()
    forbidden = ["bitsandbytes", "flash-attn", "flash_attn", "transformer_engine"]
    hits = [pkg for pkg in forbidden
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#") and pkg in line]
    if hits:
        _bad(f"requirements-neuron.txt lists CUDA-only packages: {sorted(set(hits))}")
    else:
        _ok("requirements-neuron.txt excludes CUDA-only packages")


def check_reqs_excludes_neuron_owned() -> None:
    """torch/transformers/accelerate must NOT be pinned here (version-owned by
    optimum-neuron). diffusers/peft/torchvision ARE app-required and listed here,
    but installed under a constraints lock so they cannot change torch."""
    text = (REPO / "requirements-neuron.txt").read_text()
    # Only the version-pinned core. diffusers/peft/torchvision are app deps the
    # Neuron stack does NOT provide, so they belong here (constraints-protected).
    owned = ["torch", "transformers", "accelerate", "tokenizers", "huggingface_hub", "safetensors", "numpy"]
    hits = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        name = re.split(r"[=<>~ ]", s)[0]
        if name in owned:
            hits.append(name)
    if hits:
        _bad(f"requirements-neuron.txt re-lists Neuron-owned packages (would override): {hits}")
    else:
        _ok("requirements-neuron.txt leaves the version-pinned Neuron core to optimum-neuron")


def check_reqs_has_core_model_libs() -> None:
    """train.py needs diffusers (+ torchvision); optimum-neuron does NOT pull them.
    They must be present in requirements-neuron.txt or the run fails at import."""
    text = (REPO / "requirements-neuron.txt").read_text()
    listed = {re.split(r"[=<>~ ]", l.strip())[0]
              for l in text.splitlines() if l.strip() and not l.strip().startswith("#")}
    required = ["diffusers", "torchvision", "peft"]
    missing = [p for p in required if p not in listed]
    if missing:
        _bad(f"requirements-neuron.txt missing core model libs required by train.py: {missing}")
    else:
        _ok("requirements-neuron.txt includes diffusers/torchvision/peft (train.py imports)")


# --------------------------------------------------------------------------- #
# Tier B — requires torch
# --------------------------------------------------------------------------- #


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _load_backend():
    spec = importlib.util.spec_from_file_location("flux_backend_under_test", REPO / "neuron_backend.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_backend_behaviour() -> None:
    backend = _load_backend()
    backend.set_backend("cpu")
    assert not backend.is_xla() and not backend.is_cuda()
    backend.empty_cache(); backend.synchronize(); backend.mark_step(); backend.manual_seed_all(0)
    assert backend.is_available() is False  # cpu
    _ok("backend: cpu helpers are no-ops")

    os.environ["FLUX_BACKEND"] = "xla"
    backend.set_backend("xla")
    assert backend.is_xla() and backend.is_available()
    _ok("backend: FLUX_BACKEND=xla resolves to xla")
    os.environ.pop("FLUX_BACKEND", None)


def check_simulated_xla() -> None:
    import types
    if importlib.util.find_spec("torch_xla") is None:
        import torch  # noqa: PLC0415
        xla_pkg = types.ModuleType("torch_xla")
        core = types.ModuleType("torch_xla.core")
        xm = types.ModuleType("torch_xla.core.xla_model")
        xm.xla_device = lambda: torch.device("cpu")
        xm.mark_step = lambda: None
        xm.set_rng_state = lambda s: torch.manual_seed(s)
        core.xla_model = xm
        xla_pkg.core = core
        sys.modules["torch_xla"] = xla_pkg
        sys.modules["torch_xla.core"] = core
        sys.modules["torch_xla.core.xla_model"] = xm

    backend = _load_backend()
    backend.set_backend("xla")
    assert backend.device() is not None
    backend.mark_step()
    _ok("simulated-xla: backend.device()/mark_step() use torch_xla stub")


def main() -> int:
    print("=== FLUX Neuron static check (Tier A: no torch) ===")
    check_guard_coverage()
    check_port_edits_present()
    check_artifacts()
    check_no_cuda_only_in_reqs()
    check_reqs_excludes_neuron_owned()
    check_reqs_has_core_model_libs()

    print("\n=== Tier B (requires torch) ===")
    if _torch_available():
        try:
            check_backend_behaviour()
            check_simulated_xla()
        except Exception as e:  # noqa: BLE001
            _bad(f"Tier B raised: {e!r}")
    else:
        _skip("torch not installed — Tier B skipped (run on the trn instance / a torch env)")

    print("\n=== Summary ===")
    print(f"passed: {_passes}   failed: {len(_failures)}")
    if _failures:
        print("\nFAILURES:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("All available checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
