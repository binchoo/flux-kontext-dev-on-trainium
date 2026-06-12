"""Factory for the right Accelerator per backend (FLUX.1-Kontext training).

CUDA/CPU path -> vanilla ``accelerate.Accelerator`` (unchanged behaviour).
XLA path      -> ``optimum.neuron.NeuronAccelerator`` (torch_xla under the hood).

NeuronAccelerator is a drop-in subclass of accelerate.Accelerator, so the
training loop's accelerator.backward()/.accumulate()/.prepare()/.clip_grad_norm_()
calls need no change. The optimum import is lazy so this module imports cleanly
on machines without optimum-neuron.

Note: optimum-neuron officially supports FLUX / Flux-Kontext (NeuronFluxKontext
pipeline, FluxTransformerNeuronConfig) — the DiT is a static-shape graph, so this
port targets a genuinely supported architecture.
"""

from __future__ import annotations

import neuron_backend as backend


def build_accelerator(
    *,
    gradient_accumulation_steps: int,
    mixed_precision: str,
    log_with,
    project_config,
    kwargs_handlers,
    zero_1: bool = False,
):
    """Return an Accelerator appropriate for the resolved backend.

    Args mirror the subset of accelerate.Accelerator the training script uses.
    """
    if backend.is_xla():
        from optimum.neuron import NeuronAccelerator  # noqa: PLC0415

        # On Neuron, bf16 autocast is driven by the accelerator. DDP-style
        # find_unused_parameters kwargs from CUDA are not applicable on XLA, so
        # we drop kwargs_handlers and let NeuronAccelerator manage distribution.
        return NeuronAccelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision if mixed_precision in ("bf16", "no") else "bf16",
            log_with=log_with,
            project_config=project_config,
            zero_1=zero_1,
        )

    from accelerate import Accelerator  # noqa: PLC0415

    return Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=log_with,
        project_config=project_config,
        kwargs_handlers=kwargs_handlers,
    )
