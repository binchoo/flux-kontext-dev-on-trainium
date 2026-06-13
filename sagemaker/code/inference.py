#!/usr/bin/env python
# coding=utf-8
# =============================================================================
# NOTICE - NON-COMMERCIAL USE ONLY
# -----------------------------------------------------------------------------
# This deployment serves FLUX.1-Kontext [dev], which is licensed under the
# "FLUX.1 [dev] Non-Commercial License" by Black Forest Labs. These deployment
# assets are provided for EVALUATION / PROOF-OF-CONCEPT ONLY and MUST NOT be
# used for commercial serving. See:
#   https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
# You are responsible for complying with the model license.
# =============================================================================
"""
SageMaker inference handler for an ALREADY-COMPILED FLUX.1-Kontext Neuron model.

This implements the SageMaker PyTorch / HuggingFace inference toolkit contract:
    model_fn(model_dir)              -> model object
    input_fn(request_body, ctype)    -> deserialized input
    predict_fn(data, model)          -> prediction
    output_fn(prediction, accept)    -> serialized response

The compiled artifact is the output of `pipe.save_pretrained()` from
`optimum.neuron.NeuronFluxKontextPipeline` (export=True, tensor_parallel_size=8,
1024x1024, bf16). It is loaded with `NeuronFluxKontextPipeline.from_pretrained()`.

SELF-CONTAINED ARTIFACT ASSUMPTION (confirmed):
  The base HF repo (black-forest-labs/FLUX.1-Kontext-dev) is GATED, but the
  compiled save_pretrained() artifact bundles the compiled .neff graphs AND the
  model config/weights it needs. Therefore NO Hugging Face Hub download happens
  at serve time and NO HF token is required. We force fully-offline loading
  (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE) so a network call can never silently
  re-trigger a gated download. If loading ever tries to hit the Hub, that means
  the artifact is incomplete and should be re-exported -- not patched with a token.

SHAPE CONSTRAINT:
  Height/width/TP/batch are baked into the .neff at compile time. This artifact
  is 1024x1024 / TP8 / batch=1. We resize every input image to 1024x1024.

COLD START:
  Loading + Neuron device init takes ~50s. We rely on the SageMaker container
  startup health-check timeout (set high in deploy_endpoint.py) so the endpoint
  does not flap during the cold load. model_fn runs once per worker at startup.
"""

import base64
import io
import json
import logging
import os
import time

# --- Force offline loading so a gated HF download can never be triggered. -----
# (Set BEFORE importing transformers/diffusers/optimum so it takes effect.)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# --- Pin the NeuronCores at IMPORT TIME (before any neuron/torch import) --------
# On the SageMaker endpoint the host exposes ALL cores (NEURON_CORE_HOST_TOTAL=64
# on trn1.32xlarge) and the .neff load failed in init_sp_resource /
# tdrv_get_device_resource_va ("event semaphore vaddr") -> NRT_RESOURCE /
# Allocation Failure -- even though the SAME image loads fine under local
# `docker run --device /dev/neuron0..7`. The difference: locally only 8 devices
# are visible; on SageMaker the runtime must be told WHICH cores to claim, and
# that must be set BEFORE the Neuron runtime initializes (i.e. here, at module
# import, not just via the deploy env which can apply too late relative to the
# torchserve worker's runtime init). Pin cores 0-7 (TP8) and force-set (not
# setdefault) so we win over any inherited value. Override via the deploy env
# NEURON_RT_VISIBLE_CORES if a different core range is desired.
os.environ["NEURON_RT_VISIBLE_CORES"] = os.environ.get("NEURON_RT_VISIBLE_CORES", "0-7")
# Do NOT also set NEURON_RT_NUM_CORES: having both a count and an explicit
# visible-core range can conflict during core reservation. Drop the count.
os.environ.pop("NEURON_RT_NUM_CORES", None)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Fixed shapes baked into the compiled .neff. MUST match the export-time shapes.
HEIGHT = 1024
WIDTH = 1024

# Default sampling params (overridable per-request).
DEFAULT_GUIDANCE_SCALE = 2.5
DEFAULT_NUM_INFERENCE_STEPS = 28


# -----------------------------------------------------------------------------
# model_fn
# -----------------------------------------------------------------------------
def model_fn(model_dir):
    """Load the compiled NeuronFluxKontextPipeline from `model_dir`.

    `model_dir` is the extracted root of model.tar.gz. The compiled artifact
    (save_pretrained output) lives at the archive root, so model_dir IS the
    compiled directory. This runs once per worker; expect ~50s.
    """
    t0 = time.time()
    logger.info("[model_fn] loading compiled Neuron artifact from %s ...", model_dir)

    # Import here (not at module top) so SageMaker can import this file even on
    # a machine without the Neuron stack (e.g. local linting).
    from optimum.neuron import NeuronFluxKontextPipeline

    # The compiled save_pretrained() output may sit either directly at model_dir
    # or under a subfolder if it was tarred with its directory name. Resolve it.
    candidate = model_dir
    if not _looks_like_compiled_dir(candidate):
        for name in sorted(os.listdir(model_dir)):
            sub = os.path.join(model_dir, name)
            if os.path.isdir(sub) and _looks_like_compiled_dir(sub):
                candidate = sub
                break
    logger.info("[model_fn] resolved compiled dir: %s", candidate)

    pipe = NeuronFluxKontextPipeline.from_pretrained(candidate)
    _patch_config_dtype(pipe)
    logger.info("[model_fn] pipeline loaded + device init in %.1fs", time.time() - t0)
    return pipe


def _patch_config_dtype(pipe):
    """Work around optimum-neuron 0.4.5: NeuronModelTextEncoder.forward does
    `outputs[...].to(self.config.dtype)` but the wrapped DiffusersPretrainedConfig
    has no `dtype` -> AttributeError on T5 text_encoder_2 at inference. Inject it."""
    import torch

    for name in ("text_encoder", "text_encoder_2", "transformer", "vae",
                 "vae_encoder", "vae_decoder"):
        comp = getattr(pipe, name, None)
        cfg = getattr(comp, "config", None)
        if cfg is not None and not hasattr(cfg, "dtype"):
            try:
                cfg.dtype = torch.bfloat16
            except Exception:
                pass


def _looks_like_compiled_dir(path):
    """Heuristic: a from_pretrained() dir has a model_index.json (diffusers)."""
    return os.path.isfile(os.path.join(path, "model_index.json"))


# -----------------------------------------------------------------------------
# input_fn
# -----------------------------------------------------------------------------
def input_fn(request_body, content_type="application/json"):
    """Parse the request into a dict with a PIL image + sampling params.

    Accepts application/json:
        {
          "prompt": "Change the background to a forest",
          "image": "<base64 PNG/JPEG>"  OR  "https://.../image.png",
          "guidance_scale": 2.5,          # optional
          "num_inference_steps": 28,      # optional
          "seed": 42                      # optional
        }
    """
    if content_type != "application/json":
        raise ValueError(
            "Unsupported content type: %s (expected application/json)" % content_type
        )

    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    payload = json.loads(request_body)

    prompt = payload.get("prompt")
    if not prompt:
        raise ValueError("Request must include a non-empty 'prompt'.")

    image_field = payload.get("image")
    if not image_field:
        raise ValueError("Request must include 'image' (base64 string or URL).")

    raw = _load_image(image_field).convert("RGB")
    orig_w, orig_h = raw.size
    # Letterbox to the fixed compiled shape (preserves aspect ratio). Plain
    # resize stretches non-square inputs (e.g. the cat elongated vertically).
    # We record paste_box + original size to crop the output back afterward.
    image, paste_box = _letterbox(raw, WIDTH, HEIGHT)

    return {
        "prompt": prompt,
        "image": image,
        "paste_box": paste_box,
        "orig_size": (orig_w, orig_h),
        "guidance_scale": float(payload.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)),
        "num_inference_steps": int(
            payload.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)
        ),
        "seed": payload.get("seed", None),
    }


def _letterbox(img, target_w, target_h, fill=(127, 127, 127)):
    """Fit img into target_w x target_h preserving aspect ratio, pad the rest.
    Returns (canvas, (left, top, nw, nh)). The .neff is a fixed resolution, so
    the model must always see target_w x target_h; letterbox avoids distortion."""
    from PIL import Image

    ow, oh = img.size
    scale = min(target_w / ow, target_h / oh)
    nw, nh = max(1, round(ow * scale)), max(1, round(oh * scale))
    resized = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (target_w, target_h), fill)
    left, top = (target_w - nw) // 2, (target_h - nh) // 2
    canvas.paste(resized, (left, top))
    return canvas, (left, top, nw, nh)


def _unletterbox(img, paste_box, out_w, out_h):
    """Crop the padded content out and restore the original aspect ratio."""
    from PIL import Image

    left, top, nw, nh = paste_box
    return img.crop((left, top, left + nw, top + nh)).resize((out_w, out_h), Image.BICUBIC)


def _load_image(image_field):
    """Load a PIL image from a base64 string or an http(s) URL."""
    from PIL import Image

    if isinstance(image_field, str) and image_field.lower().startswith(
        ("http://", "https://")
    ):
        # diffusers.load_image handles URL fetching + EXIF transpose.
        from diffusers.utils import load_image

        return load_image(image_field)

    # Otherwise treat as base64 (optionally a data: URI).
    b64 = image_field
    if isinstance(b64, str) and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw))


# -----------------------------------------------------------------------------
# predict_fn
# -----------------------------------------------------------------------------
def predict_fn(data, model):
    """Run the editing pipeline and return a PIL image."""
    import torch

    generator = None
    if data.get("seed") is not None:
        generator = torch.Generator("cpu").manual_seed(int(data["seed"]))

    t0 = time.time()
    result = model(
        image=data["image"],
        prompt=data["prompt"],
        guidance_scale=data["guidance_scale"],
        num_inference_steps=data["num_inference_steps"],
        generator=generator,
    )
    logger.info("[predict_fn] inference completed in %.1fs", time.time() - t0)
    image = result.images[0]
    # Restore the original aspect ratio (undo the letterbox padding).
    pb = data.get("paste_box")
    if pb is not None:
        ow, oh = data["orig_size"]
        image = _unletterbox(image, pb, ow, oh)
    return image


# -----------------------------------------------------------------------------
# output_fn
# -----------------------------------------------------------------------------
def output_fn(prediction, accept="application/json"):
    """Serialize the PIL image as a base64 PNG inside JSON."""
    if accept not in ("application/json", "*/*"):
        raise ValueError("Unsupported accept type: %s (expected application/json)" % accept)

    buf = io.BytesIO()
    prediction.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    body = json.dumps({"image_base64": image_b64})
    return body, "application/json"
