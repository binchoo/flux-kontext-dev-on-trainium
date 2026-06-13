#!/usr/bin/env python
# coding=utf-8
"""
FLUX.1-Kontext image-editing INFERENCE on AWS Trainium/Inferentia (Neuron).

Uses optimum-neuron's official NeuronFluxKontextPipeline — the SUPPORTED path
(unlike training, which optimum-neuron does not yet cover for diffusers). This
is the "train on GPU, serve on Neuron" deployment pattern.

Two phases controlled by --export:
  1. Compile: load HF weights, export to Neuron (.neff), save_pretrained to disk.
  2. Serve:   load the compiled artifact and run image editing.

Usage:
  # 1) compile (once; ~10-30 min, needs trn/inf instance):
  python infer_neuron.py --export --compiled-dir ./flux_kontext_neuron

  # 2) run inference (reuses the compiled artifact):
  python infer_neuron.py --compiled-dir ./flux_kontext_neuron \
      --image cat.png --prompt "Add a hat to the cat" --out output.png

Shapes (height/width/TP) are baked into the .neff at compile time — they must
match between --export and serve. Change them => recompile.
"""

import argparse
import os

import torch


def letterbox(img, target_w, target_h, fill=(127, 127, 127)):
    """Resize `img` to fit target_w x target_h preserving aspect ratio, padding
    the remainder (letterbox). Returns (padded_image, paste_box) where paste_box
    = (left, top, w, h) locates the real content inside the padded canvas — used
    to crop the output back to the original aspect ratio.

    The .neff is compiled for a FIXED resolution, so the model must always see
    target_w x target_h. Plain resize would distort non-square inputs (the cat
    stretched vertically); letterbox keeps the aspect ratio.
    """
    from PIL import Image

    ow, oh = img.size
    scale = min(target_w / ow, target_h / oh)
    nw, nh = max(1, round(ow * scale)), max(1, round(oh * scale))
    resized = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (target_w, target_h), fill)
    left, top = (target_w - nw) // 2, (target_h - nh) // 2
    canvas.paste(resized, (left, top))
    return canvas, (left, top, nw, nh)


def unletterbox(img, paste_box, out_w, out_h):
    """Inverse of letterbox: crop the padded content out of `img` and resize it
    back to the original aspect ratio (out_w x out_h)."""
    from PIL import Image

    left, top, nw, nh = paste_box
    cropped = img.crop((left, top, left + nw, top + nh))
    return cropped.resize((out_w, out_h), Image.BICUBIC)


def _patch_config_dtype(pipe):
    """Work around an optimum-neuron 0.4.5 bug: its NeuronModelTextEncoder.forward
    does `outputs[...].to(self.config.dtype)`, but the wrapped DiffusersPretrained
    Config has no `dtype` attr -> AttributeError at inference (T5 text_encoder_2).
    Inject dtype on every sub-model config that lacks it (bf16, our compile dtype).
    """
    for name in ("text_encoder", "text_encoder_2", "transformer", "vae",
                 "vae_encoder", "vae_decoder"):
        comp = getattr(pipe, name, None)
        cfg = getattr(comp, "config", None)
        if cfg is not None and not hasattr(cfg, "dtype"):
            try:
                cfg.dtype = torch.bfloat16
            except Exception:
                pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="black-forest-labs/FLUX.1-Kontext-dev",
                   help="HF model id (base model). Gated — needs HF auth.")
    p.add_argument("--compiled-dir", default="flux_kontext_neuron",
                   help="Where the compiled Neuron artifact is saved/loaded.")
    p.add_argument("--export", action="store_true",
                   help="Compile from HF weights to Neuron and save to --compiled-dir.")
    # Static input shapes — baked into the .neff. Must match at serve time.
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--tensor-parallel-size", type=int, default=8,
                   help="TP degree. Docs recommend 8 on inf2.24xlarge / trn1.")
    # Inference inputs
    p.add_argument("--image", default=None, help="Input image path/URL to edit.")
    p.add_argument("--prompt", default="Change the background to a green forest")
    p.add_argument("--guidance-scale", type=float, default=2.5)
    p.add_argument("--num-inference-steps", type=int, default=28)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="output.png")
    return p.parse_args()


def main():
    args = parse_args()
    from optimum.neuron import NeuronFluxKontextPipeline

    if args.export:
        # --- Phase 1: compile HF weights -> Neuron .neff ---
        print(f"[compile] exporting {args.model_id} to Neuron "
              f"({args.height}x{args.width}, bs={args.batch_size}, tp={args.tensor_parallel_size}) ...")
        compiler_args = {"auto_cast": "none"}  # bf16 weights, no extra autocast (per docs)
        input_shapes = {
            "batch_size": args.batch_size,
            "height": args.height,
            "width": args.width,
        }
        # disable_neuron_cache=True: optimum-neuron 0.4.5's MultiModelCacheEntry
        # only supports stable-diffusion (unet-based) models — for the FLUX DiT it
        # raises NotImplementedError. Disabling the compile cache skips that broken
        # registration path and compiles directly. (Cost: no cross-run cache reuse.)
        pipe = NeuronFluxKontextPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            export=True,
            tensor_parallel_size=args.tensor_parallel_size,
            disable_neuron_cache=True,
            **compiler_args,
            **input_shapes,
        )
        os.makedirs(args.compiled_dir, exist_ok=True)
        pipe.save_pretrained(args.compiled_dir)
        print(f"[compile] saved compiled artifact to {args.compiled_dir}")
    else:
        # --- Phase 2: load compiled artifact and serve ---
        print(f"[serve] loading compiled artifact from {args.compiled_dir} ...")
        pipe = NeuronFluxKontextPipeline.from_pretrained(args.compiled_dir)

    _patch_config_dtype(pipe)

    # If only compiling and no image given, stop after compile.
    if args.export and args.image is None:
        print("[done] compile-only run (no --image). Re-run without --export to serve.")
        return

    from diffusers.utils import load_image
    if args.image is None:
        raise SystemExit("Provide --image <path/URL> to run editing inference.")
    print(f"[serve] editing {args.image} with prompt: {args.prompt!r}")
    raw = load_image(args.image)
    orig_w, orig_h = raw.size
    # Letterbox to the compiled (fixed) resolution so aspect ratio is preserved —
    # plain resize stretches non-square inputs. We crop the result back afterward.
    source, paste_box = letterbox(raw, args.width, args.height)
    generator = torch.Generator("cpu").manual_seed(args.seed)
    image = pipe(
        image=source,
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    ).images[0]
    # Crop the padding out and restore the original aspect ratio.
    image = unletterbox(image, paste_box, orig_w, orig_h)
    image.save(args.out)
    print(f"[done] saved edited image to {args.out} (restored {orig_w}x{orig_h} aspect)")


if __name__ == "__main__":
    main()
