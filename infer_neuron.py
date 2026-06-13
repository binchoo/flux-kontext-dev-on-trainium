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

    # If only compiling and no image given, stop after compile.
    if args.export and args.image is None:
        print("[done] compile-only run (no --image). Re-run without --export to serve.")
        return

    from diffusers.utils import load_image
    if args.image is None:
        raise SystemExit("Provide --image <path/URL> to run editing inference.")
    print(f"[serve] editing {args.image} with prompt: {args.prompt!r}")
    source = load_image(args.image).resize((args.width, args.height))
    generator = torch.Generator("cpu").manual_seed(args.seed)
    image = pipe(
        image=source,
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    ).images[0]
    image.save(args.out)
    print(f"[done] saved edited image to {args.out}")


if __name__ == "__main__":
    main()
