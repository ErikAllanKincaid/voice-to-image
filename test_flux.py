#!/usr/bin/env python3
"""Test script for FLUX.1 image generation model.

FLUX.1 variants:
- black-forest-labs/FLUX.1-schnell  (fast, ~4 steps, Apache 2.0)
- black-forest-labs/FLUX.1-dev      (balanced, ~20-50 steps, non-commercial)

Requirements: 16GB+ VRAM recommended, 24GB for comfortable use.
"""

import argparse
import time
import torch
from diffusers import FluxPipeline
from PIL import Image


def load_flux(model_id: str = "black-forest-labs/FLUX.1-schnell", multi_gpu: bool = False):
    """Load FLUX pipeline.

    Args:
        model_id: HuggingFace model ID
        multi_gpu: If True, spread model across all available GPUs (needs 2x 12GB)
                   If False, use CPU offload (slower, works on single 12GB)
    """
    print(f"Loading FLUX model: {model_id}...")
    start = time.time()

    if multi_gpu and torch.cuda.device_count() > 1:
        # Spread model across multiple GPUs using device_map
        print(f"Using {torch.cuda.device_count()} GPUs with model parallelism...")
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="balanced",  # Spread evenly across GPUs
        )
    else:
        # Single GPU with CPU offload
        print("Using CPU offload for memory efficiency...")
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()

    print(f"Model loaded in {time.time() - start:.1f}s")
    return pipe


def generate(pipe, prompt: str, width: int = 768, height: int = 432,
             num_steps: int = 4, guidance_scale: float = 0.0):
    """Generate image with FLUX."""
    print(f"Generating: {prompt[:60]}...")
    start = time.time()

    image = pipe(
        prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    print(f"Generated in {time.time() - start:.1f}s")
    return image


def main():
    parser = argparse.ArgumentParser(description="Test FLUX.1 image generation")
    parser.add_argument("--prompt", "-p", type=str,
                        default="A robot and a spaceship in a futuristic city, cinematic lighting, highly detailed, 8k",
                        help="Text prompt for image generation")
    parser.add_argument("--model", "-m", type=str,
                        default="black-forest-labs/FLUX.1-schnell",
                        choices=["black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-dev"],
                        help="FLUX model variant")
    parser.add_argument("--width", "-W", type=int, default=768)
    parser.add_argument("--height", "-H", type=int, default=432)
    parser.add_argument("--steps", "-s", type=int, default=4,
                        help="Inference steps (schnell: 1-4, dev: 20-50)")
    parser.add_argument("--guidance", "-g", type=float, default=0.0,
                        help="Guidance scale (schnell: 0, dev: 3.5)")
    parser.add_argument("--output", "-o", type=str, default="flux_output.png")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use multiple GPUs with model parallelism (needs 2x 12GB)")
    args = parser.parse_args()

    # Check VRAM
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram:.1f}GB")
        if vram < 16:
            print("Warning: FLUX recommends 16GB+ VRAM. May OOM.")

    # Load and generate
    pipe = load_flux(args.model, multi_gpu=args.multi_gpu)
    image = generate(pipe, args.prompt, args.width, args.height,
                     args.steps, args.guidance)

    # Save
    image.save(args.output)
    print(f"Saved to {args.output}")

    # Show VRAM usage
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM used: {used:.1f}GB")


if __name__ == "__main__":
    main()
