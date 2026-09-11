#!/usr/bin/env python3
"""Create aspect-preserving, bounded-resolution images for stable VLM training."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/images_1024"))
    parser.add_argument("--max-edge", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=180_000)
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No JPEG images found in {args.input_dir}")
    resized = 0
    for source in tqdm(files, desc="preprocess"):
        target = args.output_dir / source.name
        if target.is_file():
            continue
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            scale = min(
                1.0,
                args.max_edge / max(width, height),
                (args.max_pixels / (width * height)) ** 0.5,
            )
            if scale < 1.0:
                size = (max(1, round(width * scale)), max(1, round(height * scale)))
                image = image.resize(size, Image.Resampling.LANCZOS)
                resized += 1
            image.save(target, format="JPEG", quality=args.quality, optimize=True)
    print(f"images={len(files)} newly_resized={resized} output={args.output_dir}")


if __name__ == "__main__":
    main()
