#!/usr/bin/env python3
"""
scripts/make_synthetic.py — Synthetic strip image generator for LUMINE.

Generates JPEG images that look like dipstick test strips with coloured pads
drawn from the reference-color charts.  All output images are clearly marked
as synthetic and carry ``is_synthetic = 1`` when added to a dataset.

IMPORTANT: Synthetic data is meant for testing and demos only.
           It must NEVER be used to activate an ML model in production.

Usage (from the LUMINE repo root):
  python scripts/make_synthetic.py --output-dir uploads/synthetic --count 10
  python scripts/make_synthetic.py --profile URS-4K --count 5 --seed 42 \\
         --dataset-id DS_... --db lumine.db

Options:
  --profile      Strip profile: URS-4K or SALIVA-4K  [default: URS-4K]
  --count        Number of images to generate         [default: 10]
  --output-dir   Output directory                     [default: uploads/synthetic]
  --seed         Random seed for reproducibility      [default: random]
  --dataset-id   If given, add samples to this dataset in the LUMINE DB
  --db           Path to lumine.db                    [default: lumine.db]
  --noise-std    Pixel noise std-dev                  [default: 12]
  --pad-size     Width and height of each pad (px)    [default: 60]
"""

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Allow running from repo root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.chart_colors import PROFILE_CHARTS, get_chart


# ---------------------------------------------------------------------------
# Strip geometry constants
# ---------------------------------------------------------------------------
STRIP_WIDTH_PX  = 800   # horizontal (pads run left-to-right)
STRIP_HEIGHT_PX = 160
PAD_MARGIN      = 8     # px gap between pads
HANDLE_FRAC     = 0.18  # fraction of strip that is bare handle (white plastic)
LABEL_FONT      = cv2.FONT_HERSHEY_SIMPLEX
SYNTHETIC_LABEL = "SYNTHETIC — NOT FOR CLINICAL USE"


def _gaussian_noise(shape, std: float) -> np.ndarray:
    return np.random.normal(0, std, shape).astype(np.float32)


def _clamp_rgb(rgb):
    return [max(0, min(255, int(v))) for v in rgb]


def generate_strip_image(
    chart: dict,
    markers: list,
    noise_std: float = 12.0,
    pad_size: int = 60,
    seed: int | None = None,
) -> tuple[np.ndarray, list]:
    """
    Draw a synthetic horizontal strip image.

    Returns
    -------
    img : np.ndarray  HxWx3 BGR uint8
    ground_truth : list of dicts {marker, pad_index, level, rgb}
    """
    if seed is not None:
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
    else:
        rng = random.Random()
        np_rng = np.random.default_rng()

    h = STRIP_HEIGHT_PX
    w = STRIP_WIDTH_PX

    # White plastic background (slightly warm)
    substrate_rgb = _clamp_rgb([235 + rng.gauss(0, 5),
                                 228 + rng.gauss(0, 5),
                                 215 + rng.gauss(0, 5)])
    img = np.full((h, w, 3), substrate_rgb[::-1], dtype=np.float32)  # BGR

    # Add global illumination gradient (darker toward right)
    gradient = np.linspace(1.0, 0.85, w, dtype=np.float32).reshape(1, w, 1)
    img = img * gradient

    active_w  = int(w * (1.0 - HANDLE_FRAC))
    n_pads    = len(markers)
    total_pad = n_pads * pad_size + (n_pads - 1) * PAD_MARGIN
    start_x   = (active_w - total_pad) // 2

    ground_truth = []

    for i, marker in enumerate(markers):
        levels = chart.get(marker, [])
        if not levels:
            continue
        level_name, base_rgb = rng.choice(levels)

        # Add per-pad noise
        noisy_rgb = [base_rgb[c] + np_rng.normal(0, noise_std) for c in range(3)]
        noisy_rgb = _clamp_rgb(noisy_rgb)

        x0 = start_x + i * (pad_size + PAD_MARGIN)
        x1 = x0 + pad_size
        y0 = (h - pad_size) // 2
        y1 = y0 + pad_size

        # Draw pad rectangle (BGR)
        pad_bgr = noisy_rgb[::-1]
        cv2.rectangle(img, (x0, y0), (x1, y1), pad_bgr, -1)

        # Draw subtle border
        border_color = [c * 0.75 for c in pad_bgr]
        cv2.rectangle(img, (x0, y0), (x1, y1), border_color, 1)

        # Add pixel-level noise on the pad
        pad_region = img[y0:y1, x0:x1]
        pad_noise  = np_rng.normal(0, noise_std * 0.5, pad_region.shape).astype(np.float32)
        img[y0:y1, x0:x1] = np.clip(pad_region + pad_noise, 0, 255)

        # Draw pad index label (putText requires uint8)
        img_u8_tmp = np.clip(img, 0, 255).astype(np.uint8)
        cv2.putText(img_u8_tmp, f"#{i}", (x0 + 2, y0 - 4),
                    LABEL_FONT, 0.35, (80, 80, 80), 1, cv2.LINE_AA)
        img = img_u8_tmp.astype(np.float32)

        ground_truth.append({
            'marker':    marker,
            'pad_index': i,
            'level':     level_name,
            'rgb':       noisy_rgb,
            'bbox':      [x0, y0, x1, y1],
        })

    # Watermark
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    cv2.putText(img_u8, SYNTHETIC_LABEL,
                (10, h - 10), LABEL_FONT, 0.4, (60, 60, 200), 1, cv2.LINE_AA)

    return img_u8, ground_truth


def generate_batch(
    profile_id: str,
    count: int,
    output_dir: str,
    noise_std: float = 12.0,
    pad_size: int = 60,
    seed: int | None = None,
) -> list:
    """
    Generate *count* synthetic strip images and save them as JPEG.

    Returns a list of result dicts, each with:
      image_path, strip_type, is_synthetic, ground_truth (list of pad labels)
    """
    chart = get_chart(profile_id)
    if not chart:
        raise ValueError(f"Unknown profile '{profile_id}'. "
                         f"Available: {list(PROFILE_CHARTS.keys())}")

    # Marker order is consistent (sorted) so ground truth always matches the DB
    markers = sorted(chart.keys())

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for k in range(count):
        img_seed = None if seed is None else (seed * 1000 + k)
        img, gt  = generate_strip_image(
            chart=chart, markers=markers,
            noise_std=noise_std, pad_size=pad_size, seed=img_seed,
        )
        fname = f"synth_{profile_id.lower().replace('-', '_')}_{k:04d}.jpg"
        fpath = os.path.join(output_dir, fname)
        cv2.imwrite(fpath, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        results.append({
            'image_path':  os.path.abspath(fpath),
            'strip_type':  profile_id,
            'is_synthetic': True,
            'ground_truth': gt,
            'markers':     markers,
        })
        print(f"  [{k+1}/{count}] {fname}")

    return results


def _add_to_db(results: list, dataset_id: str, db_path: str):
    """Add generated samples (and their ground-truth labels) to the LUMINE DB."""
    from app.database.db_manager import DatabaseManager
    db = DatabaseManager(db_path)

    for r in results:
        sample_id = db.add_sample(
            dataset_id=dataset_id,
            image_path=r['image_path'],
            strip_type=r['strip_type'],
            capture_meta={'source': 'make_synthetic.py'},
            is_synthetic=True,
        )
        labels = [
            {
                'marker':        p['marker'],
                'pad_index':     p['pad_index'],
                'true_level':    p['level'],
                'label_source':  'chart',
            }
            for p in r['ground_truth']
        ]
        saved = db.save_pad_labels(sample_id, labels, annotator='synthetic_generator')
        print(f"    → DB sample {sample_id}: {saved} labels saved")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic dipstick strip images for LUMINE."
    )
    parser.add_argument('--profile',    default='URS-4K',
                        choices=list(PROFILE_CHARTS.keys()),
                        help='Strip profile to generate images for')
    parser.add_argument('--count',      type=int, default=10,
                        help='Number of images to generate')
    parser.add_argument('--output-dir', default='uploads/synthetic',
                        help='Directory to write JPEG files into')
    parser.add_argument('--seed',       type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--noise-std',  type=float, default=12.0,
                        help='Gaussian noise std-dev applied to pad colors')
    parser.add_argument('--pad-size',   type=int, default=60,
                        help='Pad width and height in pixels')
    parser.add_argument('--dataset-id', default=None,
                        help='If given, register samples in this LUMINE dataset')
    parser.add_argument('--db',         default='lumine.db',
                        help='Path to lumine.db (only used with --dataset-id)')

    args = parser.parse_args()

    print(f"Generating {args.count} synthetic {args.profile} strip images "
          f"→ {args.output_dir}")

    results = generate_batch(
        profile_id=args.profile,
        count=args.count,
        output_dir=args.output_dir,
        noise_std=args.noise_std,
        pad_size=args.pad_size,
        seed=args.seed,
    )

    # Write a ground-truth JSON sidecar for inspection
    gt_path = os.path.join(args.output_dir,
                           f"ground_truth_{args.profile.lower().replace('-','_')}.json")
    with open(gt_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Ground truth written to {gt_path}")

    if args.dataset_id:
        print(f"Adding {len(results)} samples to dataset {args.dataset_id} ...")
        _add_to_db(results, args.dataset_id, args.db)
        print("Done.")
    else:
        print(f"\nDone. Pass --dataset-id <id> to register in the LUMINE DB.")


if __name__ == '__main__':
    main()
