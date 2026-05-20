"""Save a few Angry Birds observation frames for visual preprocessing checks.

Run from the project root while no other Angry Birds training process is using
the game server ports:

    conda activate aibirds_tf215
    cd ~/RL2
    python scripts/save_ab_frames.py --num 8
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.envs.angry_birds import AngryBirds


def _stats(arr: np.ndarray) -> dict[str, float | int | list[int] | str]:
    arr_np = np.asarray(arr)
    arr_float = arr_np.astype("float32", copy=False)
    return {
        "shape": list(arr_np.shape),
        "dtype": str(arr_np.dtype),
        "min": float(np.min(arr_float)),
        "max": float(np.max(arr_float)),
        "mean": float(np.mean(arr_float)),
        "std": float(np.std(arr_float)),
        "nan_count": int(np.isnan(arr_float).sum()),
        "inf_count": int(np.isinf(arr_float).sum()),
    }


def _save_contact_sheet(frame_paths: list[Path], out_path: Path, columns: int = 4) -> None:
    if not frame_paths:
        return

    thumbs = [Image.open(path).convert("RGB") for path in frame_paths]
    width, height = thumbs[0].size
    label_height = 20
    rows = int(np.ceil(len(thumbs) / columns))
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, thumb in enumerate(thumbs):
        row = idx // columns
        col = idx % columns
        x = col * width
        y = row * (height + label_height)
        sheet.paste(thumb, (x, y + label_height))
        draw.text((x + 4, y + 3), frame_paths[idx].stem, fill=(0, 0, 0))

    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save raw/preprocessed Angry Birds 128x128 observation frames."
    )
    parser.add_argument("--num", type=int, default=8, help="Number of frames/levels to sample.")
    parser.add_argument(
        "--out-dir",
        default="out/frame_checks/latest",
        help="Directory where PNGs and stats are written.",
    )
    parser.add_argument("--sim-speed", type=int, default=3, help="Science Birds simulation speed.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_128"
    pre_dir = out_dir / "preprocessed_128"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pre_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    raw_paths: list[Path] = []
    env = None
    try:
        env = AngryBirds(num_par_inst=1)
        env.set_sim_speed(args.sim_speed)

        for idx in range(1, args.num + 1):
            env.reset()
            states = env.get_states()
            raw_image = states[0][0]
            bird = states[1][0]
            pre_image = env.preprocess(states)[0][0]
            pre_uint8 = np.clip(pre_image * 255.0, 0, 255).astype("uint8")

            level = getattr(env, "current_level", None)
            raw_path = raw_dir / f"frame_{idx:02d}_level_{level}_raw.png"
            pre_path = pre_dir / f"frame_{idx:02d}_level_{level}_preprocessed.png"
            Image.fromarray(raw_image).save(raw_path)
            Image.fromarray(pre_uint8).save(pre_path)
            raw_paths.append(raw_path)

            raw_stats = _stats(raw_image)
            pre_stats = _stats(pre_image)
            row = {
                "frame": idx,
                "level": level,
                "bird_vector": json.dumps(bird.astype(float).tolist()),
                "raw_path": str(raw_path),
                "preprocessed_path": str(pre_path),
                "raw_min": raw_stats["min"],
                "raw_max": raw_stats["max"],
                "raw_mean": raw_stats["mean"],
                "raw_std": raw_stats["std"],
                "pre_min": pre_stats["min"],
                "pre_max": pre_stats["max"],
                "pre_mean": pre_stats["mean"],
                "pre_std": pre_stats["std"],
            }
            rows.append(row)
            print(
                f"frame {idx:02d} level={level} "
                f"raw mean={raw_stats['mean']:.2f} std={raw_stats['std']:.2f} "
                f"min={raw_stats['min']:.0f} max={raw_stats['max']:.0f} "
                f"bird={bird.astype(int).tolist()}"
            )

        stats_path = out_dir / "frame_stats.csv"
        with stats_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        _save_contact_sheet(raw_paths, out_dir / "contact_sheet_raw.png")
        print(f"\nSaved frames to: {out_dir}")
        print(f"Stats CSV: {stats_path}")
        print(f"Contact sheet: {out_dir / 'contact_sheet_raw.png'}")

        suspicious = [
            row for row in rows
            if float(row["raw_mean"]) > 250.0 or float(row["raw_std"]) < 5.0
        ]
        if suspicious:
            print("\nWarning: some frames look very bright or low contrast. Open the PNGs to inspect.")
        else:
            print("\nFrame stats look non-blank: mean is below 250 and std is above 5.")
    finally:
        if env is not None:
            env._cleanup()


if __name__ == "__main__":
    main()
