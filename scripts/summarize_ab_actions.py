"""Summarize Angry Birds action usage from a training diagnostics.jsonl file.

Example:
    python scripts/summarize_ab_actions.py \
        --model ab_agent_20260516_220759_rainbow_D_full_n3_noisy
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ANGLE_RESOLUTION = 20
TAP_TIME_RESOLUTION = 10
MAXIMUM_TAP_TIME = 4000
PHI = 10
PSI = 40


def action_to_params(action: int) -> tuple[int, int, int, int]:
    angle_bin = int(action) // TAP_TIME_RESOLUTION
    tap_bin = int(action) % TAP_TIME_RESOLUTION
    angle = PHI + int(angle_bin * (180 - PHI - PSI) / (ANGLE_RESOLUTION - 1))
    tap_ms = int(tap_bin / TAP_TIME_RESOLUTION * MAXIMUM_TAP_TIME)
    return angle_bin, tap_bin, angle, tap_ms


def resolve_diagnostics_path(model: str | None, diagnostics_path: str | None) -> Path:
    if diagnostics_path:
        return Path(diagnostics_path)
    if not model:
        raise SystemExit("Provide --model or --diagnostics-path.")
    return Path("out") / "angry_birds" / model / "diagnostics.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize angle/tap action usage from Angry Birds diagnostics."
    )
    parser.add_argument("--model", help="Run folder under out/angry_birds/.")
    parser.add_argument("--diagnostics-path", help="Direct path to diagnostics.jsonl.")
    parser.add_argument("--top", type=int, default=20, help="Number of top actions to print.")
    args = parser.parse_args()

    diagnostics_path = resolve_diagnostics_path(args.model, args.diagnostics_path)
    if not diagnostics_path.is_file():
        raise SystemExit(f"Diagnostics file not found: {diagnostics_path}")

    action_counts: Counter[int] = Counter()
    angle_counts: Counter[int] = Counter()
    tap_counts: Counter[int] = Counter()
    rows = 0

    with diagnostics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") != "training_step":
                continue
            action = record.get("action")
            if action is None:
                continue
            action = int(action)
            angle_bin, tap_bin, _angle, _tap_ms = action_to_params(action)
            action_counts[action] += 1
            angle_counts[angle_bin] += 1
            tap_counts[tap_bin] += 1
            rows += 1

    if rows == 0:
        raise SystemExit(f"No training_step action rows found in {diagnostics_path}")

    print(f"Diagnostics: {diagnostics_path}")
    print(f"Logged actions: {rows}")
    print(f"Unique actions: {len(action_counts)} / {ANGLE_RESOLUTION * TAP_TIME_RESOLUTION}")
    print(f"Unique angle bins: {len(angle_counts)} / {ANGLE_RESOLUTION}")
    print(f"Unique tap bins: {len(tap_counts)} / {TAP_TIME_RESOLUTION}")

    print("\nTap usage:")
    for tap_bin in range(TAP_TIME_RESOLUTION):
        count = tap_counts[tap_bin]
        pct = count / rows * 100.0
        tap_ms = int(tap_bin / TAP_TIME_RESOLUTION * MAXIMUM_TAP_TIME)
        print(f"  tap_bin={tap_bin:>2} tap_ms={tap_ms:>4}: {count:>6} ({pct:5.1f}%)")

    print("\nAngle usage:")
    for angle_bin in range(ANGLE_RESOLUTION):
        count = angle_counts[angle_bin]
        pct = count / rows * 100.0
        angle = PHI + int(angle_bin * (180 - PHI - PSI) / (ANGLE_RESOLUTION - 1))
        print(f"  angle_bin={angle_bin:>2} angle={angle:>3}: {count:>6} ({pct:5.1f}%)")

    print(f"\nTop {args.top} actions:")
    for action, count in action_counts.most_common(args.top):
        angle_bin, tap_bin, angle, tap_ms = action_to_params(action)
        pct = count / rows * 100.0
        print(
            f"  action={action:>3} count={count:>6} ({pct:5.1f}%) "
            f"angle_bin={angle_bin:>2} angle={angle:>3} "
            f"tap_bin={tap_bin:>2} tap_ms={tap_ms:>4}"
        )


if __name__ == "__main__":
    main()
