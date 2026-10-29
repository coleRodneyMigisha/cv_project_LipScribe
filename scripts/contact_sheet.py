from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FRAME_POSITIONS = (0, 25, 50, 74)


def find_speaker_crop_dirs(
    experiments_dir: Path,
) -> dict[str, Path]:
    """Find full speaker outputs and preview speaker outputs."""
    speaker_dirs: dict[str, Path] = {}

    # Full speaker outputs, for example:
    # experiments/grid_crop_s1/crops
    for crops_dir in experiments_dir.glob("grid_crop_s*/crops"):
        speaker = crops_dir.parent.name.removeprefix("grid_crop_")
        speaker_dirs[speaker] = crops_dir

    # Preview outputs, for example:
    # experiments/grid_crop_preview/s2/crops
    preview_root = experiments_dir / "grid_crop_preview"

    for crops_dir in preview_root.glob("s*/crops"):
        speaker = crops_dir.parent.name

        # Prefer a complete speaker directory if both exist.
        speaker_dirs.setdefault(speaker, crops_dir)

    return dict(sorted(speaker_dirs.items()))


def choose_utterances(
    crops_dir: Path,
    count: int,
    rng: np.random.Generator,
) -> list[Path]:
    crop_files = sorted(crops_dir.glob("*.npy"))

    if not crop_files:
        return []

    if len(crop_files) <= count:
        return crop_files

    selected_indices = rng.choice(
        len(crop_files),
        size=count,
        replace=False,
    )

    return sorted(
        crop_files[index]
        for index in selected_indices
    )


def load_valid_mask(
    crops_dir: Path,
    utterance_id: str,
) -> np.ndarray:
    valid_path = (
        crops_dir.parent
        / "valid"
        / f"{utterance_id}.npy"
    )

    if not valid_path.exists():
        return np.ones(75, dtype=bool)

    return np.load(valid_path, allow_pickle=False)


def make_speaker_sheet(
    speaker: str,
    crops_dir: Path,
    utterances: list[Path],
    output_path: Path,
) -> list[dict[str, str]]:
    rows = len(utterances)
    columns = len(FRAME_POSITIONS)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, max(4, rows * 2.2)),
        squeeze=False,
    )

    manifest_rows: list[dict[str, str]] = []

    for row, crop_path in enumerate(utterances):
        utterance_id = crop_path.stem
        crops = np.load(
            crop_path,
            allow_pickle=False,
        )
        valid = load_valid_mask(
            crops_dir,
            utterance_id,
        )

        if crops.ndim != 3:
            raise ValueError(
                f"{crop_path} has shape {crops.shape}; "
                "expected (frames, height, width)"
            )

        if crops.shape[0] <= max(FRAME_POSITIONS):
            raise ValueError(
                f"{crop_path} has only {crops.shape[0]} frames; "
                "75 frames are required"
            )

        for column, frame_index in enumerate(FRAME_POSITIONS):
            axis = axes[row, column]
            is_valid = bool(valid[frame_index])

            axis.imshow(
                crops[frame_index],
                cmap="gray",
                vmin=0,
                vmax=255,
            )
            axis.axis("off")
            axis.set_title(
                f"{utterance_id}\n"
                f"frame {frame_index} — "
                f"{'valid' if is_valid else 'INVALID'}",
                fontsize=8,
            )

            if not is_valid:
                for spine in axis.spines.values():
                    spine.set_visible(True)
                    spine.set_color("red")
                    spine.set_linewidth(3)

            manifest_rows.append(
                {
                    "speaker": speaker,
                    "utterance_id": utterance_id,
                    "frame_index": str(frame_index),
                    "valid": str(is_valid),
                    "crop_path": str(crop_path),
                }
            )

    figure.suptitle(
        f"GRID mouth-crop inspection — speaker {speaker}",
        fontsize=16,
    )
    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/contact_sheets"),
    )
    parser.add_argument(
        "--per-speaker",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261029,
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    speaker_dirs = find_speaker_crop_dirs(
        args.experiments_dir,
    )

    if not speaker_dirs:
        raise FileNotFoundError(
            "No speaker crop directories were found."
        )

    all_manifest_rows: list[dict[str, str]] = []

    print("Speakers found:", ", ".join(speaker_dirs))

    for speaker, crops_dir in speaker_dirs.items():
        utterances = choose_utterances(
            crops_dir=crops_dir,
            count=args.per_speaker,
            rng=rng,
        )

        if len(utterances) < args.per_speaker:
            print(
                f"{speaker}: only {len(utterances)} "
                f"utterances available; "
                f"requested {args.per_speaker}"
            )
        else:
            print(
                f"{speaker}: selected "
                f"{len(utterances)} utterances"
            )

        output_path = (
            args.output_dir
            / f"{speaker}_contact_sheet.png"
        )

        manifest_rows = make_speaker_sheet(
            speaker=speaker,
            crops_dir=crops_dir,
            utterances=utterances,
            output_path=output_path,
        )
        all_manifest_rows.extend(manifest_rows)

        print("  saved:", output_path)

    manifest_path = (
        args.output_dir
        / "contact_sheet_manifest.csv"
    )
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "speaker",
                "utterance_id",
                "frame_index",
                "valid",
                "crop_path",
            ],
        )
        writer.writeheader()
        writer.writerows(all_manifest_rows)

    print("Manifest saved:", manifest_path)


if __name__ == "__main__":
    main()
