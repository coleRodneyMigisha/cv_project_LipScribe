from pathlib import Path
import csv
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRID_DIR = (
    PROJECT_ROOT
    / "Data"
    / "grip_corpus"
    / "grid-corpus-dataset-for-training-lipnet"
    / "data"
    / "s1_processed"
)

MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "grid_crop_s1"

CROPS_DIR = OUTPUT_DIR / "crops"
VALID_DIR = OUTPUT_DIR / "valid"
INDICES_DIR = OUTPUT_DIR / "frame_indices"
LOG_PATH = OUTPUT_DIR / "crop_log.csv"

TARGET_FRAME_COUNT = 75

sys.path.insert(0, str(PROJECT_ROOT))

from lipscribe.preprocessing.crop import crop_video


def main():
    for directory in (
        OUTPUT_DIR,
        CROPS_DIR,
        VALID_DIR,
        INDICES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    videos = sorted(GRID_DIR.glob("*.mpg"))

    if not videos:
        raise FileNotFoundError(
            f"No MPG files found in {GRID_DIR}"
        )

    fieldnames = [
        "utterance_id",
        "video_path",
        "valid_frames",
        "invalid_frames",
        "invalid_fraction",
        "exclude_utterance",
        "status",
        "error",
    ]

    with LOG_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=fieldnames,
        )

        if LOG_PATH.stat().st_size == 0:
            writer.writeheader()

        for number, video_path in enumerate(videos, start=1):
            utterance_id = video_path.stem
            crop_path = CROPS_DIR / f"{utterance_id}.npy"

            # Resume safely: skip clips already processed successfully.
            if crop_path.exists():
                print(
                    f"[{number}/{len(videos)}] "
                    f"Skipping {utterance_id}"
                )
                continue

            print(
                f"[{number}/{len(videos)}] "
                f"Processing {utterance_id}",
                flush=True,
            )

            try:
                result = crop_video(
                    video_path=video_path,
                    model_path=MODEL_PATH,
                    target_frame_count=TARGET_FRAME_COUNT,
                )

                assert result.crops.shape == (
                    TARGET_FRAME_COUNT,
                    96,
                    96,
                )
                assert result.crops.dtype == np.uint8
                assert result.valid.shape == (
                    TARGET_FRAME_COUNT,
                )
                assert result.frame_indices.shape == (
                    TARGET_FRAME_COUNT,
                )

                np.save(crop_path, result.crops)
                np.save(
                    VALID_DIR / f"{utterance_id}.npy",
                    result.valid,
                )
                np.save(
                    INDICES_DIR / f"{utterance_id}.npy",
                    result.frame_indices,
                )

                writer.writerow(
                    {
                        "utterance_id": utterance_id,
                        "video_path": str(video_path),
                        "valid_frames": int(result.valid.sum()),
                        "invalid_frames": int(
                            (~result.valid).sum()
                        ),
                        "invalid_fraction": (
                            result.invalid_fraction
                        ),
                        "exclude_utterance": (
                            result.exclude_utterance
                        ),
                        "status": "ok",
                        "error": "",
                    }
                )

                log_file.flush()

            except Exception as error:
                print(
                    f"  ERROR: {type(error).__name__}: {error}",
                    flush=True,
                )

                writer.writerow(
                    {
                        "utterance_id": utterance_id,
                        "video_path": str(video_path),
                        "valid_frames": "",
                        "invalid_frames": "",
                        "invalid_fraction": "",
                        "exclude_utterance": "",
                        "status": "error",
                        "error": repr(error),
                    }
                )

                log_file.flush()


if __name__ == "__main__":
    main()
