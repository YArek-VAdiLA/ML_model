from pathlib import Path
import csv
import re
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"
ANNOT_DIR = PROJECT_ROOT / "annotations"
FRAMES_ROOT = PROJECT_ROOT / "frames"

TRAIN_ANNOT = ANNOT_DIR / "Annot_TrainList.txt"
TEST_ANNOT = ANNOT_DIR / "Annot_TestList.txt"

OUTPUT_TRAIN = PROJECT_ROOT / "dataset_train.csv"
OUTPUT_TEST = PROJECT_ROOT / "dataset_test.csv"

SELECTED_LABELS = {
    "DOX": "idle",
    "G01": "left_click",
    "G02": "right_click",
    "G08": "double_click",
    "G05": "space_left",
    "G06": "space_right",
}

SAMPLES_PER_SEGMENT = 10
MIN_VALID_FRAMES = 6
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def extract_last_number(path: Path) -> int:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return -1
    return int(matches[-1])


def discover_frame_dirs(root: Path) -> dict[str, list[Path]]:
    by_name = {}
    by_path = {}

    for directory in root.rglob("*"):
        if not directory.is_dir():
            continue

        images = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not images:
            continue

        images = sorted(images, key=extract_last_number)
        by_name[directory.name] = images
        by_path[str(directory)] = images

    print(f"[INFO] Найдено папок с кадрами: {len(by_name)}")
    return {"by_name": by_name, "by_path": by_path}


def find_video_frames(video_id: str, frame_index: dict) -> list[Path] | None:
    by_name = frame_index["by_name"]
    by_path = frame_index["by_path"]

    if video_id in by_name:
        return by_name[video_id]

    candidates = []
    for folder_path, images in by_path.items():
        if video_id in folder_path:
            candidates.append((folder_path, images))

    if len(candidates) == 1:
        return candidates[0][1]

    if len(candidates) > 1:
        # Берем самый длинный путь, обычно он точнее
        candidates.sort(key=lambda x: len(x[0]), reverse=True)
        return candidates[0][1]

    return None


def sample_segment_frames(segment_frames: list[Path], num_samples: int) -> list[Path]:
    if len(segment_frames) == 0:
        return []

    if len(segment_frames) <= num_samples:
        return segment_frames

    indices = np.linspace(0, len(segment_frames) - 1, num_samples, dtype=int)
    return [segment_frames[i] for i in indices]


def normalize_landmarks(landmarks) -> np.ndarray:
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)

    wrist = arr[0].copy()
    arr = arr - wrist

    palm_size = np.linalg.norm(arr[5] - arr[17])
    if palm_size < 1e-6:
        palm_size = 1.0

    arr = arr / palm_size
    return arr.flatten()


def extract_hand_features(image_path: Path, landmarker) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]

    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    wrist = arr[0].copy()
    arr = arr - wrist

    palm_size = np.linalg.norm(arr[5] - arr[17])
    if palm_size < 1e-6:
        palm_size = 1.0

    arr = arr / palm_size
    return arr.flatten()


def read_annotations(annot_file: Path) -> list[dict]:
    rows = []
    with open(annot_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def make_header(samples_per_segment: int) -> list[str]:
    header = ["label"]

    for t in range(samples_per_segment):
        for point_id in range(21):
            header.append(f"f{t}_x{point_id}")
            header.append(f"f{t}_y{point_id}")
            header.append(f"f{t}_z{point_id}")

    return header


def build_dataset(annot_file: Path, output_csv: Path, frame_index: dict):
    rows = read_annotations(annot_file)
    header = make_header(SAMPLES_PER_SEGMENT)

    total_segments = 0
    saved_segments = 0
    skipped_no_frames = 0
    skipped_no_hand = 0
    skipped_label = 0

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    with HandLandmarker.create_from_options(options) as landmarker, \
            open(output_csv, "w", newline="", encoding="utf-8") as f_out:

        writer = csv.writer(f_out)
        writer.writerow(header)

        for row in rows:
            raw_label = row["label"].strip()
            if raw_label not in SELECTED_LABELS:
                skipped_label += 1
                continue

            total_segments += 1

            video_id = row["video"].strip()
            t_start = int(row["t_start"])
            t_end = int(row["t_end"])

            all_video_frames = find_video_frames(video_id, frame_index)
            if not all_video_frames:
                skipped_no_frames += 1
                print(f"[WARN] Не найдены кадры для видео: {video_id}")
                continue

            start_idx = max(0, t_start - 1)
            end_idx = min(len(all_video_frames), t_end)

            segment_frames = all_video_frames[start_idx:end_idx]
            if not segment_frames:
                skipped_no_frames += 1
                print(f"[WARN] Пустой сегмент: {video_id} [{t_start}:{t_end}]")
                continue

            sampled_frames = sample_segment_frames(segment_frames, SAMPLES_PER_SEGMENT)

            features_list = []
            for img_path in sampled_frames:
                feats = extract_hand_features(img_path, landmarker)
                if feats is not None:
                    features_list.append(feats)

            if len(features_list) < MIN_VALID_FRAMES:
                skipped_no_hand += 1
                continue

            while len(features_list) < SAMPLES_PER_SEGMENT:
                features_list.append(features_list[-1])

            features_list = features_list[:SAMPLES_PER_SEGMENT]

            flat_features = np.concatenate(features_list).tolist()
            writer.writerow([SELECTED_LABELS[raw_label]] + flat_features)
            saved_segments += 1

    print("\n" + "=" * 60)
    print(f"[DONE] {annot_file.name} -> {output_csv.name}")
    print(f"Всего подходящих сегментов: {total_segments}")
    print(f"Сохранено: {saved_segments}")
    print(f"Пропущено (метка не нужна): {skipped_label}")
    print(f"Пропущено (не найдены кадры): {skipped_no_frames}")
    print(f"Пропущено (рука не найдена): {skipped_no_hand}")
    print("=" * 60 + "\n")

def main():
    print("[INFO] Индексация папок с кадрами...")
    frame_index = discover_frame_dirs(FRAMES_ROOT)

    print("[INFO] Сбор train датасета...")
    build_dataset(TRAIN_ANNOT, OUTPUT_TRAIN, frame_index)

    print("[INFO] Сбор test датасета...")
    build_dataset(TEST_ANNOT, OUTPUT_TEST, frame_index)


if __name__ == "__main__":
    main()