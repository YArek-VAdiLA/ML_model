from pathlib import Path
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pynput.mouse import Controller
from AppKit import NSScreen

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"

CAM_ID = 0
FRAME_W = 1280
FRAME_H = 720
MARGIN = 80
SMOOTHING = 10
INDEX_TIP_ID = 8

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найден файл модели: {MODEL_PATH}")

    screen = NSScreen.mainScreen().frame().size
    screen_w = int(screen.width)
    screen_h = int(screen.height)

    mouse = Controller()

    BaseOptions = python.BaseOptions
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

    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру")

    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0

    prev_time = time.time()

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Не удалось считать кадр с камеры")
                break

            frame = cv2.flip(frame, 1)

            frame_h, frame_w = frame.shape[:2]

            x1, y1 = MARGIN, MARGIN
            x2, y2 = frame_w - MARGIN, frame_h - MARGIN

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                tip = hand[INDEX_TIP_ID]
                px = int(tip.x * frame_w)
                py = int(tip.y * frame_h)

                cv2.circle(frame, (px, py), 12, (255, 0, 255), cv2.FILLED)

                for lm in hand:
                    lx = int(lm.x * frame_w)
                    ly = int(lm.y * frame_h)
                    cv2.circle(frame, (lx, ly), 3, (0, 255, 0), cv2.FILLED)

                if x1 <= px <= x2 and y1 <= py <= y2:
                    nx = (px - x1) / (x2 - x1)
                    ny = (py - y1) / (y2 - y1)

                    nx = clamp(nx, 0.0, 1.0)
                    ny = clamp(ny, 0.0, 1.0)

                    target_x = int(nx * screen_w)
                    target_y = int(ny * screen_h)

                    curr_x = prev_x + (target_x - prev_x) / SMOOTHING
                    curr_y = prev_y + (target_y - prev_y) / SMOOTHING

                    mouse.position = (int(curr_x), int(curr_y))

                    prev_x, prev_y = curr_x, curr_y

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            current_time = time.time()
            fps = 1 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Q - quit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.imshow("Virtual Mouse - Move Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()