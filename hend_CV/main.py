
from pathlib import Path
from collections import deque
import time
import subprocess

import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pynput.mouse import Controller, Button
import AppKit


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"
RF_MODEL_PATH = PROJECT_ROOT / "gesture_rf.joblib"
LABEL_ENCODER_PATH = PROJECT_ROOT / "label_encoder.joblib"


CAM_ID = 0
FRAME_W = 640
FRAME_H = 480

MARGIN = 70
SMOOTHING = 5


INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18


PALM_MOVE_DEADZONE = 3
COMMAND_MOTION_THRESHOLD = 6


SWIPE_X_THRESHOLD = 13
SWIPE_Y_LIMIT = 15
SWIPE_COOLDOWN = 1.0


WINDOW_SIZE = 10
PREDICT_EVERY_N_FRAMES = 2
GLOBAL_ACTION_COOLDOWN = 0.9

GESTURE_THRESHOLDS = {
    "left_click": 0.30,
    "right_click": 0.30,
    "double_click": 0.20,
}


SCROLL_DEADZONE = 3
SCROLL_STEP_PIXELS = 3
SCROLL_UNITS = 1


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def get_screen_size():
    screen = AppKit.NSScreen.mainScreen().frame().size
    return int(screen.width), int(screen.height)


def normalize_landmarks(hand_landmarks) -> np.ndarray:
    arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)

    wrist = arr[0].copy()
    arr = arr - wrist

    palm_size = np.linalg.norm(arr[5] - arr[17])
    if palm_size < 1e-6:
        palm_size = 1.0

    arr = arr / palm_size
    return arr.flatten()


def get_palm_center(hand_landmarks, frame_w, frame_h):
    ids = [0, 5, 9, 13, 17]
    xs = [hand_landmarks[i].x * frame_w for i in ids]
    ys = [hand_landmarks[i].y * frame_h for i in ids]

    cx = int(sum(xs) / len(xs))
    cy = int(sum(ys) / len(ys))
    return cx, cy


def is_open_palm_pose(hand_landmarks) -> bool:
    index_open = hand_landmarks[INDEX_TIP].y < hand_landmarks[INDEX_PIP].y
    middle_open = hand_landmarks[MIDDLE_TIP].y < hand_landmarks[MIDDLE_PIP].y
    ring_open = hand_landmarks[RING_TIP].y < hand_landmarks[RING_PIP].y
    pinky_open = hand_landmarks[PINKY_TIP].y < hand_landmarks[PINKY_PIP].y

    return index_open and middle_open and ring_open and pinky_open


def is_fist_pose(hand_landmarks) -> bool:
    index_closed = hand_landmarks[INDEX_TIP].y > hand_landmarks[INDEX_PIP].y
    middle_closed = hand_landmarks[MIDDLE_TIP].y > hand_landmarks[MIDDLE_PIP].y
    ring_closed = hand_landmarks[RING_TIP].y > hand_landmarks[RING_PIP].y
    pinky_closed = hand_landmarks[PINKY_TIP].y > hand_landmarks[PINKY_PIP].y

    return index_closed and middle_closed and ring_closed and pinky_closed


def swipe_space(direction: str):
    if direction == "left":
        print("PALM SWIPE LEFT", flush=True)
        script = 'tell application "System Events" to key code 123 using control down'
        subprocess.run(["osascript", "-e", script], check=False)

    elif direction == "right":
        print("PALM SWIPE RIGHT", flush=True)
        script = 'tell application "System Events" to key code 124 using control down'
        subprocess.run(["osascript", "-e", script], check=False)


def execute_action(label: str, mouse: Controller):
    print(f"EXECUTE ACTION: {label}", flush=True)

    if label == "left_click":
        mouse.click(Button.left, 1)
        print("LEFT CLICK SENT", flush=True)

    elif label == "right_click":
        mouse.click(Button.right, 1)
        print("RIGHT CLICK SENT", flush=True)

    elif label == "double_click":
        mouse.click(Button.left, 2)
        print("DOUBLE CLICK SENT", flush=True)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найден файл модели руки: {MODEL_PATH}")
    if not RF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найдена модель жестов: {RF_MODEL_PATH}")
    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Не найден label encoder: {LABEL_ENCODER_PATH}")

    rf_model = joblib.load(RF_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    screen_w, screen_h = get_screen_size()
    mouse = Controller()

    BaseOptions = python.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру")

    feature_window = deque(maxlen=WINDOW_SIZE)

    prev_cursor_x, prev_cursor_y = mouse.position
    last_action_time = 0.0
    last_action_label = None

    prev_scroll_y = None
    scroll_accum = 0.0

    prev_palm_x = None
    prev_palm_y = None
    last_swipe_time = 0.0

    frame_id = 0
    prev_time = time.time()

    current_label = "none"
    current_conf = 0.0
    move_mode = False
    fist_mode = False
    palm_motion = 0.0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Не удалось считать кадр", flush=True)
                break

            frame_id += 1

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            x1, y1 = MARGIN, MARGIN
            x2, y2 = w - MARGIN, h - MARGIN

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = frame_id * 33
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            move_mode = False
            fist_mode = False
            palm_motion = 0.0

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                fist_mode = is_fist_pose(hand)
                open_palm = is_open_palm_pose(hand)

                palm_x, palm_y = get_palm_center(hand, w, h)
                cv2.circle(frame, (palm_x, palm_y), 10, (0, 255, 255), cv2.FILLED)

                tip = hand[INDEX_TIP]
                px = int(tip.x * w)
                py = int(tip.y * h)
                cv2.circle(frame, (px, py), 8, (255, 0, 255), cv2.FILLED)

                if prev_palm_x is not None and prev_palm_y is not None:
                    dx = palm_x - prev_palm_x
                    dy = palm_y - prev_palm_y
                    palm_motion = (dx * dx + dy * dy) ** 0.5
                else:
                    dx = 0
                    dy = 0
                    palm_motion = 0.0

                prev_palm_x, prev_palm_y = palm_x, palm_y
                now = time.time()

                if fist_mode:
                    current_label = "scroll_mode"
                    current_conf = 1.0
                    move_mode = False

                    feature_window.clear()
                    last_action_label = None

                    if prev_scroll_y is None:
                        prev_scroll_y = palm_y
                        scroll_accum = 0.0
                    else:
                        dy_scroll = palm_y - prev_scroll_y
                        prev_scroll_y = palm_y

                        if abs(dy_scroll) >= SCROLL_DEADZONE:
                            scroll_accum += dy_scroll

                            while abs(scroll_accum) >= SCROLL_STEP_PIXELS:
                                if scroll_accum > 0:
                                    mouse.scroll(0, -SCROLL_UNITS)
                                    print("FIST SCROLL DOWN", flush=True)
                                    scroll_accum -= SCROLL_STEP_PIXELS
                                else:
                                    mouse.scroll(0, SCROLL_UNITS)
                                    print("FIST SCROLL UP", flush=True)
                                    scroll_accum += SCROLL_STEP_PIXELS

                else:
                    prev_scroll_y = None
                    scroll_accum = 0.0

                    swipe_triggered = (
                        abs(dx) >= SWIPE_X_THRESHOLD
                        and abs(dy) <= SWIPE_Y_LIMIT
                        and now - last_swipe_time >= SWIPE_COOLDOWN
                    )

                    if swipe_triggered:
                        if dx > 0:
                            swipe_space("right")
                            current_label = "space_right"
                        else:
                            swipe_space("left")
                            current_label = "space_left"

                        current_conf = abs(dx)
                        last_swipe_time = now

                        feature_window.clear()
                        last_action_label = None
                        move_mode = False

                    elif open_palm and x1 <= palm_x <= x2 and y1 <= palm_y <= y2:
                        nx = (palm_x - x1) / (x2 - x1)
                        ny = (palm_y - y1) / (y2 - y1)

                        nx = clamp(nx, 0.0, 1.0)
                        ny = clamp(ny, 0.0, 1.0)

                        target_x = int(nx * screen_w)
                        target_y = int(ny * screen_h)

                        smooth_x = prev_cursor_x + (target_x - prev_cursor_x) / SMOOTHING
                        smooth_y = prev_cursor_y + (target_y - prev_cursor_y) / SMOOTHING

                        if palm_motion >= PALM_MOVE_DEADZONE:
                            move_mode = True

                        mouse.position = (int(smooth_x), int(smooth_y))
                        prev_cursor_x, prev_cursor_y = smooth_x, smooth_y

                        current_label = "move_mode"
                        current_conf = palm_motion if palm_motion > 0 else 1.0

                        feature_window.clear()
                        last_action_label = None

                    else:
                        move_mode = False

                        if (not open_palm) and palm_motion < COMMAND_MOTION_THRESHOLD:
                            feats = normalize_landmarks(hand)
                            feature_window.append(feats)

                            if len(feature_window) == WINDOW_SIZE and frame_id % PREDICT_EVERY_N_FRAMES == 0:
                                window_features = np.concatenate(list(feature_window)).reshape(1, -1)

                                probs = rf_model.predict_proba(window_features)[0]
                                pred_idx = int(np.argmax(probs))
                                pred_conf = float(np.max(probs))
                                pred_label = label_encoder.inverse_transform([pred_idx])[0]

                                current_label = pred_label
                                current_conf = pred_conf

                                if pred_label in ("scroll_up", "scroll_down", "space_left", "space_right"):
                                    pred_label = "ignore_dynamic"

                                threshold = GESTURE_THRESHOLDS.get(pred_label, 0.45)

                                if pred_conf < 0.30:
                                    last_action_label = None

                                if pred_label in ("left_click", "right_click", "double_click"):
                                    if pred_conf >= threshold and now - last_action_time >= GLOBAL_ACTION_COOLDOWN:
                                        if pred_label != last_action_label:
                                            print(f"SOFT EXEC: {pred_label}, conf={pred_conf:.2f}", flush=True)
                                            execute_action(pred_label, mouse)
                                            last_action_time = now
                                            last_action_label = pred_label
                        else:
                            current_label = "idle_gesture"
                            current_conf = palm_motion
                            feature_window.clear()
                            last_action_label = None

            else:
                feature_window.clear()
                current_label = "none"
                current_conf = 0.0
                last_action_label = None
                prev_scroll_y = None
                scroll_accum = 0.0
                prev_palm_x = None
                prev_palm_y = None

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            cv2.putText(frame, f"FPS: {int(fps)}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Open-palm move: {'ON' if move_mode else 'OFF'}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(frame, f"Fist scroll: {'ON' if fist_mode else 'OFF'}", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)

            cv2.putText(frame, f"Cmd: {current_label}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.putText(frame, f"Conf/Motion: {current_conf:.2f}", (20, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.putText(frame, "Q-quit | Z/X/C-clicks | V/B-spaces", (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.imshow("Virtual Mouse", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("z"):
                print("MANUAL LEFT CLICK", flush=True)
                mouse.click(Button.left, 1)
            elif key == ord("x"):
                print("MANUAL RIGHT CLICK", flush=True)
                mouse.click(Button.right, 1)
            elif key == ord("c"):
                print("MANUAL DOUBLE CLICK", flush=True)
                mouse.click(Button.left, 2)
            elif key == ord("v"):
                print("MANUAL SPACE LEFT", flush=True)
                swipe_space("left")
            elif key == ord("b"):
                print("MANUAL SPACE RIGHT", flush=True)
                swipe_space("right")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()