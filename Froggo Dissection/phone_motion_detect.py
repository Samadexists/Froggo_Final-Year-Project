import argparse
import sys
import time

import cv2
import numpy as np


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def open_camera(source):
    print(f"Opening camera source: {source} ...", flush=True)
    if isinstance(source, int) and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    print("Camera opened." if cap.isOpened() else "Camera did not open.", flush=True)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera source: {source}\n"
            "- Webcam apps: try --source 0, 1 or 2\n"
            "- IP Webcam: check the URL ends with /video and both devices are on the same Wi-Fi"
        )
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def motion_mask(frame, subtractor):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    mask = subtractor.apply(blurred)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    return mask


def skin_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def largest_blob(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def main():
    parser = argparse.ArgumentParser(description="Phone camera hand movement test")
    parser.add_argument("--source", default="0", help="Webcam index (0, 1, 2) or stream URL")
    parser.add_argument("--width", type=int, default=640, help="Processing width in pixels")
    parser.add_argument("--min-area", type=int, default=1500, help="Smallest blob counted as a hand")
    args = parser.parse_args()

    cap = open_camera(parse_source(args.source))
    subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
    mode = "motion"
    trail = []
    prev_time = time.time()
    failed_reads = 0

    print("Running. Keys: q = quit, r = reset background, m = switch mode")

    while True:
        ok, frame = cap.read()
        if not ok:
            failed_reads += 1
            if failed_reads > 30:
                print("Lost the camera stream. Check the phone app and Wi-Fi.")
                break
            continue
        failed_reads = 0

        h, w = frame.shape[:2]
        scale = args.width / w
        frame = cv2.resize(frame, (args.width, int(h * scale)))

        mask = motion_mask(frame, subtractor) if mode == "motion" else skin_mask(frame)
        mask = clean_mask(mask)
        blob = largest_blob(mask, args.min_area)

        if blob is not None:
            x, y, bw, bh = cv2.boundingRect(blob)
            cx, cy = x + bw // 2, y + bh // 2
            trail.append((cx, cy))
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, "Hand movement", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        trail = trail[-30:]

        for i in range(1, len(trail)):
            cv2.line(frame, trail[i - 1], trail[i], (255, 0, 0), 2)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"Mode: {mode}  FPS: {fps:.0f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Camera", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
            trail.clear()
            print("Background reset - keep your hand out of view for a moment.")
        if key == ord("m"):
            mode = "skin" if mode == "motion" else "motion"
            trail.clear()
            print(f"Mode switched to: {mode}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()