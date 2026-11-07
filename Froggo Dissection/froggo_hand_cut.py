import argparse
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QThread, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QVBoxLayout, QWidget)

import frog_dissect_demo as demo
import phone_motion_detect as cam

GAIN_X = 700
GAIN_Y = 1100
MIDLINE_TOLERANCE = 110
START_ZONE = 0.20
MAX_STEP = 0.04
FOLLOW_SPEED = 22.0
BLUE_LOW = (100, 120, 70)
BLUE_HIGH = (128, 255, 255)
HUE_RANGE = 8
MIN_SAT = 120
MIN_VAL = 70
BLUE_HUE_MIN = 95
BLUE_HUE_MAX = 130
MIN_SOLIDITY = 0.65
SEARCH_RADIUS = 0.18
CAP_MIN_AREA = 80
CAP_MAX_AREA = 15000

START_X = demo.CX
START_Y = demo.CUT_TOP - 35
ANCHOR_FRAMES = 4
ANCHOR_STILL = 0.04
START_DEADZONE = 0.06
ARM_SECONDS = 0.3
DRIFT_FOLLOW = 0.02

FILTER_MIN_CUTOFF = 1.2
FILTER_BETA = 0.02
JUMP_LIMIT = 0.22
JUMP_CONFIRM = 3
LOST_GRACE = 8


class OneEuro:

    def __init__(self, min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = 0.0
        self.t = None

    @staticmethod
    def alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x, self.dx, self.t = None, 0.0, None

    def __call__(self, value, t):
        if self.x is None:
            self.x, self.t = value, t
            return value
        dt = max(t - self.t, 1e-3)
        self.t = t
        dx = (value - self.x) / dt
        a_d = self.alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * abs(self.dx) * 100
        a = self.alpha(cutoff, dt)
        self.x = a * value + (1 - a) * self.x
        return self.x


class LatestFrame:

    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ok = True
        self.fails = 0
        self.lock = threading.Lock()
        self.new = threading.Event()
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.fails += 1
                if self.fails > 60:
                    self.ok = False
                    self.new.set()
                    return
                time.sleep(0.01)
                continue
            self.fails = 0
            with self.lock:
                self.frame = frame
            self.new.set()

    def read(self, timeout=1.0):
        if not self.new.wait(timeout):
            return None
        self.new.clear()
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.thread.join(1.0)


def color_mask(frame, target):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if target is None:
        return cv2.inRange(hsv, np.array(BLUE_LOW, np.uint8), np.array(BLUE_HIGH, np.uint8))
    h, s, v = target
    s_low, v_low = max(int(s) - 60, MIN_SAT), max(int(v) - 80, MIN_VAL)
    lo = max(int(h) - HUE_RANGE, BLUE_HUE_MIN)
    hi = min(int(h) + HUE_RANGE, BLUE_HUE_MAX)
    return cv2.inRange(hsv, np.array([lo, s_low, v_low], np.uint8),
                       np.array([hi, 255, 255], np.uint8))


def is_blue(hsv_colour):
    h, s, v = (float(x) for x in hsv_colour)
    return BLUE_HUE_MIN <= h <= BLUE_HUE_MAX and s >= MIN_SAT and v >= MIN_VAL


def find_cap(mask, last, min_area, max_area, fw, fh, hsv=None):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, None
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull <= 0 or area / hull < MIN_SOLIDITY:
            continue
        if hsv is not None:
            x, y, bw, bh = cv2.boundingRect(c)
            inside = np.zeros((bh, bw), np.uint8)
            cv2.drawContours(inside, [c - (x, y)], -1, 255, -1)
            avg = cv2.mean(hsv[y:y + bh, x:x + bw], mask=inside)[:3]
            if not is_blue(avg):
                continue
        m = cv2.moments(c)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        if last is not None:
            score = math.hypot(cx / fw - last[0], cy / fh - last[1])
        else:
            score = -area
        if best_score is None or score < best_score:
            best, best_score = c, score
    return best


class CameraThread(QThread):
    hand = Signal(float, float)
    lost = Signal()
    preview = Signal(QImage)
    error = Signal(str)
    note = Signal(str)

    def __init__(self, source, mirror, min_area):
        super().__init__()
        self.source = source
        self.mirror = mirror
        self.min_area = min_area
        self.running = True
        self.mode = "color"
        self.pick = None
        self.target = None
        self.fx, self.fy = OneEuro(), OneEuro()
        self.last = None
        self.jump_count = 0
        self.missed = 0
        self.area = None
        self.show_mask = False

    def run(self):
        try:
            cap = cam.open_camera(self.source)
        except SystemExit as exc:
            self.error.emit(str(exc))
            return

        grabber = LatestFrame(cap)
        while self.running:
            frame = grabber.read()
            if not grabber.ok:
                self.error.emit("Camera stream stopped. Check the phone app and Wi-Fi.")
                break
            if frame is None:
                continue

            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
            if self.mirror:
                frame = cv2.flip(frame, 1)

            fh, fw = frame.shape[:2]
            if self.pick is not None:
                px = min(max(int(self.pick[0] * fw), 4), fw - 5)
                py = min(max(int(self.pick[1] * fh), 4), fh - 5)
                hsv = cv2.cvtColor(frame[py - 4:py + 5, px - 4:px + 5], cv2.COLOR_BGR2HSV)
                picked = np.median(hsv.reshape(-1, 3), axis=0)
                self.pick = None
                if is_blue(picked):
                    self.target = picked
                    self.mode = "color"
                    self.forget()
                    self.note.emit("Cap colour saved - hold the cap still for a moment, then move down")
                else:
                    self.note.emit("That is not blue - click directly on the blue cap")

            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cam.clean_mask(color_mask(frame, self.target))

            if self.area is not None:
                x1, y1, x2, y2 = self.area
                keep = np.zeros_like(mask)
                keep[int(y1 * fh):int(y2 * fh), int(x1 * fw):int(x2 * fw)] = 255
                mask &= keep
            if self.last is not None and self.missed < LOST_GRACE:
                near = np.zeros_like(mask)
                cv2.circle(near, (int(self.last[0] * fw), int(self.last[1] * fh)),
                           int(SEARCH_RADIUS * fw), 255, -1)
                mask &= near

            blob = find_cap(mask, self.last, CAP_MIN_AREA, CAP_MAX_AREA, fw, fh, frame_hsv)

            tip = None
            if blob is not None:
                m = cv2.moments(blob)
                tip = (m["m10"] / m["m00"], m["m01"] / m["m00"])
                x, y, bw, bh = cv2.boundingRect(blob)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.circle(frame, (int(tip[0]), int(tip[1])), 7, (0, 0, 255), -1)
            self.handle_tip(tip, fw, fh)

            if self.show_mask:
                frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            if self.area is not None:
                x1, y1, x2, y2 = self.area
                cv2.rectangle(frame, (int(x1 * fw), int(y1 * fh)), (int(x2 * fw), int(y2 * fh)),
                              (255, 200, 0), 2)
            label = "Blue marker" if self.target is None else "Blue marker (clicked colour)"
            cv2.putText(frame, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(rgb.data, fw, fh, 3 * fw, QImage.Format_RGB888).copy()
            self.preview.emit(image)

        grabber.stop()
        cap.release()

    def forget(self):
        self.fx.reset()
        self.fy.reset()
        self.last = None
        self.jump_count = 0

    def handle_tip(self, tip, fw, fh):
        if tip is None:
            self.missed += 1
            if self.missed == LOST_GRACE:
                self.forget()
                self.lost.emit()
            return
        self.missed = 0
        nx, ny = tip[0] / fw, tip[1] / fh
        if self.last is not None and math.hypot(nx - self.last[0], ny - self.last[1]) > JUMP_LIMIT:
            self.jump_count += 1
            if self.jump_count < JUMP_CONFIRM:
                return
            self.fx.reset()
            self.fy.reset()
        self.jump_count = 0
        self.last = (nx, ny)
        t = time.monotonic()
        self.hand.emit(self.fx(nx, t), self.fy(ny, t))

    def stop(self):
        self.running = False
        self.wait(2000)


class HandCanvas(demo.FrogCanvas):

    def __init__(self, pixmap, status_cb):
        super().__init__(pixmap, status_cb)

    def reset(self):
        super().reset()
        self.calibrate()
        self.status_cb("Ready - bring the cap into view and hold it still; the cut starts above the chest")

    def calibrate(self):
        self.anchor = None
        self.settle = []
        self.started = False
        self.lock_time = None
        self.status_ready = False
        self.hand_pt = None
        self.hand_target = None

    def armed(self):
        return self.lock_time is not None and time.monotonic() - self.lock_time >= ARM_SECONDS

    def hand_at(self, nx, ny):
        if self.anchor is None:
            self.settle = (self.settle + [(nx, ny)])[-ANCHOR_FRAMES:]
            if len(self.settle) < ANCHOR_FRAMES:
                return
            mx = sum(p[0] for p in self.settle) / ANCHOR_FRAMES
            my = sum(p[1] for p in self.settle) / ANCHOR_FRAMES
            if max(math.hypot(p[0] - mx, p[1] - my) for p in self.settle) > ANCHOR_STILL:
                return
            self.anchor = (mx, my)
            self.lock_time = time.monotonic()
            self.status_cb("Start locked - keep your hand steady...")

        if not self.started:
            self.hand_target = QPointF(START_X, START_Y)
            self.hand_pt = QPointF(START_X, START_Y)
            if not self.armed():
                self.anchor = (nx, ny)
                return
            if self.status_ready is False:
                self.status_ready = True
                self.status_cb("Ready (green) - now move slowly straight down")
            dy = ny - self.anchor[1]
            if dy < START_DEADZONE:
                if dy < 0:
                    self.anchor = (nx, ny)
                else:
                    self.anchor = (nx, self.anchor[1] + DRIFT_FOLLOW * dy)
                return
            self.started = True
            # start the pointer right at the top of the cut line, no dead travel
            self.anchor = (nx, ny - (demo.CUT_TOP - START_Y) / GAIN_Y)
            self.status_cb("Cutting by hand...")
        tx = START_X + (nx - self.anchor[0]) * GAIN_X
        ty = START_Y + (ny - self.anchor[1]) * GAIN_Y
        tx = max(0.0, min(float(demo.IMG_W), tx))
        ty = max(0.0, min(float(demo.IMG_H), ty))
        self.hand_target = QPointF(tx, ty)
        if self.hand_pt is None:
            self.hand_pt = QPointF(tx, ty)

    def hand_lost(self):
        self.hand_target = None
        self.hand_pt = None
        self.settle = []
        if self.state == "idle" or not self.started:
            self.anchor = None
            self.started = False
            self.lock_time = None
            self.status_ready = False

    def tick(self):
        if self.hand_target is not None and self.hand_pt is not None and self.started:
            a = 1.0 - math.exp(-FOLLOW_SPEED * 0.016)
            self.hand_pt = QPointF(self.hand_pt.x() + a * (self.hand_target.x() - self.hand_pt.x()),
                                   self.hand_pt.y() + a * (self.hand_target.y() - self.hand_pt.y()))
            self.cut_towards(self.hand_pt)
        super().tick()

    def cut_towards(self, p):
        if self.state not in ("idle", "manual"):
            return
        on_midline = abs(p.x() - demo.CX) < MIDLINE_TOLERANCE
        if not (on_midline and min(START_Y, demo.CUT_TOP) - 40 <= p.y() <= demo.CUT_BOTTOM + 40):
            return
        progress = max(0.0, min(1.0, (p.y() - demo.CUT_TOP) / (demo.CUT_BOTTOM - demo.CUT_TOP)))
        if self.state == "idle":
            if progress > START_ZONE:
                return
            self.state = "manual"
            self.status_cb("Cutting by hand...")
        if progress > self.cut:
            self.cut = min(progress, self.cut + MAX_STEP)
        if self.cut >= 0.98:
            self.finish_cut()

    def paintEvent(self, event):
        super().paintEvent(event)
        s, ox, oy = self.view_transform()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        if self.state == "idle":
            pen = QPen(QColor(0, 120, 0, 140), 2, Qt.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(ox + demo.CX * s, oy + demo.CUT_TOP * s),
                       QPointF(ox + demo.CX * s, oy + demo.CUT_BOTTOM * s))

        if self.state == "idle" and self.anchor is None:
            c = QPointF(ox + START_X * s, oy + START_Y * s)
            p.setPen(QPen(QColor(0, 120, 220, 200), 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(c, 16, 16)
            p.drawText(QPointF(c.x() + 22, c.y() + 5),
                       "Hold the cap still..." if self.settle else "Start")

        pt = self.hand_pt
        if pt is not None:
            on_midline = abs(pt.x() - demo.CX) < MIDLINE_TOLERANCE
            if not self.started and not self.armed():
                colour = QColor(0, 120, 220)
            else:
                colour = QColor(0, 170, 0) if on_midline else QColor(230, 140, 0)
            p.setPen(QPen(colour, 3))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(ox + pt.x() * s, oy + pt.y() * s), 14, 14)
        p.end()


class ClickLabel(QLabel):
    clicked = Signal(float, float)
    area = Signal(float, float, float, float)

    def __init__(self, *args):
        super().__init__(*args)
        self.start = None

    def to_norm(self, event):
        pix = self.pixmap()
        if pix is None or pix.isNull():
            return None
        ox = (self.width() - pix.width()) / 2
        oy = (self.height() - pix.height()) / 2
        nx = (event.position().x() - ox) / pix.width()
        ny = (event.position().y() - oy) / pix.height()
        return (min(max(nx, 0.0), 1.0), min(max(ny, 0.0), 1.0))

    def mousePressEvent(self, event):
        self.start = self.to_norm(event)

    def mouseReleaseEvent(self, event):
        end = self.to_norm(event)
        if self.start is None or end is None:
            return
        (ax, ay), (bx, by) = self.start, end
        self.start = None
        if abs(bx - ax) < 0.03 and abs(by - ay) < 0.03:
            self.clicked.emit(ax, ay)
        else:
            self.area.emit(min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))


class HandWindow(QWidget):
    def __init__(self, pixmap, args):
        super().__init__()
        self.setWindowTitle("Froggo - Hand dissection")

        self.status = QLabel()
        self.canvas = HandCanvas(pixmap, self.status.setText)

        self.preview = ClickLabel("Opening camera...")
        self.preview.setFixedSize(320, 240)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background: #222; color: #ddd;")

        hint = QLabel("Hold the blue cap still for a moment,\n"
                      "when it turns green, move slowly down.")

        start = QPushButton("Start dissection (auto)")
        start.clicked.connect(self.canvas.start_auto)
        reset = QPushButton("Reset")
        reset.clicked.connect(self.canvas.reset)

        side = QVBoxLayout()
        side.addWidget(self.preview)
        side.addWidget(hint)
        for w in (start, reset):
            side.addWidget(w)
        side.addStretch()

        row = QHBoxLayout()
        row.addWidget(self.canvas, 1)
        row.addLayout(side)

        layout = QVBoxLayout(self)
        layout.addLayout(row, 1)
        layout.addWidget(self.status)
        self.resize(1000, 900)

        self.cam = CameraThread(cam.parse_source(args.source), args.mirror, args.min_area)
        self.cam.hand.connect(self.canvas.hand_at)
        self.cam.lost.connect(self.canvas.hand_lost)
        self.cam.preview.connect(self.show_preview)
        self.cam.error.connect(self.camera_error)
        self.cam.note.connect(self.status.setText)
        self.preview.clicked.connect(self.pick_colour)
        self.preview.area.connect(self.set_area)
        self.cam.start()

    def show_preview(self, image):
        pix = QPixmap.fromImage(image).scaled(self.preview.size(), Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation)
        self.preview.setPixmap(pix)

    def camera_error(self, message):
        self.status.setText(message)
        QMessageBox.warning(self, "Froggo", message)

    def pick_colour(self, nx, ny):
        self.canvas.calibrate()
        self.cam.pick = (nx, ny)

    def set_area(self, x1, y1, x2, y2):
        self.cam.area = (x1, y1, x2, y2)
        self.cam.forget()
        self.canvas.calibrate()
        self.status.setText("Work area set - nothing outside it will be detected (A = clear)")

    def clear_area(self):
        self.cam.area = None
        self.status.setText("Area cleared - using the whole frame")

    def toggle_mask(self):
        self.cam.show_mask = not self.cam.show_mask

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            self.canvas.start_auto()
        elif key == Qt.Key_R:
            self.canvas.reset()
        elif key == Qt.Key_A:
            self.clear_area()
        elif key == Qt.Key_V:
            self.toggle_mask()

    def closeEvent(self, event):
        self.cam.stop()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Froggo: cut the frog by hand")
    parser.add_argument("--source", default="0", help="Webcam number (0, 1, 2) or the phone's /video link")
    parser.add_argument("--min-area", type=int, default=1500, help="(old, no longer used)")
    parser.add_argument("--mirror", action="store_true", help="Mirror the camera picture")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    image_path = Path(__file__).with_name("frog.png")
    pixmap = QPixmap(str(image_path))
    if pixmap.isNull():
        QMessageBox.critical(None, "Froggo", f"Could not load {image_path}\nPut frog.png in the same folder.")
        sys.exit(1)
    if pixmap.width() != demo.IMG_W or pixmap.height() != demo.IMG_H:
        pixmap = pixmap.scaled(demo.IMG_W, demo.IMG_H, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    window = HandWindow(pixmap, args)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()