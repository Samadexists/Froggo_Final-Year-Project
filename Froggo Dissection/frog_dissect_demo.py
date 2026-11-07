import random
import sys
from pathlib import Path

from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (QBrush, QColor, QFont, QPainter, QPainterPath, QPen,
                           QPixmap, QRadialGradient)
from PySide6.QtWidgets import (QApplication, QCheckBox, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QVBoxLayout, QWidget)


IMG_W, IMG_H = 1000, 1400
CX, CY, RX, RY = 500, 600, 165, 285
CUT_TOP, CUT_BOTTOM = 330, 870
CUT_SECONDS = 2.5
OPEN_SECONDS = 1.8
PAUSE_SECONDS = 0.4


def smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


class FrogCanvas(QWidget):
    """Draws the frog, the incision, the opening flaps, organs and blood."""

    def __init__(self, pixmap: QPixmap, status_cb):
        super().__init__()
        self.pix = pixmap
        self.status_cb = status_cb
        self.setMinimumSize(500, 700)
        self.setMouseTracking(True)

        self.show_blood = True
        self.show_labels = True

        belly = QPainterPath()
        belly.addEllipse(QRectF(CX - RX, CY - RY, 2 * RX, 2 * RY))
        self.belly = belly
        left = QPainterPath()
        left.addRect(QRectF(0, 0, CX, IMG_H))
        right = QPainterPath()
        right.addRect(QRectF(CX, 0, IMG_W - CX, IMG_H))
        self.left_flap = belly.intersected(left)
        self.right_flap = belly.intersected(right)
        whole = QPainterPath()
        whole.addRect(QRectF(0, 0, IMG_W, IMG_H))
        self.outside = whole.subtracted(belly)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.clock = QElapsedTimer()
        self.reset()
        self.timer.start(16)
        self.clock.start()


    def reset(self):
        self.state = "idle"
        self.cut = 0.0
        self.open = 0.0
        self.pause = 0.0
        self.drops = []
        self.status_cb("Ready - press Start or drag down the middle of the belly")
        self.update()

    def start_auto(self):
        if self.state in ("idle", "manual"):
            self.state = "cutting"
            self.status_cb("Cutting along the midline...")


    def tick(self):
        dt = self.clock.restart() / 1000.0
        dt = min(dt, 0.05)

        if self.state == "cutting":
            self.cut = min(1.0, self.cut + dt / CUT_SECONDS)
            if self.cut >= 1.0:
                self.finish_cut()
        elif self.state == "pause":
            self.pause += dt
            if self.pause >= PAUSE_SECONDS:
                self.state = "opening"
                self.status_cb("Opening the skin flaps...")
        elif self.state == "opening":
            self.open = min(1.0, self.open + dt / OPEN_SECONDS)
            if self.open >= 1.0:
                self.state = "done"
                self.status_cb("Body cavity exposed. Press Reset to try again.")

        self.update_blood(dt)
        self.update()

    def finish_cut(self):
        self.cut = 1.0
        self.state = "pause"
        self.pause = 0.0
        self.status_cb("Incision complete")

    def tip_y(self):
        return CUT_TOP + (CUT_BOTTOM - CUT_TOP) * self.cut

    def update_blood(self, dt):
        cutting_now = self.state in ("cutting", "manual") and self.cut > 0
        if self.show_blood and cutting_now and random.random() < 0.55:
            self.spawn_drop(self.tip_y())

        if self.show_blood and self.state == "pause" and random.random() < 0.3:
            self.spawn_drop(random.uniform(CUT_TOP, CUT_BOTTOM))

        alive = []
        for d in self.drops:
            d["age"] += dt
            if d["age"] < d["life"]:
                d["x"] += d["vx"] * dt
                d["y"] += d["vy"] * dt
                d["vx"] *= 0.96
                alive.append(d)
        self.drops = alive

    def spawn_drop(self, y):
        self.drops.append({
            "x": CX + random.uniform(-4, 4), "y": y,
            "vx": random.uniform(-28, 28), "vy": random.uniform(8, 35),
            "r": random.uniform(2.5, 6.0), "age": 0.0,
            "life": random.uniform(1.2, 2.6),
        })


    def to_image(self, pos):
        s, ox, oy = self.view_transform()
        return QPointF((pos.x() - ox) / s, (pos.y() - oy) / s)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton):
            return
        if self.state not in ("idle", "manual"):
            return
        p = self.to_image(event.position())
        on_midline = abs(p.x() - CX) < 60
        if on_midline and CUT_TOP - 30 <= p.y() <= CUT_BOTTOM + 30:
            progress = (p.y() - CUT_TOP) / (CUT_BOTTOM - CUT_TOP)
            if progress > self.cut:
                self.cut = max(0.0, min(1.0, progress))
                if self.state == "idle":
                    self.state = "manual"
                    self.status_cb("Cutting by hand...")
            if self.cut >= 0.98:
                self.finish_cut()


    def view_transform(self):
        s = min(self.width() / IMG_W, self.height() / IMG_H)
        ox = (self.width() - IMG_W * s) / 2
        oy = (self.height() - IMG_H * s) / 2
        return s, ox, oy

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(236, 232, 222))

        s, ox, oy = self.view_transform()
        p.translate(ox, oy)
        p.scale(s, s)

        e = smooth(self.open)
        if e <= 0:
            p.drawPixmap(0, 0, self.pix)
        else:
            # frog outside the belly stays as it is
            p.save()
            p.setClipPath(self.outside)
            p.drawPixmap(0, 0, self.pix)
            p.restore()

            p.save()
            p.setClipPath(self.belly)
            self.draw_organs(p)
            p.restore()

            self.draw_flap(p, self.left_flap, CX - RX, e)
            self.draw_flap(p, self.right_flap, CX + RX, e)

        if e <= 0 and self.cut > 0:
            self.draw_cut(p)

        if self.show_blood:
            self.draw_blood(p, fade=1.0 - e)

        if self.show_labels and self.open > 0.9:
            self.draw_labels(p)

        p.end()

    def draw_flap(self, p, flap, hinge_x, e):
        p.save()
        p.translate(hinge_x, 0)
        p.scale(1.0 - 0.88 * e, 1.0)
        p.translate(-hinge_x, 0)
        p.setClipPath(flap)
        p.drawPixmap(0, 0, self.pix)

        p.fillPath(flap, QColor(170, 80, 80, int(120 * e)))
        p.setPen(QPen(QColor(140, 20, 20), 5))
        p.drawLine(QPointF(CX, CY - RY), QPointF(CX, CY + RY))
        p.restore()

    def draw_cut(self, p):
        y = self.tip_y()
        start, end = QPointF(CX, CUT_TOP), QPointF(CX, y)
        glow = QPen(QColor(170, 0, 0, 90), 12, Qt.SolidLine, Qt.RoundCap)
        p.setPen(glow)
        p.drawLine(start, end)
        line = QPen(QColor(140, 10, 10), 4, Qt.SolidLine, Qt.RoundCap)
        p.setPen(line)
        p.drawLine(start, end)
        if self.state in ("cutting", "manual"):
            # scalpel tip
            p.setPen(QPen(QColor(90, 90, 100), 2))
            p.setBrush(QBrush(QColor(210, 214, 222)))
            p.drawEllipse(end, 10, 10)

    def draw_blood(self, p, fade):
        p.setPen(Qt.NoPen)
        for d in self.drops:
            a = int(210 * (1 - d["age"] / d["life"]) * fade)
            if a <= 0:
                continue
            p.setBrush(QColor(125, 8, 12, a))
            p.drawEllipse(QPointF(d["x"], d["y"]), d["r"], d["r"] * 1.25)

    def draw_organs(self, p):
        cavity = QRadialGradient(QPointF(CX, CY), RY)
        cavity.setColorAt(0.0, QColor(128, 46, 40))
        cavity.setColorAt(1.0, QColor(70, 20, 20))
        p.fillRect(QRectF(CX - RX, CY - RY, 2 * RX, 2 * RY), QBrush(cavity))
        p.setPen(QPen(QColor(60, 20, 20, 120), 2))

        def blob(x, y, rx, ry, colour):
            p.setBrush(QColor(*colour))
            p.drawEllipse(QPointF(x, y), rx, ry)

        blob(432, 400, 44, 62, (226, 150, 152))
        blob(568, 400, 44, 62, (226, 150, 152))
        blob(500, 392, 36, 46, (186, 30, 42))
        blob(448, 478, 84, 50, (124, 54, 40))
        blob(558, 472, 80, 48, (124, 54, 40))
        blob(506, 502, 13, 17, (92, 132, 60))

        p.save()
        p.translate(408, 572)
        p.rotate(-25)
        blob(0, 0, 32, 76, (222, 172, 150))
        p.restore()

        blob(395, 700, 26, 18, (238, 200, 90))
        blob(605, 700, 26, 18, (238, 200, 90))

        coil = QPainterPath(QPointF(560, 600))
        rows = [(440, 630), (575, 665), (430, 700), (570, 735), (450, 770)]
        for x, y in rows:
            coil.lineTo(x, y)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(200, 130, 125), 30, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(coil)
        p.setPen(QPen(QColor(240, 188, 178), 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(coil)

        p.setPen(QPen(QColor(60, 20, 20, 120), 2))
        blob(500, 822, 48, 30, (196, 120, 108))

    def draw_labels(self, p):
        labels = [
            ("Heart", 500, 392), ("Lung", 432, 340), ("Lung", 568, 340),
            ("Liver", 448, 478), ("Liver", 558, 472), ("Stomach", 408, 572),
            ("Small intestine", 505, 695), ("Large intestine", 500, 822),
            ("Fat body", 395, 735), ("Fat body", 605, 735),
        ]
        font = QFont("Arial", 15)
        font.setBold(True)
        p.setFont(font)
        for text, x, y in labels:
            rect = QRectF(x - 90, y - 12, 180, 24)
            p.setPen(QColor(0, 0, 0, 170))
            p.drawText(rect.translated(1.5, 1.5), Qt.AlignCenter, text)
            p.setPen(QColor(255, 255, 255))
            p.drawText(rect, Qt.AlignCenter, text)


class MainWindow(QWidget):
    def __init__(self, pixmap):
        super().__init__()
        self.setWindowTitle("Froggo - Dissection demo")

        self.status = QLabel()
        self.canvas = FrogCanvas(pixmap, self.status.setText)

        start = QPushButton("Start dissection")
        start.clicked.connect(self.canvas.start_auto)
        reset = QPushButton("Reset")
        reset.clicked.connect(self.canvas.reset)
        blood = QCheckBox("Blood effect")
        blood.setChecked(True)
        blood.toggled.connect(lambda on: setattr(self.canvas, "show_blood", on))
        labels = QCheckBox("Organ labels")
        labels.setChecked(True)
        labels.toggled.connect(lambda on: setattr(self.canvas, "show_labels", on))

        row = QHBoxLayout()
        for w in (start, reset, blood, labels):
            row.addWidget(w)
        row.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(row)
        layout.addWidget(self.status)
        self.resize(620, 960)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.canvas.start_auto()
        elif event.key() == Qt.Key_R:
            self.canvas.reset()


def main():
    app = QApplication(sys.argv)
    image_path = Path(__file__).with_name("frog.png")
    pixmap = QPixmap(str(image_path))
    if pixmap.isNull():
        QMessageBox.critical(None, "Froggo", f"Could not load {image_path}\nPut frog.png next to this script.")
        sys.exit(1)
    if pixmap.width() != IMG_W or pixmap.height() != IMG_H:
        pixmap = pixmap.scaled(IMG_W, IMG_H, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    window = MainWindow(pixmap)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()