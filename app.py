import cv2
import numpy as np
import sys
import subprocess
import re
import select
import threading
import os
import shutil
import time
import json
from collections import deque
from datetime import datetime
import pystray
from PIL import Image, ImageDraw

from fpn_correction import FPNCorrector
from denoise import ThermalDenoiser

# macOS .app bundles do not inherit the shell PATH. Add common Homebrew paths:
os.environ['PATH'] += os.pathsep + '/opt/homebrew/bin' + os.pathsep + '/usr/local/bin'
FFMPEG_PATH = shutil.which('ffmpeg') or 'ffmpeg'

PALETTES = []
if hasattr(cv2, 'COLORMAP_INFERNO'): PALETTES.append(("Inferno", cv2.COLORMAP_INFERNO))
PALETTES.append(("Jet", cv2.COLORMAP_JET))
PALETTES.append(("Hot", cv2.COLORMAP_HOT))
if hasattr(cv2, 'COLORMAP_PLASMA'): PALETTES.append(("Plasma", cv2.COLORMAP_PLASMA))
if hasattr(cv2, 'COLORMAP_MAGMA'): PALETTES.append(("Magma", cv2.COLORMAP_MAGMA))
if hasattr(cv2, 'COLORMAP_VIRIDIS'): PALETTES.append(("Viridis", cv2.COLORMAP_VIRIDIS))
PALETTES.append(("Rainbow", cv2.COLORMAP_RAINBOW))
PALETTES.append(("Bone", cv2.COLORMAP_BONE))
PALETTES.append(("Ocean", cv2.COLORMAP_OCEAN))

# Key codes: Latin letters (lowercase + uppercase) and common arrow/special codes
KEY_QUIT = {ord('q'), ord('Q')}
KEY_PALETTE = {ord('p'), ord('P')}
KEY_MINMAX = {ord('m'), ord('M')}
KEY_DEBUG = {ord('d'), ord('D')}
KEY_SPACE = {32}
KEY_SNAPSHOT = {ord('s'), ord('S')}
KEY_FAHRENHEIT = {ord('f'), ord('F')}
KEY_PALETTE_1 = {ord('1')}
KEY_PALETTE_2 = {ord('2')}
KEY_PALETTE_3 = {ord('3')}
KEY_PALETTE_4 = {ord('4')}
KEY_PALETTE_5 = {ord('5')}
KEY_PALETTE_6 = {ord('6')}
KEY_PALETTE_7 = {ord('7')}
KEY_PALETTE_8 = {ord('8')}
KEY_PALETTE_9 = {ord('9')}
KEY_RANGE_CYCLE = {ord('0')}
KEY_ROI = {ord('r'), ord('R')}
KEY_ALARM = {ord('a'), ord('A')}
KEY_HQ = {ord('h'), ord('H')}
KEY_BRACKET_L = {ord('[')}
KEY_BRACKET_R = {ord(']')}
KEY_SHARP_MINUS = {ord('-')}
KEY_SHARP_PLUS = {ord('='), ord('+')}
KEY_HELP = {ord('i'), ord('I')}
KEY_FPN = {ord('c'), ord('C')}
KEY_FPN_ONE = {ord('x'), ord('X')}
KEY_LINE_PROFILE = {ord('l'), ord('L')}
KEY_ISOTHERM = {ord('t'), ord('T')}
KEY_DELTA_T = {ord('v'), ord('V')}
KEY_TREND = {ord('w'), ord('W')}
KEY_HISTOGRAM = {ord('b'), ord('B')}
KEY_ANOMALY = {ord('n'), ord('N')}
KEY_PALETTE_INVERT = {ord('z'), ord('Z')}
KEY_DENOISE = {ord('g'), ord('G')}
# Arrow keys: macOS 63232-63235, GTK 65361-65364, some 0-3 or 81-84
KEY_ARROW_LEFT = {63234, 65361, 2, 84}
KEY_ARROW_RIGHT = {63235, 65363, 3, 82}
KEY_ARROW_UP = {63232, 65362, 0, 81}
KEY_ARROW_DOWN = {63233, 65364, 1, 83}

# --- Bottom toolbar: two rows of clickable icon-buttons ---
TOOLBAR_H = 44
_TB_ROW_H = 20
_TB_PAD = 2

_TOOLBAR_DEFS = [
    # Row 0: Display & Measurement
    ('palette', 'PAL'),  ('invert',  'INV'),  ('range',   'RNG'),  ('minmax',  'M/M'),
    ('freeze',  'FRZ'),  ('snap',    'SNAP'), ('fahr',    'F/C'),
    ('roi',     'ROI'),  ('line',    'LINE'), ('iso',     'ISO'),
    ('hist',    'HIST'), ('trend',   'TRND'), ('anomaly', 'ANOM'),
    ('delta',   'dT'),   ('alarm',   'ALM'),
    # Row 1: Enhancement & Control
    ('dde',     'DDE'),  ('cont_dn', 'C-'),   ('cont_up', 'C+'),
    ('shrp_dn', 'S-'),   ('shrp_up', 'S+'),   ('fpn',     'FPN'),
    ('fpn1',    '1FPN'), ('dnz',     'DNZ'),  ('rot_l',   'RoL'),  ('rot_r',   'RoR'),
    ('flip_v',  'FlV'),  ('flip_h',  'FlH'),  ('debug',   'DBG'),
    ('help',    'HLP'),  ('quit',    'QUIT'),
]
_TB_ROW0 = 15

_TOOLBAR_KEY_MAP = {
    'palette': ord('p'), 'invert': ord('z'), 'range': ord('0'), 'minmax': ord('m'),
    'freeze': 32, 'snap': ord('s'), 'fahr': ord('f'),
    'roi': ord('r'), 'line': ord('l'), 'iso': ord('t'),
    'hist': ord('b'), 'trend': ord('w'), 'anomaly': ord('n'),
    'delta': ord('v'), 'alarm': ord('a'),
    'dde': ord('h'), 'cont_dn': ord('['), 'cont_up': ord(']'),
    'shrp_dn': ord('-'), 'shrp_up': ord('='), 'fpn': ord('c'),
    'fpn1': ord('x'), 'dnz': ord('g'), 'rot_l': 63234, 'rot_r': 63235,
    'flip_v': 63232, 'flip_h': 63233, 'debug': ord('d'),
    'help': ord('i'), 'quit': ord('q'),
}

_TOOLBAR_TOGGLES = {
    'minmax', 'freeze', 'fahr', 'invert', 'roi', 'line', 'iso', 'hist',
    'trend', 'anomaly', 'delta', 'alarm', 'dde', 'fpn', 'fpn1', 'dnz',
    'flip_v', 'flip_h', 'debug', 'help',
}


def _toolbar_find_button(x, y_rel, dw):
    """Return button id at (x, y_rel) where y_rel is relative to toolbar top."""
    if y_rel < _TB_PAD or y_rel >= TOOLBAR_H:
        return None
    row_idx = min((y_rel - _TB_PAD) // _TB_ROW_H, 1)
    row = _TOOLBAR_DEFS[:_TB_ROW0] if row_idx == 0 else _TOOLBAR_DEFS[_TB_ROW0:]
    if not row:
        return None
    btn_w = dw / len(row)
    idx = int(x / btn_w)
    if 0 <= idx < len(row):
        return row[idx][0]
    return None


def draw_toolbar(img, y0, dw, states, hover_id=None):
    """Draw 2-row icon toolbar on *img* starting at row *y0*."""
    img[y0:y0 + TOOLBAR_H, 0:dw] = (30, 30, 30)
    cv2.line(img, (0, y0), (dw, y0), (80, 80, 80), 1)

    for row_idx in range(2):
        row = _TOOLBAR_DEFS[:_TB_ROW0] if row_idx == 0 else _TOOLBAR_DEFS[_TB_ROW0:]
        ry = y0 + _TB_PAD + row_idx * _TB_ROW_H
        btn_w = dw / len(row)

        for i, (bid, label) in enumerate(row):
            bx = int(i * btn_w) + 1
            bx2 = int((i + 1) * btn_w) - 1
            is_on = states.get(bid, False)
            is_toggle = bid in _TOOLBAR_TOGGLES

            if bid == 'quit':
                bg = (30, 30, 70)
                tc = (120, 120, 255)
            elif is_toggle and is_on:
                bg = (0, 90, 0)
                tc = (100, 255, 100)
            else:
                bg = (55, 55, 55)
                tc = (190, 190, 190)

            if bid == hover_id:
                bg = tuple(min(255, c + 35) for c in bg)
                tc = (255, 255, 255)

            cv2.rectangle(img, (bx, ry), (bx2, ry + _TB_ROW_H - 2), bg, -1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
            tx = bx + (bx2 - bx - tw) // 2
            ty = ry + (_TB_ROW_H - 2 + th) // 2
            cv2.putText(img, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, tc, 1, cv2.LINE_AA)


def get_camera_index():
    # Run ffmpeg to list avfoundation devices
    cmd = [FFMPEG_PATH, '-f', 'avfoundation', '-list_devices', 'true', '-i', '""']
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    
    lines = result.stderr.split('\n')
    for line in lines:
        if 'AVFoundation video devices:' in line:
            continue
        if 'AVFoundation audio devices:' in line:
            break
        
        match = re.search(r'\[(\d+)\]\s+(USB Camera|PureThermal|P2 Pro)', line, re.IGNORECASE)
        if match:
            return match.group(1)
            
    return None

def format_temp(celsius, use_fahrenheit):
    if use_fahrenheit:
        return f"{celsius * 9/5 + 32:.1f} F"
    return f"{celsius:.1f} C"


def unsharp_mask(img_uint8, sigma=1.0, strength=1.5):
    """Apply unsharp mask to grayscale uint8 image for sharpening."""
    blurred = cv2.GaussianBlur(img_uint8, (0, 0), sigma)
    sharp = cv2.addWeighted(img_uint8, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def _read_exact(stream, size):
    """Read exactly `size` bytes; return bytes or empty if EOF/closed."""
    buf = b''
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            return buf
        buf += chunk
    return buf


PANELS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.p2pro_layout.json')
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.p2pro_settings.json')
_TITLE_H = 18
_GRIP = 10


class Panel:
    """Draggable, resizable overlay panel for graph widgets."""

    def __init__(self, name, x, y, w, h, min_w=100, min_h=60):
        self.name = name
        self.x, self.y, self.w, self.h = x, y, w, h
        self.min_w, self.min_h = min_w, min_h
        self._dragging = False
        self._resizing = False
        self._ox = 0
        self._oy = 0

    @property
    def content_rect(self):
        """(x, y, w, h) of drawable area below the title bar."""
        ch = max(0, self.h - _TITLE_H - 3)
        return (self.x + 2, self.y + _TITLE_H + 1, self.w - 4, ch)

    def hit_title(self, mx, my):
        return self.x <= mx < self.x + self.w and self.y <= my < self.y + _TITLE_H

    def hit_grip(self, mx, my):
        return (self.x + self.w - _GRIP <= mx < self.x + self.w and
                self.y + self.h - _GRIP <= my < self.y + self.h)

    def hit(self, mx, my):
        return self.x <= mx < self.x + self.w and self.y <= my < self.y + self.h

    def start_drag(self, mx, my):
        self._dragging = True
        self._ox, self._oy = mx - self.x, my - self.y

    def start_resize(self, mx, my):
        self._resizing = True
        self._ox, self._oy = mx, my

    def on_move(self, mx, my, max_w, max_h):
        if self._dragging:
            self.x = max(0, min(mx - self._ox, max_w - self.w))
            self.y = max(0, min(my - self._oy, max_h - self.h))
        elif self._resizing:
            self.w = max(self.min_w, self.w + mx - self._ox)
            self.h = max(self.min_h, self.h + my - self._oy)
            if self.x + self.w > max_w:
                self.w = max_w - self.x
            if self.y + self.h > max_h:
                self.h = max_h - self.y
            self._ox, self._oy = mx, my

    def stop(self):
        self._dragging = self._resizing = False

    @property
    def active(self):
        return self._dragging or self._resizing

    def draw_frame(self, img, title_color=(0, 255, 255)):
        ih, iw = img.shape[:2]
        x1, y1 = max(0, self.x), max(0, self.y)
        x2, y2 = min(iw, self.x + self.w), min(ih, self.y + self.h)
        if x2 - x1 < 4 or y2 - y1 < _TITLE_H + 4:
            return
        img[y1:y2, x1:x2] = (img[y1:y2, x1:x2].astype(np.float32) * 0.3).astype(np.uint8)
        tb = min(y1 + _TITLE_H, y2)
        img[y1:tb, x1:x2] = (40, 40, 40)
        cv2.putText(img, self.name, (x1 + 5, y1 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, title_color, 1, cv2.LINE_AA)
        border_color = (255, 255, 255) if self._dragging else (80, 80, 80)
        border_thickness = 2 if self._dragging else 1
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), border_color, border_thickness)
        for i in range(3):
            off = 3 + i * 3
            if off < min(_GRIP, x2 - x1, y2 - y1):
                cv2.line(img, (x2 - off, y2 - 1), (x2 - 1, y2 - off), (150, 150, 150), 1)

    def to_dict(self):
        return {'x': self.x, 'y': self.y, 'w': self.w, 'h': self.h}

    def load_dict(self, d, max_w, max_h):
        if not d:
            return
        self.w = max(self.min_w, d.get('w', self.w))
        self.h = max(self.min_h, d.get('h', self.h))
        self.x = max(0, min(d.get('x', self.x), max_w - self.w))
        self.y = max(0, min(d.get('y', self.y), max_h - self.h))


def load_panel_layout(panels, max_w, max_h):
    try:
        with open(PANELS_FILE, 'r') as f:
            data = json.load(f)
        for name, p in panels.items():
            if name in data:
                p.load_dict(data[name], max_w, max_h)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        pass


def save_panel_layout(panels):
    try:
        with open(PANELS_FILE, 'w') as f:
            json.dump({n: p.to_dict() for n, p in panels.items()}, f)
    except OSError:
        pass


def load_app_settings(num_palettes):
    """Load last palette and DDE (contrast/sharpness) settings. Returns dict."""
    try:
        with open(SETTINGS_FILE, 'r') as f:
            data = json.load(f)
        out = {}
        if 'palette_idx' in data:
            out['palette_idx'] = max(0, min(int(data['palette_idx']), num_palettes - 1))
        if 'dde_clip_limit' in data:
            out['dde_clip_limit'] = max(0.0, min(10.0, float(data['dde_clip_limit'])))
        if 'dde_sharp_strength' in data:
            out['dde_sharp_strength'] = max(0.0, min(10.0, float(data['dde_sharp_strength'])))
        if 'dde_sharp_sigma' in data:
            out['dde_sharp_sigma'] = max(0.2, min(5.0, float(data['dde_sharp_sigma'])))
        if 'palette_invert' in data:
            out['palette_invert'] = bool(data['palette_invert'])
        return out
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return {}


def save_app_settings(palette_idx, dde_clip_limit, dde_sharp_strength, dde_sharp_sigma, palette_invert=False):
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump({
                'palette_idx': palette_idx,
                'dde_clip_limit': dde_clip_limit,
                'dde_sharp_strength': dde_sharp_strength,
                'dde_sharp_sigma': dde_sharp_sigma,
                'palette_invert': palette_invert,
            }, f, indent=2)
    except OSError:
        pass


def draw_line_profile(colormap, temp_celsius, start, end, scale, use_fahrenheit, panel):
    """Draw measurement line on image and temperature profile graph in panel."""
    cv2.line(colormap, start, end, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(colormap, start, 5, (0, 255, 255), -1)
    cv2.circle(colormap, end, 5, (0, 255, 255), -1)

    sx1, sy1 = start[0] // scale, start[1] // scale
    sx2, sy2 = end[0] // scale, end[1] // scale
    num_samples = max(int(np.hypot(sx2 - sx1, sy2 - sy1)), 2)
    xs = np.clip(np.linspace(sx1, sx2, num_samples).astype(int), 0, temp_celsius.shape[1] - 1)
    ys = np.clip(np.linspace(sy1, sy2, num_samples).astype(int), 0, temp_celsius.shape[0] - 1)
    temps = temp_celsius[ys, xs]

    panel.draw_frame(colormap, (0, 255, 255))
    cx, cy, cw, ch = panel.content_rect
    pad = 4
    gx, gy, gw, gh = cx + pad, cy + pad, cw - 2 * pad, ch - 2 * pad - 12
    if gw < 10 or gh < 10:
        return

    t_min, t_max = float(np.min(temps)), float(np.max(temps))
    if t_max - t_min < 0.1:
        t_max = t_min + 0.1

    pts = np.zeros((len(temps), 1, 2), dtype=np.int32)
    for i in range(len(temps)):
        pts[i, 0, 0] = gx + int(i / max(len(temps) - 1, 1) * gw)
        pts[i, 0, 1] = gy + gh - int((temps[i] - t_min) / (t_max - t_min) * gh)
    cv2.polylines(colormap, [pts], False, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(colormap, format_temp(t_max, use_fahrenheit), (gx, gy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(colormap, format_temp(t_min, use_fahrenheit), (gx, gy + gh + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1, cv2.LINE_AA)


def draw_isotherms(colormap, temp_celsius, range_min, range_max, scale, use_fahrenheit, num_lines=6):
    """Draw isotherm contour lines at regular temperature intervals."""
    height, width = colormap.shape[:2]
    if range_max - range_min < 1.0:
        return
    temp_up = cv2.resize(temp_celsius.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    step = (range_max - range_min) / (num_lines + 1)

    for i in range(1, num_lines + 1):
        threshold = range_min + i * step
        binary = (temp_up >= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        t = i / (num_lines + 1)
        color = (int(255 * (1 - t)), int(200 * (1 - abs(t - 0.5) * 2)), int(255 * t))
        cv2.drawContours(colormap, contours, -1, color, 1, cv2.LINE_AA)

        big = [c for c in contours if cv2.contourArea(c) > 100]
        if big:
            largest = max(big, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.putText(colormap, format_temp(threshold, use_fahrenheit), (cx - 15, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)


def draw_trend_graph(colormap, trend_data, use_fahrenheit, panel):
    """Draw cursor temperature trend in panel."""
    panel.draw_frame(colormap, (0, 255, 0))
    if len(trend_data) < 2:
        return
    cx, cy, cw, ch = panel.content_rect
    pad = 4
    gx, gy, gw, gh = cx + pad, cy + pad, cw - 2 * pad, ch - 2 * pad - 12
    if gw < 10 or gh < 10:
        return

    temps = list(trend_data)
    t_min, t_max = min(temps), max(temps)
    if t_max - t_min < 0.5:
        mid = (t_max + t_min) / 2
        t_min, t_max = mid - 0.25, mid + 0.25

    pts = np.zeros((len(temps), 1, 2), dtype=np.int32)
    for i, t in enumerate(temps):
        pts[i, 0, 0] = gx + int(i / max(len(temps) - 1, 1) * gw)
        pts[i, 0, 1] = gy + gh - int((t - t_min) / (t_max - t_min) * gh)
    cv2.polylines(colormap, [pts], False, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(colormap, format_temp(t_max, use_fahrenheit), (gx, gy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(colormap, format_temp(t_min, use_fahrenheit), (gx, gy + gh + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(colormap, format_temp(temps[-1], use_fahrenheit), (gx + gw - 35, gy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 0), 1, cv2.LINE_AA)


def draw_histogram(colormap, temp_celsius, use_fahrenheit, panel):
    """Draw temperature distribution histogram in panel."""
    panel.draw_frame(colormap, (200, 200, 255))
    cx, cy, cw, ch = panel.content_rect
    pad = 4
    hx, hy, hw, hh = cx + pad, cy + pad, cw - 2 * pad, ch - 2 * pad - 12
    if hw < 10 or hh < 10:
        return

    t_min, t_max = float(np.min(temp_celsius)), float(np.max(temp_celsius))
    if t_max - t_min < 0.1:
        return
    num_bins = max(hw // 7, 5)
    hist_vals, _ = np.histogram(temp_celsius, bins=num_bins, range=(t_min, t_max))
    max_count = int(np.max(hist_vals)) or 1
    bar_w = max(hw // num_bins, 1)

    for i in range(num_bins):
        bh = int(hist_vals[i] / max_count * hh)
        x1 = hx + i * bar_w
        y1 = hy + hh - bh
        x2 = x1 + bar_w - 1
        y2 = hy + hh
        t = i / num_bins
        color = (int(255 * (1 - t)), int(128 * (1 - abs(t - 0.5) * 2)), int(255 * t))
        cv2.rectangle(colormap, (x1, y1), (x2, y2), color, -1)

    cv2.putText(colormap, format_temp(t_min, use_fahrenheit), (hx, hy + hh + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1, cv2.LINE_AA)
    right_label = format_temp(t_max, use_fahrenheit)
    (lw, _), _ = cv2.getTextSize(right_label, cv2.FONT_HERSHEY_SIMPLEX, 0.25, 1)
    cv2.putText(colormap, right_label, (hx + hw - lw, hy + hh + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1, cv2.LINE_AA)


def draw_anomaly_overlay(colormap, temp_celsius, sensitivity, scale, use_fahrenheit):
    """Highlight anomalous hot regions using z-score detection."""
    height, width = colormap.shape[:2]
    mean_t = np.mean(temp_celsius)
    std_t = np.std(temp_celsius)
    if std_t < 0.1:
        return
    threshold = mean_t + sensitivity * std_t
    hot_mask = (temp_celsius > threshold).astype(np.uint8) * 255
    hot_mask_up = cv2.resize(hot_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(hot_mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant = [c for c in contours if cv2.contourArea(c) > 30]

    if significant:
        overlay = colormap.copy()
        cv2.drawContours(overlay, significant, -1, (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.25, colormap, 0.75, 0, colormap)
        cv2.drawContours(colormap, significant, -1, (0, 0, 255), 2, cv2.LINE_AA)
        label = f"ANOMALIES: {len(significant)} (>{format_temp(threshold, use_fahrenheit)})"
        cv2.putText(colormap, label, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


class FrameReader(threading.Thread):
    def __init__(self, process, frame_size):
        super().__init__()
        self.process = process
        self.frame_size = frame_size
        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()
        self.daemon = True

    def run(self):
        stdout = self.process.stdout
        while self.running:
            raw_bytes = _read_exact(stdout, self.frame_size)
            if len(raw_bytes) != self.frame_size:
                self.running = False
                break
            # Drain pipe: if more frames are buffered, read and drop until we have only the latest (real-time, no lag)
            try:
                while select.select([stdout], [], [], 0)[0]:
                    next_bytes = _read_exact(stdout, self.frame_size)
                    if len(next_bytes) != self.frame_size:
                        break
                    raw_bytes = next_bytes
            except (ValueError, OSError):
                pass  # select on closed fd or unsupported
            with self.lock:
                self.latest_frame = raw_bytes

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame

def main():
    print("Searching for Infiray P2 Pro via ffmpeg...")
    cam_index = get_camera_index()
    
    if cam_index is None:
        print("Infiray P2 Pro not found. Make sure it's plugged in.")
        sys.exit(1)
        
    print(f"Found P2 Pro at AVFoundation index {cam_index}")
    
    # Command to capture raw YUYV stream
    # Added -fflags nobuffer and -flags low_delay to remove latency
    command = [
        FFMPEG_PATH,
        '-hide_banner',
        '-loglevel', 'error',
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-f', 'avfoundation',
        '-framerate', '25',
        '-video_size', '256x384',
        '-pix_fmt', 'yuyv422',
        '-i', str(cam_index),
        '-f', 'rawvideo',
        '-pix_fmt', 'yuyv422',
        '-'
    ]
    
    frame_size = 256 * 384 * 2  # 256x384 resolution, 2 bytes per pixel (YUYV)
    # One-frame buffer; reader drains pipe to latest frame for real-time
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_size)

    # Start background thread to continuously read frames and drop old ones
    reader = FrameReader(process, frame_size)
    reader.start()

    print("Starting video stream.")
    print("Controls:")
    print("  'q' - Quit   'p' - Next Palette   'm' - Min/Max pointers   'd' - Debug")
    print("  SPACE - Freeze frame   's' - Snapshot   'f' - Celsius/Fahrenheit")
    print("  '1'-'9' - Palette (LUT)   'z' - Invert palette   '0' - Temp range cycle: Auto / Room / Wide")
    print("  'r' - ROI: draw rectangle for min/max/avg   'a' - High temp alarm")
    print("  'h' - High quality: DDE (Detail Enhancement) & Sharpening")
    print("  '['/']' - Decrease/Increase DDE Contrast   '-'/'=' - Decrease/Increase Sharpness")
    print("  'c' - FPN correction (running)   'x' - FPN from one frame   'i' - Help")
    print("  'l' - Line profile   't' - Isotherms   'v' - Delta-T mode")
    print("  'w' - Temp trend   'b' - Histogram   'n' - Anomaly detection")
    print("  Left/Right arrows - Rotate 90 deg   Up/Down arrows - Flip vertical/horizontal")
    
    # Upscale factor (e.g. 3 means 256x192 -> 768x576)
    SCALE = 3
    WIDTH = 256 * SCALE
    HEIGHT = 192 * SCALE

    # Toggles and state
    global running
    running = True
    show_min_max = False
    debug_mode = False
    frame_count = 0
    use_fahrenheit = False
    frozen = False
    frozen_frame = None  # when frozen, we show this (BGR image with bar)
    range_preset = 0  # 0=auto, 1=room 15-35, 2=wide -20-150
    roi_state = {'active': False, 'start': None, 'end': None, 'dragging': False}
    alarm_enabled = False
    alarm_threshold = 60.0  # Celsius
    high_quality_mode = False  # 'h': DDE
    show_controls_overlay = False  # 'i': show all keys
    palette_invert = False  # 'z': invert colormap (hot/cold swap)
    fps_times = deque(maxlen=30)  # for FPS calculation
    
    # DDE Parameters
    dde_clip_limit = 3.0     # Contrast limit for CLAHE
    dde_sharp_strength = 2.5 # Sharpness strength for Unsharp Mask
    dde_sharp_sigma = 1.2    # Sharpness radius

    # FPN (fixed-pattern noise) correction: running per-pixel calibration
    fpn_enabled = False
    fpn_corrector = FPNCorrector((192, 256), alpha=0.995)
    # One-frame FPN: capture current frame on X, apply that offset to all following frames; X again = reset
    single_frame_fpn_enabled = False
    single_frame_offset_map = None  # float64 (192,256), offset to subtract each frame
    # Thermal de-noise: temporal averaging + spatial bilateral filter
    denoise_enabled = False
    denoiser = ThermalDenoiser((192, 256))
    last_raw_bottom_half = None    # last raw thermal frame, for X capture

    line_profile_mode = False
    line_profile = {'start': None, 'end': None, 'dragging': False}
    isotherm_enabled = False
    delta_t_enabled = False
    delta_t_reference = None
    trend_enabled = False
    trend_data = deque(maxlen=300)
    histogram_enabled = False
    anomaly_enabled = False

    # Rotate (0, 90, 180, 270) and flip for display
    rotation_deg = 0
    flip_h = False
    flip_v = False

    def apply_rotate_flip(img):
        out = img.copy()
        if flip_v:
            out = cv2.flip(out, 0)
        if flip_h:
            out = cv2.flip(out, 1)
        if rotation_deg == 90:
            out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_deg == 180:
            out = cv2.rotate(out, cv2.ROTATE_180)
        elif rotation_deg == 270:
            out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return out

    def display_to_logical(dx, dy):
        """Convert display (window) coords to logical image coords (for cursor, ROI, etc.)."""
        disp_w = HEIGHT if rotation_deg in (90, 270) else WIDTH
        disp_h = WIDTH if rotation_deg in (90, 270) else HEIGHT
        dx = max(0, min(dx, disp_w - 1))
        dy = max(0, min(dy, disp_h - 1))
        if rotation_deg == 90:
            lx, ly = dy, HEIGHT - 1 - dx
        elif rotation_deg == 180:
            lx, ly = WIDTH - 1 - dx, HEIGHT - 1 - dy
        elif rotation_deg == 270:
            lx, ly = WIDTH - 1 - dy, dx
        else:
            lx, ly = dx, dy
        if flip_h:
            lx = WIDTH - 1 - lx
        if flip_v:
            ly = HEIGHT - 1 - ly
        return (max(0, min(lx, WIDTH - 1)), max(0, min(ly, HEIGHT - 1)))

    def logical_to_display(lx, ly):
        """Convert logical image coords to display coords (for drawing text on display_img)."""
        px = (WIDTH - 1 - lx) if flip_h else lx
        py = (HEIGHT - 1 - ly) if flip_v else ly
        if rotation_deg == 90:
            return (HEIGHT - 1 - py, px)
        if rotation_deg == 180:
            return (WIDTH - 1 - px, HEIGHT - 1 - py)
        if rotation_deg == 270:
            return (py, WIDTH - 1 - px)
        return (px, py)

    panels = {
        'trend': Panel("Trend (W)", WIDTH - 185, 45, 180, 130),
        'histogram': Panel("Histogram (B)", WIDTH - 185, 185, 180, 110),
        'line_profile': Panel("Line Profile (L)", 15, HEIGHT - 140, WIDTH - 30, 120, min_w=150),
    }
    load_panel_layout(panels, WIDTH, HEIGHT)

    # Mouse callback: cursor position + ROI drag + panel interaction
    mouse_x, mouse_y = WIDTH // 2, HEIGHT // 2
    _panel_drag = {'ref': None}

    _tb_display_dims = [WIDTH, HEIGHT]
    _toolbar_injected_key = [-1]
    _tb_hover_id = [None]

    # Initialize AI Super Resolution (FSRCNN)
    HAS_SR = False
    sr = None
    HAS_VULKAN_SR = False
    sr_vulkan = None

    print("Note: AI Upscaling disabled in favor of DDE (Digital Detail Enhancement) for pure sharpness.")
    
    def on_mouse(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y

        tb_dh = _tb_display_dims[1]
        if y >= tb_dh:
            if event == cv2.EVENT_LBUTTONDOWN:
                bid = _toolbar_find_button(x, y - tb_dh, _tb_display_dims[0])
                if bid and bid in _TOOLBAR_KEY_MAP:
                    _toolbar_injected_key[0] = _TOOLBAR_KEY_MAP[bid]
            elif event == cv2.EVENT_MOUSEMOVE:
                _tb_hover_id[0] = _toolbar_find_button(x, y - tb_dh, _tb_display_dims[0])
            return
        _tb_hover_id[0] = None

        mouse_x, mouse_y = display_to_logical(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            lp_shown = (line_profile_mode and line_profile.get('start')
                        and line_profile.get('end') and not line_profile.get('dragging'))
            visible = []
            if lp_shown:
                visible.append(panels['line_profile'])
            if histogram_enabled:
                visible.append(panels['histogram'])
            if trend_enabled:
                visible.append(panels['trend'])
            for p in visible:
                if p.hit_grip(mouse_x, mouse_y):
                    p.start_resize(mouse_x, mouse_y)
                    _panel_drag['ref'] = p
                    return
                if p.hit_title(mouse_x, mouse_y):
                    p.start_drag(mouse_x, mouse_y)
                    _panel_drag['ref'] = p
                    return
            if line_profile_mode:
                line_profile['start'] = (mouse_x, mouse_y)
                line_profile['end'] = (mouse_x, mouse_y)
                line_profile['dragging'] = True
            elif roi_state['active']:
                roi_state['start'] = (mouse_x, mouse_y)
                roi_state['end'] = (mouse_x, mouse_y)
                roi_state['dragging'] = True

        elif event == cv2.EVENT_MOUSEMOVE:
            ap = _panel_drag['ref']
            if ap and ap.active:
                ap.on_move(mouse_x, mouse_y, WIDTH, HEIGHT)
                return
            if line_profile.get('dragging'):
                line_profile['end'] = (mouse_x, mouse_y)
            elif roi_state['dragging']:
                roi_state['end'] = (mouse_x, mouse_y)

        elif event == cv2.EVENT_LBUTTONUP:
            ap = _panel_drag['ref']
            if ap and ap.active:
                ap.stop()
                _panel_drag['ref'] = None
                save_panel_layout(panels)
                return
            if line_profile.get('dragging'):
                line_profile['end'] = (mouse_x, mouse_y)
                line_profile['dragging'] = False
            elif roi_state['dragging']:
                roi_state['end'] = (mouse_x, mouse_y)
                roi_state['dragging'] = False

    window_name = 'Infiray P2 Pro Thermal Viewer'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    def window_closed():
        try:
            return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    palette_idx = 0

    # Restore last saved palette and DDE (contrast/sharpness) settings
    _saved = load_app_settings(len(PALETTES))
    if _saved:
        palette_idx = _saved.get('palette_idx', palette_idx)
        if 'dde_clip_limit' in _saved:
            dde_clip_limit = _saved['dde_clip_limit']
        if 'dde_sharp_strength' in _saved:
            dde_sharp_strength = _saved['dde_sharp_strength']
        if 'dde_sharp_sigma' in _saved:
            dde_sharp_sigma = _saved['dde_sharp_sigma']
        if 'palette_invert' in _saved:
            palette_invert = _saved['palette_invert']

    def on_quit(icon, item):
        global running
        running = False
        icon.stop()

    def create_image():
        # Create a simple icon image (e.g. a small thermal-like colored square)
        image = Image.new('RGB', (64, 64), color=(0, 0, 0))
        d = ImageDraw.Draw(image)
        d.rectangle((16, 16, 48, 48), fill=(255, 100, 0))
        return image

    tray_icon = pystray.Icon("Infiray P2 Pro", create_image(), "Infiray P2 Pro", menu=pystray.Menu(
        pystray.MenuItem("Quit", on_quit)
    ))
    tray_icon.run_detached()

    while reader.running and running:
        if frozen and frozen_frame is not None:
            display_frozen = apply_rotate_flip(frozen_frame.copy())
            df_h, df_w = display_frozen.shape[:2]
            cv2.putText(display_frozen, " FROZEN (SPACE to resume) ", (df_w // 2 - 120, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            if show_controls_overlay:
                overlay = display_frozen.copy()
                cv2.rectangle(overlay, (10, 10), (df_w - 10, df_h - 10), (30, 30, 30), -1)
                cv2.addWeighted(overlay, 0.88, display_frozen, 0.12, 0, display_frozen)
                line_h, y0 = 24, 35
                font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.5
                cv2.putText(display_frozen, "--- CONTROLS (I to close) ---", (20, y0), font, 0.55, (200, 255, 200), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_frozen, "Q - Quit   P - Next palette   M - Min/Max   D - Debug", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_frozen, "SPACE - Resume   S - Snapshot   F - C/F", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_frozen, "1-9 - Palette   0 - Range   R - ROI   A - Alarm   H - DDE   [ ] = - DDE   I - Help", (20, y0), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            _tb_display_dims[0], _tb_display_dims[1] = df_w, df_h
            frozen_tb_states = {
                'minmax': show_min_max, 'freeze': True, 'fahr': use_fahrenheit,
                'invert': palette_invert, 'roi': roi_state['active'], 'line': line_profile_mode,
                'iso': isotherm_enabled, 'hist': histogram_enabled,
                'trend': trend_enabled, 'anomaly': anomaly_enabled,
                'delta': delta_t_enabled, 'alarm': alarm_enabled,
                'dde': high_quality_mode, 'fpn': fpn_enabled,
                'fpn1': single_frame_fpn_enabled, 'dnz': denoise_enabled,
                'flip_v': flip_v,
                'flip_h': flip_h, 'debug': debug_mode, 'help': show_controls_overlay,
            }
            ftb = np.zeros((TOOLBAR_H, df_w, 3), dtype=np.uint8)
            display_frozen = np.vstack([display_frozen, ftb])
            draw_toolbar(display_frozen, df_h, df_w, frozen_tb_states, _tb_hover_id[0])
            cv2.imshow(window_name, display_frozen)
            key = cv2.waitKey(1)
            if window_closed():
                break
            if key < 0 and _toolbar_injected_key[0] >= 0:
                key = _toolbar_injected_key[0]
                _toolbar_injected_key[0] = -1
            elif _toolbar_injected_key[0] >= 0:
                _toolbar_injected_key[0] = -1
            if key < 0:
                continue
            key = key & 0xFFFF
            if key in KEY_SPACE:
                frozen = False
            elif key in KEY_PALETTE:
                palette_idx = (palette_idx + 1) % len(PALETTES)
            elif key in KEY_PALETTE_1:
                palette_idx = min(0, len(PALETTES) - 1)
            elif key in KEY_PALETTE_2:
                palette_idx = min(1, len(PALETTES) - 1)
            elif key in KEY_PALETTE_3:
                palette_idx = min(2, len(PALETTES) - 1)
            elif key in KEY_PALETTE_4:
                palette_idx = min(3, len(PALETTES) - 1)
            elif key in KEY_PALETTE_5:
                palette_idx = min(4, len(PALETTES) - 1)
            elif key in KEY_PALETTE_6:
                palette_idx = min(5, len(PALETTES) - 1)
            elif key in KEY_PALETTE_7:
                palette_idx = min(6, len(PALETTES) - 1)
            elif key in KEY_PALETTE_8:
                palette_idx = min(7, len(PALETTES) - 1)
            elif key in KEY_PALETTE_9:
                palette_idx = min(8, len(PALETTES) - 1)
            elif key in KEY_RANGE_CYCLE:
                range_preset = (range_preset + 1) % 3
            elif key in KEY_FPN:
                fpn_enabled = not fpn_enabled
                if not fpn_enabled:
                    fpn_corrector.reset()
            elif key in KEY_FPN_ONE and single_frame_fpn_enabled:
                single_frame_fpn_enabled = False
                single_frame_offset_map = None
            elif key in KEY_DENOISE:
                denoise_enabled = not denoise_enabled
                if not denoise_enabled:
                    denoiser.reset()
            elif key in KEY_HELP:
                show_controls_overlay = not show_controls_overlay
            elif key in KEY_PALETTE_INVERT:
                palette_invert = not palette_invert
            elif key in KEY_SNAPSHOT:
                fn = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(fn, apply_rotate_flip(frozen_frame))
                print(f"Saved {fn}")
            elif key in KEY_QUIT:
                break
            elif key in KEY_ARROW_LEFT:
                rotation_deg = (rotation_deg - 90) % 360
            elif key in KEY_ARROW_RIGHT:
                rotation_deg = (rotation_deg + 90) % 360
            elif key in KEY_ARROW_UP:
                flip_v = not flip_v
            elif key in KEY_ARROW_DOWN:
                flip_h = not flip_h
            continue

        raw_bytes = reader.get_latest_frame()
        if raw_bytes is None:
            cv2.waitKey(10)
            continue

        frame_count += 1
        fps_times.append(time.perf_counter())

        # Convert bottom half to 16-bit array
        bottom_half_bytes = raw_bytes[256*192*2:]
        bottom_half = np.frombuffer(bottom_half_bytes, dtype=np.uint16).reshape((192, 256)).copy()
        last_raw_bottom_half = bottom_half.copy()

        # One-frame FPN: subtract offset from captured frame (X) to all frames
        if single_frame_fpn_enabled and single_frame_offset_map is not None:
            bottom_half = np.clip(
                bottom_half.astype(np.float64) - single_frame_offset_map, 0, 65535
            ).astype(np.uint16)

        # Running FPN correction (removes fixed per-pixel offset pattern over time)
        if fpn_enabled:
            bottom_half = fpn_corrector.apply(bottom_half)

        # Calculate temperature
        temp_celsius = (bottom_half / 64.0) - 273.15
        raw_min_c, raw_max_c = np.min(temp_celsius), np.max(temp_celsius)
        # Filter out obvious outliers
        temp_celsius = np.clip(temp_celsius, -20, 150)

        # Temporal + spatial de-noise (G)
        if denoise_enabled:
            temp_celsius = denoiser.apply(temp_celsius)
        
        # Calculate min/max on the original 256x192 array for accuracy and speed
        min_temp = np.min(temp_celsius)
        max_temp = np.max(temp_celsius)
        if range_preset == 1:
            range_min, range_max = 15.0, 35.0
        elif range_preset == 2:
            range_min, range_max = -20.0, 150.0
        else:
            range_min, range_max = min_temp, max_temp
        
        # Find pixel coordinates of min and max
        min_flat = np.argmin(temp_celsius)
        max_flat = np.argmax(temp_celsius)
        min_pt = np.unravel_index(min_flat, temp_celsius.shape)  # (y, x)
        max_pt = np.unravel_index(max_flat, temp_celsius.shape)  # (y, x)
        min_x, min_y = min_pt[1] * SCALE, min_pt[0] * SCALE
        max_x, max_y = max_pt[1] * SCALE, max_pt[0] * SCALE
        
        # Normalize to 8-bit BEFORE upscaling
        delta_display = None
        if delta_t_enabled and delta_t_reference is not None:
            delta_display = temp_celsius - delta_t_reference
            abs_range = max(abs(float(np.min(delta_display))), abs(float(np.max(delta_display))), 0.1)
            normalized_small = ((delta_display + abs_range) / (2 * abs_range) * 255).astype(np.uint8)
            normalized_small = np.clip(normalized_small, 0, 255)
        elif range_max > range_min:
            normalized_small = ((temp_celsius - range_min) / (range_max - range_min) * 255).astype(np.uint8)
            normalized_small = np.clip(normalized_small, 0, 255)
        else:
            normalized_small = np.zeros_like(temp_celsius, dtype=np.uint8)

        # Upscale
        if high_quality_mode:
            # DDE (Digital Detail Enhancement) for Thermal Images
            # 1. Local contrast enhancement via CLAHE (skip when contrast=0)
            if dde_clip_limit > 0:
                clahe = cv2.createCLAHE(clipLimit=dde_clip_limit, tileGridSize=(8, 8))
                enhanced = clahe.apply(normalized_small)
            else:
                enhanced = normalized_small

            # 2. High-quality mathematical upscale (preserves 100% of raw sensor details)
            normalized = cv2.resize(enhanced, (WIDTH, HEIGHT), interpolation=cv2.INTER_LANCZOS4)

            # 3. Aggressive sharpening to make edges pop without hallucinating
            if dde_sharp_strength > 0:
                normalized = unsharp_mask(normalized, sigma=dde_sharp_sigma, strength=dde_sharp_strength)
        else:
            normalized = cv2.resize(normalized_small, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
            
        # Apply selected palette
        palette_name, colormap_flag = PALETTES[palette_idx]
        to_map = (255 - normalized).astype(np.uint8) if palette_invert else normalized
        colormap = cv2.applyColorMap(to_map, colormap_flag)
        
        if isotherm_enabled:
            draw_isotherms(colormap, temp_celsius, range_min, range_max, SCALE, use_fahrenheit)

        if anomaly_enabled:
            draw_anomaly_overlay(colormap, temp_celsius, 2.0, SCALE, use_fahrenheit)

        # Get temperature at mouse cursor
        if 0 <= mouse_x < WIDTH and 0 <= mouse_y < HEIGHT:
            cursor_temp = temp_celsius[mouse_y // SCALE, mouse_x // SCALE]
        else:
            cursor_temp = 0.0
        
        # Draw min/max markers only when enabled (markers on colormap; labels drawn on display later)
        if show_min_max:
            cv2.drawMarker(colormap, (min_x, min_y), (255, 100, 0), cv2.MARKER_TRIANGLE_DOWN, 24, 2)  # blue = min
            cv2.drawMarker(colormap, (max_x, max_y), (0, 0, 255), cv2.MARKER_TRIANGLE_UP, 24, 2)   # red = max
        
        if trend_enabled:
            trend_data.append(cursor_temp)

        # Draw crosshair at cursor (text drawn on display later)
        cv2.drawMarker(colormap, (mouse_x, mouse_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
        cursor_label = format_temp(cursor_temp, use_fahrenheit)
        if delta_display is not None:
            d = delta_display[mouse_y // SCALE, mouse_x // SCALE]
            cursor_label += f" ({'+' if d >= 0 else ''}{d:.1f})"
        
        if roi_state['start'] and roi_state['end']:
            x1, y1 = min(roi_state['start'][0], roi_state['end'][0]), min(roi_state['start'][1], roi_state['end'][1])
            x2, y2 = max(roi_state['start'][0], roi_state['end'][0]), max(roi_state['start'][1], roi_state['end'][1])
            x1, x2 = max(0, x1), min(WIDTH, x2)
            y1, y2 = max(0, y1), min(HEIGHT, y2)
            if x2 > x1 and y2 > y1:
                sx1, sy1 = x1 // SCALE, y1 // SCALE
                sx2, sy2 = max(sx1 + 1, x2 // SCALE), max(sy1 + 1, y2 // SCALE)
                roi_temps = temp_celsius[sy1:sy2, sx1:sx2]
                roi_min, roi_max = np.min(roi_temps), np.max(roi_temps)
                roi_avg = np.mean(roi_temps)
                cv2.rectangle(colormap, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if alarm_enabled and max_temp >= alarm_threshold:
            cv2.rectangle(colormap, (0, 0), (WIDTH - 1, HEIGHT - 1), (0, 0, 255), 4)

        # FPS calculation: frames per second = num_frames / elapsed_time
        if len(fps_times) >= 2:
            elapsed = fps_times[-1] - fps_times[0]
            fps = len(fps_times) / elapsed if elapsed > 0 else 0.0
        else:
            fps = 0.0
        fps_text = f"{fps:.1f} FPS"
        fps_scale = 0.55
        (fw, fh), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, fps_scale, 1)
        fps_x = WIDTH - fw - 10
        status_scale = 0.45
        gap = 20
        status_right = fps_x - gap

        if line_profile_mode and line_profile['start'] and line_profile['end']:
            if line_profile.get('dragging'):
                cv2.line(colormap, line_profile['start'], line_profile['end'], (0, 255, 255), 2, cv2.LINE_AA)
            else:
                draw_line_profile(colormap, temp_celsius, line_profile['start'], line_profile['end'], SCALE, use_fahrenheit, panels['line_profile'])

        if trend_enabled:
            draw_trend_graph(colormap, trend_data, use_fahrenheit, panels['trend'])

        if histogram_enabled:
            draw_histogram(colormap, temp_celsius, use_fahrenheit, panels['histogram'])

        display_img = apply_rotate_flip(colormap)
        dh, dw = display_img.shape[:2]

        # --- All text/overlay drawn on display_img in SCREEN coords (always readable, fixed positions) ---

        # Cursor label follows mouse in screen space
        cmx, cmy = logical_to_display(mouse_x, mouse_y)
        cmx, cmy = max(0, min(cmx, dw - 1)), max(0, min(cmy, dh - 1))
        cv2.putText(display_img, cursor_label, (cmx + 10, cmy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Min/Max labels follow their markers in screen space
        if show_min_max:
            dmn = logical_to_display(min_x, min_y)
            cv2.putText(display_img, f"MIN {format_temp(min_temp, use_fahrenheit)}",
                        (dmn[0] + 12, dmn[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2, cv2.LINE_AA)
            dmx = logical_to_display(max_x, max_y)
            cv2.putText(display_img, f"MAX {format_temp(max_temp, use_fahrenheit)}",
                        (dmx[0] + 12, dmx[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2, cv2.LINE_AA)

        # ROI label follows ROI in screen space
        if roi_state['start'] and roi_state['end']:
            x1, y1 = min(roi_state['start'][0], roi_state['end'][0]), min(roi_state['start'][1], roi_state['end'][1])
            x2, y2 = max(roi_state['start'][0], roi_state['end'][0]), max(roi_state['start'][1], roi_state['end'][1])
            x1, x2 = max(0, x1), min(WIDTH, x2)
            y1, y2 = max(0, y1), min(HEIGHT, y2)
            if x2 > x1 and y2 > y1:
                sx1, sy1 = x1 // SCALE, y1 // SCALE
                sx2, sy2 = max(sx1 + 1, x2 // SCALE), max(sy1 + 1, y2 // SCALE)
                roi_temps = temp_celsius[sy1:sy2, sx1:sx2]
                roi_min, roi_max = np.min(roi_temps), np.max(roi_temps)
                roi_avg = np.mean(roi_temps)
                dr = logical_to_display(x1, y1)
                cv2.putText(display_img, f"ROI: {format_temp(roi_min, use_fahrenheit)} / {format_temp(roi_avg, use_fahrenheit)} / {format_temp(roi_max, use_fahrenheit)}",
                            (dr[0], dr[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # --- Fixed HUD: always at screen edges, using dw/dh ---
        if delta_display is not None:
            d_min, d_max = float(np.min(delta_display)), float(np.max(delta_display))
            cv2.putText(display_img, f"DELTA-T: {d_min:+.1f} to {d_max:+.1f} C  (V to reset)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(display_img, f"Min: {format_temp(min_temp, use_fahrenheit)}  Max: {format_temp(max_temp, use_fahrenheit)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if alarm_enabled and max_temp >= alarm_threshold:
            cv2.putText(display_img, f" HOT! {format_temp(max_temp, use_fahrenheit)} ", (dw // 2 - 80, dh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(display_img, f"Palette: {palette_name}{' [inv]' if palette_invert else ''} (1-9,P,Z) | Range: {'Auto' if range_preset == 0 else 'Room' if range_preset == 1 else 'Wide'} (0)",
                    (10, dh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        if high_quality_mode:
            cv2.putText(display_img, f" HQ (H) | Cont: {dde_clip_limit:.1f} | Shrp: {dde_sharp_strength:.1f} ",
                        (dw - 280, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1, cv2.LINE_AA)
        if roi_state['active']:
            cv2.putText(display_img, "ROI: drag (R cancel)", (10, dh - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        elif line_profile_mode:
            cv2.putText(display_img, "LINE: drag to draw (L cancel)", (10, dh - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

        (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, fps_scale, 1)
        d_fps_x = dw - fw - 10
        cv2.putText(display_img, fps_text, (d_fps_x, dh - 10), cv2.FONT_HERSHEY_SIMPLEX, fps_scale, (255, 255, 255), 1, cv2.LINE_AA)
        status_right = d_fps_x - gap
        if single_frame_fpn_enabled:
            (sw, _), _ = cv2.getTextSize(" 1-frame FPN (X) ", cv2.FONT_HERSHEY_SIMPLEX, status_scale, 1)
            cv2.putText(display_img, " 1-frame FPN (X) ", (status_right - sw, dh - 12), cv2.FONT_HERSHEY_SIMPLEX, status_scale, (200, 255, 200), 1, cv2.LINE_AA)
            status_right -= sw + 8
        if fpn_enabled:
            (sw, _), _ = cv2.getTextSize(" FPN (C) ", cv2.FONT_HERSHEY_SIMPLEX, status_scale, 1)
            cv2.putText(display_img, " FPN (C) ", (status_right - sw, dh - 12), cv2.FONT_HERSHEY_SIMPLEX, status_scale, (0, 255, 200), 1, cv2.LINE_AA)
            status_right -= sw + 8
        if denoise_enabled:
            (sw, _), _ = cv2.getTextSize(" DNZ (G) ", cv2.FONT_HERSHEY_SIMPLEX, status_scale, 1)
            cv2.putText(display_img, " DNZ (G) ", (status_right - sw, dh - 12), cv2.FONT_HERSHEY_SIMPLEX, status_scale, (100, 200, 255), 1, cv2.LINE_AA)
            status_right -= sw + 8
        if delta_t_enabled:
            (sw, _), _ = cv2.getTextSize(" Delta-T (V) ", cv2.FONT_HERSHEY_SIMPLEX, status_scale, 1)
            cv2.putText(display_img, " Delta-T (V) ", (status_right - sw, dh - 12), cv2.FONT_HERSHEY_SIMPLEX, status_scale, (0, 200, 255), 1, cv2.LINE_AA)
            status_right -= sw + 8
        if isotherm_enabled:
            (sw, _), _ = cv2.getTextSize(" ISO (T) ", cv2.FONT_HERSHEY_SIMPLEX, status_scale, 1)
            cv2.putText(display_img, " ISO (T) ", (status_right - sw, dh - 12), cv2.FONT_HERSHEY_SIMPLEX, status_scale, (200, 200, 0), 1, cv2.LINE_AA)
            status_right -= sw + 8
        if anomaly_enabled:
            (sw, _), _ = cv2.getTextSize(" AI (N) ", cv2.FONT_HERSHEY_SIMPLEX, status_scale, 1)
            cv2.putText(display_img, " AI (N) ", (status_right - sw, dh - 12), cv2.FONT_HERSHEY_SIMPLEX, status_scale, (0, 0, 255), 1, cv2.LINE_AA)

        if show_controls_overlay:
            overlay = display_img.copy()
            cv2.rectangle(overlay, (10, 10), (dw - 10, dh - 10), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.88, display_img, 0.12, 0, display_img)
            line_h, y0 = 24, 35
            font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.5
            cv2.putText(display_img, "--- CONTROLS (I to close) ---", (20, y0), font, 0.55, (200, 255, 200), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, "Q - Quit   P - Next palette   M - Min/Max markers   D - Debug", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, "SPACE - Freeze frame   S - Snapshot   F - Celsius/Fahrenheit", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, "1-9 - Palette   P - Next   Z - Invert palette   0 - Temp range: Auto / Room / Wide", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, "R - ROI: draw rectangle (min/avg/max)   A - High temp alarm", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, "H - High quality: DDE + Sharpening", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, "[ ] - DDE Contrast less/more   - = - Sharpness less/more", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, "C - FPN correction (running calibration)", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, "X - One-frame FPN: capture current frame, apply to all; X again = reset", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, "G - De-noise: temporal averaging + bilateral filter (matrix noise)", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, "L - Line profile   T - Isotherms   V - Delta-T mode", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, "W - Temp trend   B - Histogram   N - Anomaly detection", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, "Left/Right - Rotate 90 deg   Up/Down - Flip V/H", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, "I - Show/hide this help", (20, y0), font, fs, (200, 255, 200), 1)

        if debug_mode:
            overlay = display_img.copy()
            cv2.rectangle(overlay, (10, 50), (dw - 10, dh - 50), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, display_img, 0.25, 0, display_img)
            line_h, y0 = 22, 65
            font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.55
            cv2.putText(display_img, "--- DEBUG (D to toggle) ---", (15, y0), font, fs, (200, 200, 200), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, f"FPS: {fps:.1f}", (15, y0), font, fs, (200, 255, 200), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, f"frame_count: {frame_count}", (15, y0), font, fs, (200, 255, 200), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, f"cam_index: {cam_index}", (15, y0), font, fs, (200, 255, 200), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, f"frame_size (bytes): {frame_size}", (15, y0), font, fs, (200, 255, 200), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, f"SCALE: {SCALE}  WIDTH: {WIDTH}  HEIGHT: {HEIGHT}", (15, y0), font, fs, (200, 255, 200), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, f"palette_idx: {palette_idx}  palette: {palette_name}", (15, y0), font, fs, (200, 255, 200), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, f"show_min_max: {show_min_max}", (15, y0), font, fs, (200, 255, 200), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, f"reader.running: {reader.running}", (15, y0), font, fs, (200, 255, 200), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, f"raw temp (before clip) min: {raw_min_c:.2f} C  max: {raw_max_c:.2f} C", (15, y0), font, fs, (255, 255, 200), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, f"display temp min: {min_temp:.2f} C  max: {max_temp:.2f} C", (15, y0), font, fs, (255, 255, 200), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, f"min_pt: ({min_x}, {min_y})  max_pt: ({max_x}, {max_y})", (15, y0), font, fs, (255, 255, 200), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, f"cursor: ({mouse_x}, {mouse_y})  cursor_temp: {cursor_temp:.2f} C", (15, y0), font, fs, (255, 255, 200), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, f"bottom_half shape: {bottom_half.shape}  dtype: {bottom_half.dtype}", (15, y0), font, fs, (200, 200, 255), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, f"temp_celsius shape: {temp_celsius.shape}", (15, y0), font, fs, (200, 200, 255), 1, cv2.LINE_AA); y0 += line_h
            cv2.putText(display_img, f"high_quality_mode: {high_quality_mode} (DDE Sharpness)", (15, y0), font, fs, (200, 255, 200), 1, cv2.LINE_AA)

        _tb_display_dims[0], _tb_display_dims[1] = dw, dh
        tb_states = {
            'minmax': show_min_max, 'freeze': frozen, 'fahr': use_fahrenheit,
            'invert': palette_invert, 'roi': roi_state['active'], 'line': line_profile_mode,
            'iso': isotherm_enabled, 'hist': histogram_enabled,
            'trend': trend_enabled, 'anomaly': anomaly_enabled,
            'delta': delta_t_enabled, 'alarm': alarm_enabled,
            'dde': high_quality_mode, 'fpn': fpn_enabled,
            'fpn1': single_frame_fpn_enabled, 'dnz': denoise_enabled,
            'flip_v': flip_v,
            'flip_h': flip_h, 'debug': debug_mode, 'help': show_controls_overlay,
        }
        tb_strip = np.zeros((TOOLBAR_H, dw, 3), dtype=np.uint8)
        display_img = np.vstack([display_img, tb_strip])
        draw_toolbar(display_img, dh, dw, tb_states, _tb_hover_id[0])
        cv2.imshow(window_name, display_img)

        key = cv2.waitKey(1)
        if window_closed():
            break
        if key < 0 and _toolbar_injected_key[0] >= 0:
            key = _toolbar_injected_key[0]
            _toolbar_injected_key[0] = -1
        elif _toolbar_injected_key[0] >= 0:
            _toolbar_injected_key[0] = -1
        if key < 0:
            continue
        key = key & 0xFFFF
        if key in KEY_QUIT:
            break
        elif key in KEY_PALETTE:
            palette_idx = (palette_idx + 1) % len(PALETTES)
        elif key in KEY_PALETTE_1:
            palette_idx = min(0, len(PALETTES) - 1)
        elif key in KEY_PALETTE_2:
            palette_idx = min(1, len(PALETTES) - 1)
        elif key in KEY_PALETTE_3:
            palette_idx = min(2, len(PALETTES) - 1)
        elif key in KEY_PALETTE_4:
            palette_idx = min(3, len(PALETTES) - 1)
        elif key in KEY_PALETTE_5:
            palette_idx = min(4, len(PALETTES) - 1)
        elif key in KEY_PALETTE_6:
            palette_idx = min(5, len(PALETTES) - 1)
        elif key in KEY_PALETTE_7:
            palette_idx = min(6, len(PALETTES) - 1)
        elif key in KEY_PALETTE_8:
            palette_idx = min(7, len(PALETTES) - 1)
        elif key in KEY_PALETTE_9:
            palette_idx = min(8, len(PALETTES) - 1)
        elif key in KEY_RANGE_CYCLE:
            range_preset = (range_preset + 1) % 3
        elif key in KEY_MINMAX:
            show_min_max = not show_min_max
        elif key in KEY_DEBUG:
            debug_mode = not debug_mode
        elif key in KEY_SPACE:
            frozen_frame = colormap.copy()
            frozen = True
        elif key in KEY_SNAPSHOT:
            fn = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(fn, display_img)
            print(f"Saved {fn}")
        elif key in KEY_FAHRENHEIT:
            use_fahrenheit = not use_fahrenheit
        elif key in KEY_PALETTE_INVERT:
            palette_invert = not palette_invert
        elif key in KEY_ROI:
            roi_state['active'] = not roi_state['active']
            if roi_state['active']:
                line_profile_mode = False
                line_profile['start'] = line_profile['end'] = None
                line_profile['dragging'] = False
            else:
                roi_state['start'] = roi_state['end'] = None
        elif key in KEY_ALARM:
            alarm_enabled = not alarm_enabled
        elif key in KEY_HQ:
            high_quality_mode = not high_quality_mode
        elif key in KEY_BRACKET_L:
            dde_clip_limit = max(0.0, dde_clip_limit - 0.5)
        elif key in KEY_BRACKET_R:
            dde_clip_limit += 0.5
        elif key in KEY_SHARP_MINUS:
            dde_sharp_strength = max(0.0, dde_sharp_strength - 0.5)
        elif key in KEY_SHARP_PLUS:
            dde_sharp_strength += 0.5
        elif key in KEY_HELP:
            show_controls_overlay = not show_controls_overlay
        elif key in KEY_FPN:
            fpn_enabled = not fpn_enabled
            if not fpn_enabled:
                fpn_corrector.reset()
        elif key in KEY_FPN_ONE:
            if single_frame_fpn_enabled:
                single_frame_fpn_enabled = False
                single_frame_offset_map = None
            else:
                if last_raw_bottom_half is not None:
                    single_frame_offset_map = (
                        last_raw_bottom_half.astype(np.float64) - np.mean(last_raw_bottom_half)
                    ).astype(np.float64)
                    single_frame_fpn_enabled = True
        elif key in KEY_LINE_PROFILE:
            line_profile_mode = not line_profile_mode
            if line_profile_mode:
                roi_state['active'] = False
                roi_state['start'] = roi_state['end'] = None
            else:
                line_profile['start'] = line_profile['end'] = None
                line_profile['dragging'] = False
        elif key in KEY_ISOTHERM:
            isotherm_enabled = not isotherm_enabled
        elif key in KEY_DELTA_T:
            if delta_t_enabled:
                delta_t_enabled = False
                delta_t_reference = None
            else:
                delta_t_reference = temp_celsius.copy()
                delta_t_enabled = True
        elif key in KEY_TREND:
            trend_enabled = not trend_enabled
            if not trend_enabled:
                trend_data.clear()
        elif key in KEY_HISTOGRAM:
            histogram_enabled = not histogram_enabled
        elif key in KEY_ANOMALY:
            anomaly_enabled = not anomaly_enabled
        elif key in KEY_DENOISE:
            denoise_enabled = not denoise_enabled
            if not denoise_enabled:
                denoiser.reset()
        elif key in KEY_ARROW_LEFT:
            rotation_deg = (rotation_deg - 90) % 360
        elif key in KEY_ARROW_RIGHT:
            rotation_deg = (rotation_deg + 90) % 360
        elif key in KEY_ARROW_UP:
            flip_v = not flip_v
        elif key in KEY_ARROW_DOWN:
            flip_h = not flip_h

    save_app_settings(palette_idx, dde_clip_limit, dde_sharp_strength, dde_sharp_sigma, palette_invert)
    save_panel_layout(panels)
    reader.running = False
    process.terminate()
    cv2.destroyAllWindows()
    if 'tray_icon' in locals():
        tray_icon.stop()

if __name__ == "__main__":
    main()
