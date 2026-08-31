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
import traceback
from collections import deque
from datetime import datetime
import pystray
from PIL import Image, ImageDraw

from fpn_correction import FPNCorrector
from denoise import ThermalDenoiser
from ibp_upscale import ibp_upscale
from burst_sr import BurstSR
from rgb_camera import find_rgb_camera_index, RGBCamera, GradientAligner

# macOS .app bundles do not inherit the shell PATH. Add common Homebrew paths:
os.environ['PATH'] += os.pathsep + '/opt/homebrew/bin' + os.pathsep + '/usr/local/bin'
FFMPEG_PATH = shutil.which('ffmpeg') or 'ffmpeg'

# `ffmpeg -list_devices` can hang (permission prompt, device in a bad state).
# Every call is bounded so the GUI thread can never be stuck on it.
DEVICE_SCAN_TIMEOUT_S = 5.0

APP_NAME = 'Infiray P2 Pro'
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _user_data_dir():
    """Writable per-user directory for settings.

    Must not be next to __file__: inside a PyInstaller .app that path is a
    read-only bundle (or a temp dir that is wiped on exit), so settings would
    silently never persist.
    """
    home = os.path.expanduser('~')
    if sys.platform == 'darwin':
        base = os.path.join(home, 'Library', 'Application Support', APP_NAME)
    elif os.name == 'nt':
        base = os.path.join(os.environ.get('APPDATA') or home, APP_NAME)
    else:
        base = os.path.join(os.environ.get('XDG_CONFIG_HOME') or os.path.join(home, '.config'), APP_NAME)
    try:
        os.makedirs(base, exist_ok=True)
        return base
    except OSError:
        return _SCRIPT_DIR


def _snapshot_dir():
    """Where snapshots go. CWD is unusable for a .app bundle (often '/')."""
    pictures = os.path.join(os.path.expanduser('~'), 'Pictures')
    if os.path.isdir(pictures):
        target = os.path.join(pictures, APP_NAME)
        try:
            os.makedirs(target, exist_ok=True)
            return target
        except OSError:
            return pictures
    return _user_data_dir()


def save_snapshot(img):
    """Write a snapshot PNG. Returns the path, or None if the write failed."""
    path = os.path.join(_snapshot_dir(), f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    try:
        ok = cv2.imwrite(path, img)
    except cv2.error:
        ok = False
    return path if ok else None

# All available OpenCV colormaps; P cycles through them. Optional ones guarded by hasattr.
PALETTES = []
if hasattr(cv2, 'COLORMAP_INFERNO'): PALETTES.append(("Inferno", cv2.COLORMAP_INFERNO))
PALETTES.append(("Jet", cv2.COLORMAP_JET))
PALETTES.append(("Hot", cv2.COLORMAP_HOT))
if hasattr(cv2, 'COLORMAP_PLASMA'): PALETTES.append(("Plasma", cv2.COLORMAP_PLASMA))
if hasattr(cv2, 'COLORMAP_MAGMA'): PALETTES.append(("Magma", cv2.COLORMAP_MAGMA))
if hasattr(cv2, 'COLORMAP_VIRIDIS'): PALETTES.append(("Viridis", cv2.COLORMAP_VIRIDIS))
PALETTES.append(("Rainbow", cv2.COLORMAP_RAINBOW))
PALETTES.append(("Bone", cv2.COLORMAP_BONE))
PALETTES.append(("Pink", cv2.COLORMAP_PINK))   # was Ocean; slot 9 = Pink
if hasattr(cv2, 'COLORMAP_TURBO'): PALETTES.append(("Turbo", cv2.COLORMAP_TURBO))
if hasattr(cv2, 'COLORMAP_PARULA'): PALETTES.append(("Parula", cv2.COLORMAP_PARULA))
PALETTES.append(("Autumn", cv2.COLORMAP_AUTUMN))
PALETTES.append(("Winter", cv2.COLORMAP_WINTER))
PALETTES.append(("Summer", cv2.COLORMAP_SUMMER))
PALETTES.append(("Spring", cv2.COLORMAP_SPRING))
PALETTES.append(("Cool", cv2.COLORMAP_COOL))
PALETTES.append(("HSV", cv2.COLORMAP_HSV))
if hasattr(cv2, 'COLORMAP_CIVIDIS'): PALETTES.append(("Cividis", cv2.COLORMAP_CIVIDIS))
if hasattr(cv2, 'COLORMAP_TWILIGHT'): PALETTES.append(("Twilight", cv2.COLORMAP_TWILIGHT))
if hasattr(cv2, 'COLORMAP_TWILIGHT_SHIFTED'): PALETTES.append(("TwilightShifted", cv2.COLORMAP_TWILIGHT_SHIFTED))
if hasattr(cv2, 'COLORMAP_DEEPGREEN'): PALETTES.append(("Deepgreen", cv2.COLORMAP_DEEPGREEN))

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
KEY_PALETTE_10 = {ord('!')}   # Shift+1
KEY_PALETTE_11 = {ord('@')}   # Shift+2
KEY_PALETTE_12 = {ord('#')}   # Shift+3
KEY_PALETTE_13 = {ord('$')}   # Shift+4
KEY_PALETTE_14 = {ord('%')}   # Shift+5
KEY_PALETTE_15 = {ord('^')}   # Shift+6
KEY_PALETTE_16 = {ord('&')}   # Shift+7
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
KEY_DENOISE_DOWN = {ord(','), ord('<')}
KEY_DENOISE_UP = {ord('.'), ord('>')}
KEY_IBP_UPSCALE = {ord('y'), ord('Y')}
KEY_IBP_STRENGTH_DOWN = {ord('j'), ord('J')}
KEY_IBP_STRENGTH_UP = {ord('k'), ord('K')}
KEY_BURST_SR = {ord('e'), ord('E')}  # E = burst/multi-frame SR (B is histogram)
KEY_VIRTUAL_CAM = {ord('u'), ord('U')}
KEY_RGB_CAM = {ord('o'), ord('O')}         # toggle RGB camera blend
KEY_RGB_CALIBRATE = {ord('/')}             # auto-calibrate alignment
KEY_RGB_BLEND_MODE = {ord(';')}            # cycle blend mode: full / hot_mask
KEY_RGB_HOT_UP = {ord("'")}               # raise hot-mask threshold +1 °C
KEY_RGB_HOT_DOWN = {ord('\\')}             # lower hot-mask threshold -1 °C
KEY_RGB_ALPHA_UP = {ord('`')}             # increase full-blend alpha +0.05
KEY_RGB_ALPHA_DOWN = {ord('~')}           # decrease full-blend alpha -0.05
KEY_RGB_ZOOM_IN  = {ord(']')}             # zoom in (when RGB on; else ] = DDE contrast)
KEY_RGB_ZOOM_OUT = {ord('[')}             # zoom out (when RGB on)
KEY_CONTRAST_MODE = {ord('}')}             # cycle contrast: off -> gamma -> percentile
# Arrow keys: macOS 63232-63235, GTK 65361-65364, some backends 0-3.
# Codes 81-84 are deliberately NOT listed: they are ord('Q'/'R'/'S'/'T'),
# already claimed by quit/ROI/snapshot/isotherm above, so they never reach here.
KEY_ARROW_LEFT = {63234, 65361, 2}
KEY_ARROW_RIGHT = {63235, 65363, 3}
KEY_ARROW_UP = {63232, 65362, 0}
KEY_ARROW_DOWN = {63233, 65364, 1}

# --- Bottom toolbar: two rows of clickable icon-buttons ---
TOOLBAR_H = 44
STATUS_BAR_H = 26   # Dark strip below image: left = mode/params, right = FPS (no overlap with image)
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
    ('fpn1',    '1FPN'), ('dnz',     'DNZ'),  ('dnz_dn',  'D-'),   ('dnz_up',  'D+'),
    ('ibp',     'IBP'),  ('ibp_dn',  'I-'),   ('ibp_up',  'I+'),   ('burst',   'BRST'),
    ('rot_l',   'RoL'),  ('rot_r',   'RoR'),
    ('flip_v',  'FlV'),  ('flip_h',  'FlH'),  ('vcam',    'VCAM'),
    ('rgb',     'RGB'),  ('rgb_cal', 'CAL'),  ('rgb_bld', 'BLD'),
    ('debug',   'DBG'),  ('help',    'HLP'),  ('quit',    'QUIT'),
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
    'fpn1': ord('x'), 'dnz': ord('g'), 'dnz_dn': ord(','), 'dnz_up': ord('.'),
    'ibp': ord('y'), 'ibp_dn': ord('j'), 'ibp_up': ord('k'), 'burst': ord('e'), 'rot_l': 63234, 'rot_r': 63235,
    'flip_v': 63232, 'flip_h': 63233, 'vcam': ord('u'),
    'rgb': ord('o'), 'rgb_cal': ord('/'), 'rgb_bld': ord(';'),
    'debug': ord('d'), 'help': ord('i'), 'quit': ord('q'),
}

_TOOLBAR_TOGGLES = {
    'minmax', 'freeze', 'fahr', 'invert', 'roi', 'line', 'iso', 'hist',
    'trend', 'anomaly', 'delta', 'alarm', 'dde', 'fpn', 'fpn1', 'dnz', 'ibp', 'burst',
    'flip_v', 'flip_h', 'vcam', 'rgb', 'debug', 'help',
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


def draw_status_bar(img, y0, dw, state):
    """Draw status bar on *img* at rows [y0, y0+STATUS_BAR_H): dark background, left=modes/params, right=FPS."""
    img[y0:y0 + STATUS_BAR_H, 0:dw] = (28, 28, 28)
    cv2.line(img, (0, y0), (dw, y0), (60, 60, 60), 1)
    font_scale = 0.42
    color = (200, 200, 200)
    y_text = y0 + 17
    # Left: palette, HQ, contrast, sharpness, then enabled flags
    cont_label = state.get('cont_label', 'Off')
    parts = [
        state.get('palette_name', 'Jet'),
        'HQ' if state.get('high_quality_mode') else '—',
        f"Cont:{state.get('dde_clip_limit', 0):.1f}({cont_label})",
        f"Shrp:{state.get('dde_sharp_strength', 0):.1f}",
    ]
    if state.get('single_frame_fpn_enabled'):
        parts.append('1FPN')
    if state.get('fpn_enabled'):
        parts.append('FPN')
    if state.get('denoise_enabled'):
        parts.append(f"DNZ{state.get('denoiser_strength', 0):.1f}")
    if state.get('ibp_upscale_enabled'):
        parts.append(f"IBP{state.get('ibp_strength', 0)}")
    if state.get('burst_sr_enabled'):
        parts.append(f"Burst{state.get('burst_frame_count', 0)}")
    if state.get('delta_t_enabled'):
        parts.append('dT')
    if state.get('isotherm_enabled'):
        parts.append('ISO')
    if state.get('anomaly_enabled'):
        parts.append('AI')
    left_text = '  |  '.join(parts)
    (fw_fps, _), _ = cv2.getTextSize(state.get('fps_text', '0.0 FPS'), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    max_left_w = dw - fw_fps - 24
    (left_w, _), _ = cv2.getTextSize(left_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    if left_w > max_left_w and len(parts) > 4:
        while parts and left_w > max_left_w:
            parts.pop()
            left_text = '  |  '.join(parts)
            (left_w, _), _ = cv2.getTextSize(left_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.putText(img, left_text, (8, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    # Right: FPS
    fps_text = state.get('fps_text', '0.0 FPS')
    (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(img, fps_text, (dw - fw - 8, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


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


def get_camera_index(timeout=DEVICE_SCAN_TIMEOUT_S):
    """AVFoundation index of the P2 Pro, or None.

    Never blocks longer than *timeout*: a hung `ffmpeg -list_devices` (camera
    permission prompt, device in a bad state) must not freeze the caller.
    """
    cmd = [FFMPEG_PATH, '-f', 'avfoundation', '-list_devices', 'true', '-i', '""']
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

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


def compute_fps(fps_times):
    """Frames/sec from timestamps of *completed* frames.

    With times t0..t_{n-1} there are (n-1) intervals, so FPS = (n-1)/(t_last-t_first).
    Using n/elapsed systematically overestimates (e.g. 30 samples @ 25 Hz → ~25.9).
    """
    if len(fps_times) < 2:
        return 0.0
    elapsed = fps_times[-1] - fps_times[0]
    if elapsed <= 0:
        return 0.0
    return (len(fps_times) - 1) / elapsed


def _read_exact(stream, size):
    """Read exactly `size` bytes; return bytes or short/empty on EOF or closed stream."""
    buf = b''
    while len(buf) < size:
        try:
            chunk = stream.read(size - len(buf))
        except (ValueError, OSError):
            return buf  # stream closed underneath us
        if not chunk:
            return buf  # b'' = EOF; None cannot happen on a blocking pipe
        buf += chunk
    return buf


PANELS_FILE = os.path.join(_user_data_dir(), 'layout.json')
SETTINGS_FILE = os.path.join(_user_data_dir(), 'settings.json')
# Files written by older versions, next to the script; read once if no new file exists.
_LEGACY_PANELS_FILE = os.path.join(_SCRIPT_DIR, '.p2pro_layout.json')
_LEGACY_SETTINGS_FILE = os.path.join(_SCRIPT_DIR, '.p2pro_settings.json')


def _read_json(path, legacy_path):
    """Load *path*; fall back to the legacy location only when it does not exist yet.

    A corrupt current file must not silently resurrect stale settings, so the
    fallback is tried only for a missing file, not for an unreadable one.
    """
    if not os.path.exists(path):
        path = legacy_path
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
_TITLE_H = 18
_GRIP = 10


class Panel:
    """Draggable, resizable overlay panel for graph widgets."""

    def __init__(self, name, x, y, w, h, min_w=100, min_h=60):
        self.name = name
        self.x, self.y, self.w, self.h = x, y, w, h
        self.min_w, self.min_h = min_w, min_h
        # Geometry the user asked for, kept separate from the rendered one.
        # clamp() fits the panel to the current display without writing back
        # here, so rotating the view (768x576 <-> 576x768) and rotating back
        # restores the layout instead of permanently shrinking it.
        self._want = [x, y, w, h]
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

    def start_drag(self, mx, my):
        self._dragging = True
        self._ox, self._oy = mx - self.x, my - self.y

    def start_resize(self, mx, my):
        self._resizing = True
        self._ox, self._oy = mx, my

    def on_move(self, mx, my, max_w, max_h):
        if self._dragging:
            self._want[0] = max(0, min(mx - self._ox, max_w - self.w))
            self._want[1] = max(0, min(my - self._oy, max_h - self.h))
        elif self._resizing:
            self._want[2] = max(self.min_w, self._want[2] + mx - self._ox)
            self._want[3] = max(self.min_h, self._want[3] + my - self._oy)
            self._ox, self._oy = mx, my
        else:
            return
        self.clamp(max_w, max_h)

    def stop(self):
        self._dragging = self._resizing = False

    @property
    def active(self):
        return self._dragging or self._resizing

    def clamp(self, max_w, max_h):
        """Fit the wanted geometry into the display. Does not mutate the wanted one."""
        wx, wy, ww, wh = self._want
        self.w = max(self.min_w, min(ww, max_w - 4))
        self.h = max(self.min_h, min(wh, max_h - 4))
        self.x = max(0, min(wx, max_w - self.w))
        self.y = max(0, min(wy, max_h - self.h))

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
        wx, wy, ww, wh = self._want
        return {'x': wx, 'y': wy, 'w': ww, 'h': wh}

    def load_dict(self, d, max_w, max_h):
        if not d:
            return
        self._want = [
            max(0, int(d.get('x', self._want[0]))),
            max(0, int(d.get('y', self._want[1]))),
            max(self.min_w, int(d.get('w', self._want[2]))),
            max(self.min_h, int(d.get('h', self._want[3]))),
        ]
        self.clamp(max_w, max_h)


def load_panel_layout(panels, max_w, max_h):
    data = _read_json(PANELS_FILE, _LEGACY_PANELS_FILE)
    if not isinstance(data, dict):
        return
    for name, p in panels.items():
        if name in data:
            try:
                p.load_dict(data[name], max_w, max_h)
            except (KeyError, TypeError, ValueError):
                pass


def save_panel_layout(panels):
    try:
        with open(PANELS_FILE, 'w') as f:
            json.dump({n: p.to_dict() for n, p in panels.items()}, f)
    except OSError:
        pass


def _clamp_num(v, lo, hi, cast=float):
    v = cast(v)
    return max(lo, min(hi, v))


def _one_of(v, choices):
    if v not in choices:
        raise ValueError(v)
    return v


# Everything the app restores between runs. Each entry validates one value;
# a bad or missing key falls back to the in-code default rather than failing
# the whole file. Keep this table as the single place a setting is declared.
_SETTINGS_SCHEMA = {
    'palette_idx':        lambda v, n: _clamp_num(v, 0, n - 1, int),
    'palette_invert':     lambda v, n: bool(v),
    'contrast_mode':      lambda v, n: _one_of(v, ('off', 'gamma', 'percentile', 'clahe')),
    'dde_clip_limit':     lambda v, n: _clamp_num(v, 0.0, 10.0),
    'dde_sharp_strength': lambda v, n: _clamp_num(v, 0.0, 10.0),
    'dde_sharp_sigma':    lambda v, n: _clamp_num(v, 0.2, 5.0),
    'high_quality_mode':  lambda v, n: bool(v),
    'use_fahrenheit':     lambda v, n: bool(v),
    'range_preset':       lambda v, n: _clamp_num(v, 0, 2, int),
    'rotation_deg':       lambda v, n: _one_of(int(v), (0, 90, 180, 270)),
    'flip_h':             lambda v, n: bool(v),
    'flip_v':             lambda v, n: bool(v),
    'denoise_strength':   lambda v, n: _clamp_num(v, 1.0, 10.0),
    'ibp_strength':       lambda v, n: _clamp_num(v, 1, 10, int),
    'alarm_threshold':    lambda v, n: _clamp_num(v, -20.0, 150.0),
    'rgb_blend_mode':     lambda v, n: _one_of(v, ('full', 'hot_mask')),
    'rgb_alpha':          lambda v, n: _clamp_num(v, 0.0, 1.0),
    'rgb_hot_min':        lambda v, n: _clamp_num(v, -20.0, 149.0),
    'rgb_zoom':           lambda v, n: _clamp_num(v, 0.5, 2.0),
}


def load_app_settings(num_palettes):
    """Load persisted settings. Unknown/invalid entries are dropped, not fatal."""
    data = _read_json(SETTINGS_FILE, _LEGACY_SETTINGS_FILE)
    if not isinstance(data, dict):
        return {}
    out = {}
    for key, validate in _SETTINGS_SCHEMA.items():
        if key not in data:
            continue
        try:
            out[key] = validate(data[key], num_palettes)
        except (TypeError, ValueError):
            pass   # one bad value must not discard the rest of the file
    return out


def save_app_settings(values):
    """Persist the subset of *values* the schema knows about."""
    try:
        payload = {k: values[k] for k in _SETTINGS_SCHEMA if k in values}
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except (OSError, TypeError):
        pass


def draw_line_profile_markers(colormap, start, end):
    """Draw measurement line and endpoint circles on the thermal image."""
    cv2.line(colormap, start, end, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(colormap, start, 5, (0, 255, 255), -1)
    cv2.circle(colormap, end, 5, (0, 255, 255), -1)


def draw_line_profile_graph(img, temp_celsius, start, end, scale, use_fahrenheit, panel):
    """Draw temperature profile graph in panel (on display image, rotation-independent)."""
    sx1, sy1 = start[0] // scale, start[1] // scale
    sx2, sy2 = end[0] // scale, end[1] // scale
    num_samples = max(int(np.hypot(sx2 - sx1, sy2 - sy1)), 2)
    xs = np.clip(np.linspace(sx1, sx2, num_samples).astype(int), 0, temp_celsius.shape[1] - 1)
    ys = np.clip(np.linspace(sy1, sy2, num_samples).astype(int), 0, temp_celsius.shape[0] - 1)
    temps = temp_celsius[ys, xs]

    panel.draw_frame(img, (0, 255, 255))
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
    cv2.polylines(img, [pts], False, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img, format_temp(t_max, use_fahrenheit), (gx, gy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, format_temp(t_min, use_fahrenheit), (gx, gy + gh + 10),
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


def draw_trend_graph(img, trend_data, use_fahrenheit, panel):
    """Draw cursor temperature trend in panel."""
    panel.draw_frame(img, (0, 255, 0))
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
    cv2.polylines(img, [pts], False, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(img, format_temp(t_max, use_fahrenheit), (gx, gy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, format_temp(t_min, use_fahrenheit), (gx, gy + gh + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, format_temp(temps[-1], use_fahrenheit), (gx + gw - 35, gy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 0), 1, cv2.LINE_AA)


def draw_histogram(img, temp_celsius, use_fahrenheit, panel):
    """Draw temperature distribution histogram in panel."""
    panel.draw_frame(img, (200, 200, 255))
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
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    cv2.putText(img, format_temp(t_min, use_fahrenheit), (hx, hy + hh + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1, cv2.LINE_AA)
    right_label = format_temp(t_max, use_fahrenheit)
    (lw, _), _ = cv2.getTextSize(right_label, cv2.FONT_HERSHEY_SIMPLEX, 0.25, 1)
    cv2.putText(img, right_label, (hx + hw - lw, hy + hh + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1, cv2.LINE_AA)


PIN_SIZE = 11   # marker length from tip to base; labels are offset clear of it


def draw_pin_marker(img, pos, color, tip_dir=(0, 1), size=PIN_SIZE):
    """Triangular pin whose TIP lands exactly on *pos*.

    *tip_dir* is a unit step in image coords pointing from the body toward the
    tip, so the body trails away on the opposite side and never covers the pixel
    being measured. (The old marker was centred on *pos* and pointed away from
    it, which read as an arrow aimed at nothing.)
    """
    x, y = int(pos[0]), int(pos[1])
    dx, dy = int(tip_dir[0]), int(tip_dir[1])
    bx, by = x - dx * size, y - dy * size      # centre of the base
    px, py = -dy, dx                           # perpendicular to the axis
    hw = size // 2 + 1
    pts = np.array([
        [x, y],
        [bx + px * hw, by + py * hw],
        [bx - px * hw, by - py * hw],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)


def draw_measure_label(img, text, anchor, color, above, dw, dh, scale=0.5, pad=9, clear=0):
    """Measurement text beside *anchor*, outlined so it survives any palette.

    Thin coloured glyphs alone were unreadable over a busy thermal image, so a
    dark stroke goes down first. *clear* is the height of a marker body sitting
    on the *above*/below side of the anchor, which the text is pushed past. The
    box is flipped and clamped to stay on screen.
    """
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x = anchor[0] + pad
    if x + tw > dw - 2:
        x = anchor[0] - pad - tw               # no room on the right: put it left
    x = max(2, min(x, dw - tw - 2))
    y = (anchor[1] - pad - clear) if above else (anchor[1] + pad + clear + th)
    y = max(th + 2, min(y, dh - base - 2))
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


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


FRAME_W, FRAME_H = 256, 384
FRAME_SIZE = FRAME_W * FRAME_H * 2  # YUYV: 2 bytes/pixel
RECONNECT_INTERVAL_S = 1.0
STALL_TIMEOUT_S = 2.0
# While frozen there is no new data; poll at UI rate instead of spinning.
FROZEN_POLL_MS = 30


class FrameReader(threading.Thread):
    def __init__(self, process, frame_size):
        super().__init__()
        self.process = process
        self.frame_size = frame_size
        self.latest_frame = None
        self.last_frame_time = 0.0  # monotonic time of last full frame
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
            # Drain pipe: if more frames are queued, read and drop until we have only the latest (real-time, no lag).
            # NOTE: this is only correct because stdout is unbuffered (Popen bufsize=0).
            # With a BufferedReader, select() sees the fd but not bytes already sitting
            # in Python's internal buffer, so a whole stale frame could stay hidden and
            # latency would creep up instead of being dropped.
            try:
                while select.select([stdout], [], [], 0)[0]:
                    next_bytes = _read_exact(stdout, self.frame_size)
                    if len(next_bytes) != self.frame_size:
                        self.running = False
                        break
                    raw_bytes = next_bytes
            except (ValueError, OSError):
                pass  # select on closed fd or unsupported
            if not self.running:
                break
            with self.lock:
                self.latest_frame = raw_bytes
                self.last_frame_time = time.monotonic()

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame

    def seconds_since_frame(self):
        with self.lock:
            t = self.last_frame_time
        if t <= 0:
            return None
        return time.monotonic() - t


def _ffmpeg_capture_command(cam_index):
    # -fflags nobuffer / -flags low_delay: minimize latency
    return [
        FFMPEG_PATH,
        '-hide_banner',
        '-loglevel', 'error',
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-f', 'avfoundation',
        '-framerate', '25',
        '-video_size', f'{FRAME_W}x{FRAME_H}',
        '-pix_fmt', 'yuyv422',
        '-i', str(cam_index),
        '-f', 'rawvideo',
        '-pix_fmt', 'yuyv422',
        '-',
    ]


def start_thermal_capture(cam_index=None):
    """Open P2 Pro via ffmpeg. Returns (process, reader, cam_index) or (None, None, None)."""
    if cam_index is None:
        cam_index = get_camera_index()
    if cam_index is None:
        return None, None, None
    process = subprocess.Popen(
        _ffmpeg_capture_command(cam_index),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,  # raw, unbuffered: keeps select() in FrameReader.run() truthful
    )
    reader = FrameReader(process, FRAME_SIZE)
    reader.start()
    return process, reader, cam_index


def stop_thermal_capture(process, reader):
    """Stop FrameReader and terminate ffmpeg (best-effort).

    Order matters: killing ffmpeg gives the reader EOF, so it leaves read().
    Only once the reader thread is gone may stdout be closed -- closing a
    stream another thread is reading from is a data race in the io layer.
    """
    if reader is not None:
        reader.running = False
    if process is not None:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    reader_gone = True
    if reader is not None:
        reader.join(timeout=2.0)
        reader_gone = not reader.is_alive()

    # Leave the pipe open if the reader is somehow still inside read();
    # the thread is a daemon and the fd is reclaimed when the process exits.
    if process is not None and reader_gone:
        try:
            if process.stdout:
                process.stdout.close()
        except Exception:
            pass


class CaptureOpener(threading.Thread):
    """Open the thermal capture off the GUI thread.

    Device enumeration goes through `ffmpeg -list_devices`, which costs a few
    hundred ms and can stall much longer. Doing it inline made the whole
    "Reconnecting..." screen unresponsive once per second.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.result = (None, None, None)
        self.done = threading.Event()

    def run(self):
        try:
            self.result = start_thermal_capture()
        except Exception as e:
            print("Capture open failed:", e)
            self.result = (None, None, None)
        finally:
            self.done.set()

    def abandon(self):
        """Called when quitting mid-attempt: tear down whatever we opened."""
        if self.done.wait(timeout=DEVICE_SCAN_TIMEOUT_S + 2.0):
            proc, rdr, _ = self.result
            if proc is not None or rdr is not None:
                stop_thermal_capture(proc, rdr)


def main():
    # Prefer SIMD OpenCV paths when available
    cv2.setUseOptimized(True)

    print("Searching for Infiray P2 Pro via ffmpeg...")
    process, reader, cam_index = start_thermal_capture()

    if process is None:
        print("Infiray P2 Pro not found — will keep trying. Plug it in or press Q to quit.")
    else:
        print(f"Found P2 Pro at AVFoundation index {cam_index}")
        print("Starting video stream.")
    print("Controls:")
    print("  'q' - Quit   'p' - Next Palette   'm' - Min/Max pointers   'd' - Debug")
    print("  SPACE - Freeze frame   's' - Snapshot   'f' - Celsius/Fahrenheit")
    print("  '1'-'9' '!@#$%^&' - Palette 1-16 (LUT)   'z' - Invert palette   '0' - Temp range cycle: Auto / Room / Wide")
    print("  'r' - ROI: draw rectangle for min/max/avg   'a' - High temp alarm")
    print("  'h' - High quality: DDE (Detail Enhancement) & Sharpening")
    print("  '['/']' - Decrease/Increase DDE Contrast   '-'/'=' - Decrease/Increase Sharpness")
    print("  'c' - FPN correction (running)   'x' - FPN from one frame   'i' - Help")
    print("  'l' - Line profile   't' - Isotherms   'v' - Delta-T mode")
    print("  'w' - Temp trend   'b' - Histogram   'n' - Anomaly detection")
    print("  Left/Right arrows - Rotate 90 deg   Up/Down arrows - Flip vertical/horizontal")
    print("  'o' - RGB blend   ';' - blend mode   '`'/'~' - blend alpha +/-   '/' - reset alignment")
    
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
    high_quality_mode = True   # 'h': DDE, on by default
    show_controls_overlay = False  # 'i': show all keys
    palette_invert = False  # 'z': invert colormap (hot/cold swap)
    fps_times = deque(maxlen=60)  # ~1–2 s window at 25–60 FPS; compute_fps uses (n-1)/dt
    last_seen_raw_bytes = None    # skip reprocessing / FPS inflation on duplicate frames
    _clahe = None                 # reused CLAHE (recreated when clip limit changes)
    _clahe_clip = None
    _ui_canvas = None             # reusable image+status+toolbar buffer (avoids vstack alloc)
    _frozen_cache = {'sig': None, 'img': None, 'image_h': 0}  # composed frozen canvas, rebuilt on state change
    virtual_cam_enabled = False  # 'u': send processed stream to virtual webcam (OBS on macOS)
    _virtual_cam = [None]  # pyvirtualcam.Camera instance when active
    _vcam_send_count = [0]  # for debug: frames sent to virtual cam
    _VCAM_SIZE = (640, 480)  # output size for messengers
    
    # DDE Parameters
    dde_clip_limit = 0.0     # Contrast amount (0 = no change); meaning depends on contrast_mode
    contrast_mode = 'gamma'  # 'off' | 'gamma' | 'percentile' | 'clahe'
    dde_sharp_strength = 8.0 # Sharpness strength for Unsharp Mask
    dde_sharp_sigma = 1.2    # Sharpness radius

    # FPN (fixed-pattern noise) correction: running per-pixel calibration
    fpn_enabled = False
    fpn_corrector = FPNCorrector((192, 256), alpha=0.995)
    # One-frame FPN: capture current frame on X, apply that offset to all following frames; X again = reset
    single_frame_fpn_enabled = False
    single_frame_offset_map = None  # float64 (192,256), offset to subtract each frame
    # Thermal de-noise: temporal EMA + Edge-Gated Local Mean (preserves edges)
    denoise_enabled = False
    denoiser = ThermalDenoiser((192, 256))
    ibp_upscale_enabled = False  # Iterative Back-Projection (training-free, good for thermal)
    ibp_strength = 10  # 1–10: max at 10 (30 iters, lam 1.4)
    burst_sr_enabled = False  # multi-frame SR from subpixel shifts (Samsung-style)
    burst_sr = BurstSR((192, 256), scale=SCALE, max_frames=12)
    last_raw_bottom_half = None    # last raw thermal frame, for X capture

    # RGB camera blend state
    rgb_cam_enabled = False        # 'o': toggle second (RGB) camera overlay
    rgb_blend_mode = 'full'        # 'full' | 'hot_mask'
    rgb_alpha = 0.4                # blend strength for full mode
    rgb_hot_min = 35.0             # temperature threshold for hot_mask mode (°C)
    rgb_aligner = GradientAligner()
    _rgb_cam = [None]              # RGBCamera instance, created on first enable

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
        # Identity still copies once (HUD draws on the result without touching source).
        # With rotation/flip, OpenCV already allocates — skip the extra upfront copy.
        if rotation_deg == 0 and not flip_h and not flip_v:
            return img.copy()
        out = img
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

    def screen_down_in_logical():
        """Logical unit step that appears as 'downward' on screen.

        Markers are drawn on the un-rotated colormap but their labels are placed
        in display space, so the pin has to be aimed through the same transform
        or it points sideways as soon as the view is rotated.
        """
        ax, ay = logical_to_display(WIDTH // 2, HEIGHT // 2)
        _, by = logical_to_display(WIDTH // 2, HEIGHT // 2 + 1)
        if by != ay:
            return (0, 1) if by > ay else (0, -1)
        _, cy = logical_to_display(WIDTH // 2 + 1, HEIGHT // 2)
        return (1, 0) if cy > ay else (-1, 0)

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
    # Keys are only delivered during waitKey/pollKey; queue them so idle spins don't drop presses.
    _pending_keys = deque()

    print("Note: AI Upscaling disabled in favor of DDE (Digital Detail Enhancement) for pure sharpness.")

    
    def on_mouse(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y

        tb_dh = _tb_display_dims[1]
        toolbar_y0 = tb_dh + STATUS_BAR_H
        if y >= toolbar_y0:
            if event == cv2.EVENT_LBUTTONDOWN:
                bid = _toolbar_find_button(x, y - toolbar_y0, _tb_display_dims[0])
                if bid and bid in _TOOLBAR_KEY_MAP:
                    _toolbar_injected_key[0] = _TOOLBAR_KEY_MAP[bid]
            elif event == cv2.EVENT_MOUSEMOVE:
                _tb_hover_id[0] = _toolbar_find_button(x, y - toolbar_y0, _tb_display_dims[0])
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
                if p.hit_grip(x, y):
                    p.start_resize(x, y)
                    _panel_drag['ref'] = p
                    return
                if p.hit_title(x, y):
                    p.start_drag(x, y)
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
                ap.on_move(x, y, _tb_display_dims[0], _tb_display_dims[1])
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

    def _send_to_virtual_cam(img, where="live"):
        """Push one frame to the virtual camera. Returns False if the device is gone."""
        cam = _virtual_cam[0]
        if cam is None:
            return False
        try:
            vframe = cv2.resize(img, _VCAM_SIZE, interpolation=cv2.INTER_LINEAR)
            cam.send(cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"VCAM send ({where}) failed:", e)
            try:
                cam.close()
            except Exception:
                pass
            _virtual_cam[0] = None
            return False
        _vcam_send_count[0] += 1
        if _vcam_send_count[0] <= 3:   # confirm it works, then stay quiet
            print(f"VCAM send #{_vcam_send_count[0]}  {_VCAM_SIZE[0]}x{_VCAM_SIZE[1]}")
        return True

    def window_closed():
        try:
            return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def poll_keys(wait_ms=1):
        """Pump HighGUI events and enqueue all pending keys (keep rapid presses)."""
        k = cv2.waitKey(wait_ms)
        if k >= 0:
            _pending_keys.append(int(k) & 0xFFFF)
        if hasattr(cv2, 'pollKey'):
            while True:
                k = cv2.pollKey()
                if k < 0:
                    break
                _pending_keys.append(int(k) & 0xFFFF)
        else:
            for _ in range(16):
                k = cv2.waitKey(1)
                if k < 0:
                    break
                _pending_keys.append(int(k) & 0xFFFF)
        inj = _toolbar_injected_key[0]
        if inj >= 0:
            _pending_keys.append(int(inj) & 0xFFFF)
            _toolbar_injected_key[0] = -1

    palette_idx = 0
    temp_celsius = None
    colormap = None
    display_img = None
    palette_name = PALETTES[0][0]   # bound before any frame so the frozen path is safe
    fps = 0.0

    # Restore everything the schema knows about. Orientation especially matters:
    # re-setting rotation on every launch is pure friction for a fixed mount.
    _saved = load_app_settings(len(PALETTES))
    palette_idx        = _saved.get('palette_idx', palette_idx)
    palette_invert     = _saved.get('palette_invert', palette_invert)
    contrast_mode      = _saved.get('contrast_mode', contrast_mode)
    dde_clip_limit     = _saved.get('dde_clip_limit', dde_clip_limit)
    dde_sharp_strength = _saved.get('dde_sharp_strength', dde_sharp_strength)
    dde_sharp_sigma    = _saved.get('dde_sharp_sigma', dde_sharp_sigma)
    high_quality_mode  = _saved.get('high_quality_mode', high_quality_mode)
    use_fahrenheit     = _saved.get('use_fahrenheit', use_fahrenheit)
    range_preset       = _saved.get('range_preset', range_preset)
    rotation_deg       = _saved.get('rotation_deg', rotation_deg)
    flip_h             = _saved.get('flip_h', flip_h)
    flip_v             = _saved.get('flip_v', flip_v)
    ibp_strength       = _saved.get('ibp_strength', ibp_strength)
    alarm_threshold    = _saved.get('alarm_threshold', alarm_threshold)
    rgb_blend_mode     = _saved.get('rgb_blend_mode', rgb_blend_mode)
    rgb_alpha          = _saved.get('rgb_alpha', rgb_alpha)
    rgb_hot_min        = _saved.get('rgb_hot_min', rgb_hot_min)
    denoiser.strength  = _saved.get('denoise_strength', denoiser.strength)
    rgb_aligner.set_zoom(_saved.get('rgb_zoom', rgb_aligner.get_zoom()))

    _tray_stopped = [False]

    def on_quit(icon, item):
        global running
        running = False
        _tray_stopped[0] = True   # stopping twice raises on some pystray backends
        try:
            icon.stop()
        except Exception:
            pass

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

    reconnecting = process is None
    reconnect_attempt = 0
    last_reconnect_try = 0.0
    capture_opener = None    # CaptureOpener while an open attempt is in flight
    capture_started_at = time.monotonic() if process is not None else 0.0
    last_display_img = None  # last live frame for reconnect overlay
    last_image_h = HEIGHT    # height of the image area inside last_display_img
    loop_errors = 0
    last_loop_error = 0.0
    MAX_LOOP_ERRORS = 10

    def apply_live_keys(from_frozen=False):
        """Apply all queued keys. Returns False if the app should quit."""
        nonlocal palette_idx, range_preset, show_min_max, debug_mode, frozen, frozen_frame
        nonlocal use_fahrenheit, palette_invert, line_profile_mode, alarm_enabled
        nonlocal high_quality_mode, dde_clip_limit, contrast_mode, dde_sharp_strength
        nonlocal show_controls_overlay, virtual_cam_enabled, fpn_enabled
        nonlocal single_frame_fpn_enabled, single_frame_offset_map
        nonlocal isotherm_enabled, delta_t_enabled, delta_t_reference
        nonlocal trend_enabled, histogram_enabled, anomaly_enabled
        nonlocal denoise_enabled, ibp_upscale_enabled, burst_sr_enabled, ibp_strength
        nonlocal rgb_cam_enabled, rgb_blend_mode, rgb_hot_min, rgb_alpha
        nonlocal rotation_deg, flip_v, flip_h

        while _pending_keys:
            key = _pending_keys.popleft()
            if key in KEY_QUIT:
                return False
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
            elif key in KEY_PALETTE_10:
                palette_idx = min(9, len(PALETTES) - 1)
            elif key in KEY_PALETTE_11:
                palette_idx = min(10, len(PALETTES) - 1)
            elif key in KEY_PALETTE_12:
                palette_idx = min(11, len(PALETTES) - 1)
            elif key in KEY_PALETTE_13:
                palette_idx = min(12, len(PALETTES) - 1)
            elif key in KEY_PALETTE_14:
                palette_idx = min(13, len(PALETTES) - 1)
            elif key in KEY_PALETTE_15:
                palette_idx = min(14, len(PALETTES) - 1)
            elif key in KEY_PALETTE_16:
                palette_idx = min(15, len(PALETTES) - 1)
            elif key in KEY_RANGE_CYCLE:
                range_preset = (range_preset + 1) % 3
            elif key in KEY_MINMAX:
                show_min_max = not show_min_max
            elif key in KEY_DEBUG:
                debug_mode = not debug_mode
            elif key in KEY_SPACE:
                if from_frozen:
                    frozen = False
                elif colormap is not None:
                    frozen_frame = colormap.copy()
                    frozen = True
            elif key in KEY_SNAPSHOT:
                shot = None
                if from_frozen and frozen_frame is not None:
                    shot = apply_rotate_flip(frozen_frame)
                elif display_img is not None:
                    shot = display_img
                if shot is not None:
                    saved = save_snapshot(shot)
                    print(f"Saved {saved}" if saved else "Snapshot failed: could not write file")
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
            elif key in KEY_RGB_ZOOM_OUT and rgb_cam_enabled:
                rgb_aligner.set_zoom(rgb_aligner.get_zoom() - 0.05)
            elif key in KEY_RGB_ZOOM_IN and rgb_cam_enabled:
                rgb_aligner.set_zoom(rgb_aligner.get_zoom() + 0.05)
            elif key in KEY_BRACKET_L:
                dde_clip_limit = max(0.0, dde_clip_limit - 0.5)
            elif key in KEY_BRACKET_R:
                dde_clip_limit += 0.5
            elif key in KEY_CONTRAST_MODE:
                contrast_mode = {'off': 'gamma', 'gamma': 'percentile', 'percentile': 'clahe', 'clahe': 'off'}[contrast_mode]
            elif key in KEY_SHARP_MINUS:
                dde_sharp_strength = max(0.0, dde_sharp_strength - 0.5)
            elif key in KEY_SHARP_PLUS:
                dde_sharp_strength += 0.5
            elif key in KEY_HELP:
                show_controls_overlay = not show_controls_overlay
            elif key in KEY_VIRTUAL_CAM:
                virtual_cam_enabled = not virtual_cam_enabled
                if not virtual_cam_enabled and _virtual_cam[0] is not None:
                    try:
                        _virtual_cam[0].close()
                    except Exception:
                        pass
                    _virtual_cam[0] = None
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
                if temp_celsius is None:
                    pass
                elif delta_t_enabled:
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
            elif key in KEY_DENOISE_DOWN:
                denoiser.adjust(-denoiser.STRENGTH_STEP)
            elif key in KEY_DENOISE_UP:
                denoiser.adjust(denoiser.STRENGTH_STEP)
            elif key in KEY_IBP_UPSCALE:
                ibp_upscale_enabled = not ibp_upscale_enabled
            elif key in KEY_BURST_SR:
                burst_sr_enabled = not burst_sr_enabled
                if not burst_sr_enabled:
                    burst_sr.reset()
            elif key in KEY_IBP_STRENGTH_DOWN:
                ibp_strength = max(1, ibp_strength - 1)
            elif key in KEY_IBP_STRENGTH_UP:
                ibp_strength = min(10, ibp_strength + 1)
            elif key in KEY_RGB_CAM:
                rgb_cam_enabled = not rgb_cam_enabled
                if rgb_cam_enabled:
                    if _rgb_cam[0] is None:
                        cam_idx = find_rgb_camera_index(cam_index)
                        if cam_idx is not None:
                            cam = RGBCamera(cam_idx)
                            if cam.start():
                                _rgb_cam[0] = cam
                            else:
                                print(f"RGB camera: failed to open index {cam_idx}")
                                rgb_cam_enabled = False
                        else:
                            print("RGB camera: no suitable camera found")
                            rgb_cam_enabled = False
                else:
                    if _rgb_cam[0] is not None:
                        _rgb_cam[0].stop()
                        _rgb_cam[0] = None
                    rgb_aligner.reset()
            elif key in KEY_RGB_CALIBRATE:
                rgb_aligner.reset()
                print("RGB gradient aligner: reset.")
            elif key in KEY_RGB_BLEND_MODE:
                rgb_blend_mode = 'hot_mask' if rgb_blend_mode == 'full' else 'full'
            elif key in KEY_RGB_HOT_UP:
                rgb_hot_min = min(rgb_hot_min + 1.0, 149.0)
            elif key in KEY_RGB_HOT_DOWN:
                rgb_hot_min = max(rgb_hot_min - 1.0, -20.0)
            elif key in KEY_RGB_ALPHA_UP:
                rgb_alpha = round(min(rgb_alpha + 0.05, 1.0), 2)
            elif key in KEY_RGB_ALPHA_DOWN:
                rgb_alpha = round(max(rgb_alpha - 0.05, 0.0), 2)
            elif key in KEY_ARROW_LEFT:
                rotation_deg = (rotation_deg - 90) % 360
            elif key in KEY_ARROW_RIGHT:
                rotation_deg = (rotation_deg + 90) % 360
            elif key in KEY_ARROW_UP:
                flip_v = not flip_v
            elif key in KEY_ARROW_DOWN:
                flip_h = not flip_h
        return True

    while running:
        try:
            # --- Auto-reconnect: EOF/dead reader or stalled stream ---
            stream_dead = reader is None or not reader.running
            stalled = False
            if reader is not None and reader.running:
                age = reader.seconds_since_frame()
                if age is not None and age > STALL_TIMEOUT_S:
                    stalled = True
                elif age is None and capture_started_at > 0 and (time.monotonic() - capture_started_at) > STALL_TIMEOUT_S:
                    # Opened but never delivered a frame
                    stalled = True

            if stream_dead or stalled:
                if not reconnecting or stalled:
                    reason = "stalled (no frames)" if stalled else "disconnected"
                    print(f"Camera {reason} — reconnecting...")
                if reader is not None or process is not None:
                    stop_thermal_capture(process, reader)
                    process, reader = None, None
                reconnecting = True
                capture_started_at = 0.0

            if reconnecting:
                now = time.monotonic()
                # Opening goes through a worker thread: device enumeration blocks for
                # a few hundred ms (sometimes much longer) and must not stall the UI.
                if process is None and capture_opener is None and now - last_reconnect_try >= RECONNECT_INTERVAL_S:
                    reconnect_attempt += 1
                    capture_opener = CaptureOpener()
                    capture_opener.start()
                if capture_opener is not None and capture_opener.done.is_set():
                    process, reader, cam_index = capture_opener.result
                    capture_opener = None
                    last_reconnect_try = time.monotonic()  # back off from the end of the attempt
                    if process is not None:
                        capture_started_at = time.monotonic()
                        print(f"Camera found (index {cam_index}), waiting for frames...")

                # Overlay on last frame (or black placeholder)
                if last_display_img is not None:
                    recon_img = last_display_img.copy()
                    img_h = last_image_h
                elif frozen and frozen_frame is not None:
                    recon_img = apply_rotate_flip(frozen_frame.copy())
                    img_h = recon_img.shape[0]
                else:
                    recon_img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                    img_h = HEIGHT
                rh, rw = recon_img.shape[:2]
                img_h = max(1, min(img_h, rh))
                dim = recon_img.copy()
                cv2.rectangle(dim, (0, 0), (rw, rh), (20, 20, 20), -1)
                cv2.addWeighted(dim, 0.55, recon_img, 0.45, 0, recon_img)
                msg = f"Reconnecting... (attempt {reconnect_attempt})"
                (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.putText(recon_img, msg, ((rw - tw) // 2, img_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(recon_img, "Q to quit", ((rw - 80) // 2, img_h // 2 + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
                cv2.imshow(window_name, recon_img)
                poll_keys(50)
                if window_closed():
                    break
                if any(k in KEY_QUIT for k in _pending_keys):
                    break
                _pending_keys.clear()  # ignore other keys while reconnecting

                # Cleared only after a real frame is available
                if reader is not None and reader.get_latest_frame() is not None:
                    print("Camera reconnected.")
                    reconnecting = False
                    reconnect_attempt = 0
                    fps_times.clear()       # otherwise the outage is averaged into the rate
                    last_seen_raw_bytes = None
                    denoiser.reset()        # temporal history belongs to the old stream
                    burst_sr.reset()
                    # Fall through to normal frame path this iteration
                else:
                    continue

            if frozen and frozen_frame is not None:
                # The picture does not change while frozen, so recompose it only when
                # something that affects it actually changed. Rebuilding every spin
                # cost ~740 full redraws/s of an identical image.
                frozen_sig = (
                    id(frozen_frame), rotation_deg, flip_h, flip_v, show_controls_overlay,
                    palette_name, high_quality_mode, contrast_mode, dde_clip_limit,
                    dde_sharp_strength, single_frame_fpn_enabled, fpn_enabled,
                    denoise_enabled, denoiser.strength, ibp_upscale_enabled, ibp_strength,
                    burst_sr_enabled, delta_t_enabled, isotherm_enabled, anomaly_enabled,
                    show_min_max, use_fahrenheit, palette_invert, roi_state['active'],
                    line_profile_mode, histogram_enabled, trend_enabled, alarm_enabled,
                    virtual_cam_enabled, rgb_cam_enabled, debug_mode, _tb_hover_id[0],
                )
                if _frozen_cache['sig'] == frozen_sig and _frozen_cache['img'] is not None:
                    display_frozen = _frozen_cache['img']
                    df_h = _frozen_cache['image_h']
                    df_w = display_frozen.shape[1]
                    _tb_display_dims[0], _tb_display_dims[1] = df_w, df_h
                    last_display_img = display_frozen
                    last_image_h = df_h
                    if virtual_cam_enabled and _virtual_cam[0] is not None:
                        if not _send_to_virtual_cam(display_frozen[:df_h], "frozen"):
                            virtual_cam_enabled = False
                    cv2.imshow(window_name, display_frozen)
                    poll_keys(FROZEN_POLL_MS)
                    if window_closed():
                        break
                    if not apply_live_keys(from_frozen=True):
                        break
                    continue

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
                    cv2.putText(display_frozen, "1-9 !@#$%^& - Palette 1-16   0 - Range   R - ROI   A - Alarm   H - DDE   [ ] = - DDE   I - Help", (20, y0), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
                if virtual_cam_enabled and _virtual_cam[0] is not None:
                    if not _send_to_virtual_cam(display_frozen, "frozen"):
                        virtual_cam_enabled = False
                _tb_display_dims[0], _tb_display_dims[1] = df_w, df_h
                cont_label = "Off" if contrast_mode == 'off' or dde_clip_limit <= 0 else ("CLAHE" if contrast_mode == 'clahe' else contrast_mode.capitalize())
                frozen_status_state = {
                    'palette_name': palette_name, 'high_quality_mode': high_quality_mode,
                    'cont_label': cont_label, 'dde_clip_limit': dde_clip_limit, 'dde_sharp_strength': dde_sharp_strength,
                    'single_frame_fpn_enabled': single_frame_fpn_enabled, 'fpn_enabled': fpn_enabled,
                    'denoise_enabled': denoise_enabled, 'denoiser_strength': denoiser.strength if denoise_enabled else 0,
                    'ibp_upscale_enabled': ibp_upscale_enabled, 'ibp_strength': ibp_strength,
                    'burst_sr_enabled': burst_sr_enabled, 'burst_frame_count': burst_sr.frame_count if burst_sr_enabled else 0,
                    'delta_t_enabled': delta_t_enabled, 'isotherm_enabled': isotherm_enabled, 'anomaly_enabled': anomaly_enabled,
                    'fps_text': '— FPS',
                }
                fstatus_strip = np.zeros((STATUS_BAR_H, df_w, 3), dtype=np.uint8)
                display_frozen = np.vstack([display_frozen, fstatus_strip])
                draw_status_bar(display_frozen, df_h, df_w, frozen_status_state)
                frozen_tb_states = {
                    'minmax': show_min_max, 'freeze': True, 'fahr': use_fahrenheit,
                    'invert': palette_invert, 'roi': roi_state['active'], 'line': line_profile_mode,
                    'iso': isotherm_enabled, 'hist': histogram_enabled,
                    'trend': trend_enabled, 'anomaly': anomaly_enabled,
                    'delta': delta_t_enabled, 'alarm': alarm_enabled,
                    'dde': high_quality_mode, 'fpn': fpn_enabled,
                    'fpn1': single_frame_fpn_enabled, 'dnz': denoise_enabled,
                    'ibp': ibp_upscale_enabled, 'burst': burst_sr_enabled,
                    'flip_v': flip_v, 'flip_h': flip_h, 'vcam': virtual_cam_enabled,
                    'rgb': rgb_cam_enabled,
                    'debug': debug_mode, 'help': show_controls_overlay,
                }
                ftb = np.zeros((TOOLBAR_H, df_w, 3), dtype=np.uint8)
                display_frozen = np.vstack([display_frozen, ftb])
                draw_toolbar(display_frozen, df_h + STATUS_BAR_H, df_w, frozen_tb_states, _tb_hover_id[0])
                _frozen_cache['sig'] = frozen_sig
                _frozen_cache['img'] = display_frozen
                _frozen_cache['image_h'] = df_h
                last_display_img = display_frozen
                last_image_h = df_h
                cv2.imshow(window_name, display_frozen)
                poll_keys(FROZEN_POLL_MS)
                if window_closed():
                    break
                if not apply_live_keys(from_frozen=True):
                    break
                continue

            if reader is None:
                reconnecting = True
                continue

            raw_bytes = reader.get_latest_frame()
            # No new frame: still pump/apply keys (do not discard waitKey results).
            if raw_bytes is None or raw_bytes is last_seen_raw_bytes:
                poll_keys(1)
                if not _pending_keys:
                    continue
                if last_display_img is not None:
                    cv2.imshow(window_name, last_display_img)
                if window_closed():
                    break
                if not apply_live_keys():
                    break
                continue
            last_seen_raw_bytes = raw_bytes

            frame_count += 1
            fps_times.append(time.perf_counter())

            # Convert bottom half to 16-bit array (one copy; last_raw shares until FPN reassigns)
            bottom_half_bytes = raw_bytes[256 * 192 * 2:]
            bottom_half = np.frombuffer(bottom_half_bytes, dtype=np.uint16).reshape((192, 256)).copy()
            last_raw_bottom_half = bottom_half

            # One-frame FPN: subtract offset from captured frame (X) to all frames
            if single_frame_fpn_enabled and single_frame_offset_map is not None:
                bottom_half = np.clip(
                    bottom_half.astype(np.float32) - single_frame_offset_map, 0, 65535
                ).astype(np.uint16)

            # Running FPN correction (removes fixed per-pixel offset pattern over time)
            if fpn_enabled:
                bottom_half = fpn_corrector.apply(bottom_half)

            # Calculate temperature (float32 is enough and faster than float64)
            temp_celsius = bottom_half.astype(np.float32) * (1.0 / 64.0) - np.float32(273.15)
            raw_min_c = float(temp_celsius.min())
            raw_max_c = float(temp_celsius.max())
            # Filter out obvious outliers
            temp_celsius = np.clip(temp_celsius, -20, 150)

            # Temporal + spatial de-noise (G)
            if denoise_enabled:
                temp_celsius = denoiser.apply(temp_celsius)

            # Min/max in two passes (argmin/argmax) instead of four (min+max+argmin+argmax)
            min_flat = int(temp_celsius.argmin())
            max_flat = int(temp_celsius.argmax())
            min_temp = float(temp_celsius.flat[min_flat])
            max_temp = float(temp_celsius.flat[max_flat])
            if range_preset == 1:
                range_min, range_max = 15.0, 35.0
            elif range_preset == 2:
                range_min, range_max = -20.0, 150.0
            else:
                range_min, range_max = min_temp, max_temp

            min_y, min_x = divmod(min_flat, temp_celsius.shape[1])
            max_y, max_x = divmod(max_flat, temp_celsius.shape[1])
            min_x, min_y = min_x * SCALE, min_y * SCALE
            max_x, max_y = max_x * SCALE, max_y * SCALE

            # Normalize to 8-bit BEFORE upscaling
            delta_display = None
            if delta_t_enabled and delta_t_reference is not None:
                delta_display = temp_celsius - delta_t_reference
                abs_range = max(abs(float(delta_display.min())), abs(float(delta_display.max())), 0.1)
                normalized_small = ((delta_display + abs_range) / (2 * abs_range) * 255).astype(np.uint8)
            elif range_max > range_min:
                normalized_small = (
                    (temp_celsius - range_min) * (np.float32(255.0) / np.float32(range_max - range_min))
                ).astype(np.uint8)
            else:
                normalized_small = np.zeros_like(temp_celsius, dtype=np.uint8)

            # Upscale
            if high_quality_mode:
                # DDE (Digital Detail Enhancement) for Thermal Images
                # 1. Contrast: gamma (global), percentile (global), or CLAHE (local)
                if dde_clip_limit > 0 and contrast_mode != 'off':
                    x = normalized_small.astype(np.float32) * (1.0 / 255.0)
                    if contrast_mode == 'gamma':
                        gamma = 0.5 + dde_clip_limit * 0.15
                        enhanced = (np.power(np.clip(x, 1e-6, 1), gamma) * 255).astype(np.uint8)
                    elif contrast_mode == 'percentile':
                        lo = float(np.percentile(normalized_small, 2))
                        hi = float(np.percentile(normalized_small, 98))
                        if hi > lo:
                            stretched = ((normalized_small.astype(np.float32) - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
                        else:
                            stretched = normalized_small
                        alpha = min(1.0, dde_clip_limit / 5.0)
                        enhanced = (normalized_small.astype(np.float32) * (1 - alpha) + stretched.astype(np.float32) * alpha).astype(np.uint8)
                    else:  # clahe (local adaptive) — reuse instance across frames
                        if _clahe is None or _clahe_clip != dde_clip_limit:
                            _clahe = cv2.createCLAHE(clipLimit=dde_clip_limit, tileGridSize=(8, 8))
                            _clahe_clip = dde_clip_limit
                        enhanced = _clahe.apply(normalized_small)
                else:
                    enhanced = normalized_small

                # 2. Upscale: Burst SR (E) | IBP (Y) | Lanczos4
                if burst_sr_enabled:
                    burst_sr.push(enhanced)
                    hr = burst_sr.get_hr()
                    if hr is not None:
                        normalized = hr
                    else:
                        normalized = cv2.resize(enhanced, (WIDTH, HEIGHT), interpolation=cv2.INTER_LANCZOS4)
                elif ibp_upscale_enabled:
                    n_iters = 3 + (ibp_strength - 1) * 3  # 3..30, max strength
                    lam = 0.3 + (ibp_strength - 1) * 0.12  # 0.3..1.38
                    normalized = ibp_upscale(enhanced, SCALE, n_iters=n_iters, lam=lam)
                else:
                    normalized = cv2.resize(enhanced, (WIDTH, HEIGHT), interpolation=cv2.INTER_LANCZOS4)

                # 3. Aggressive sharpening to make edges pop without hallucinating
                if dde_sharp_strength > 0:
                    normalized = unsharp_mask(normalized, sigma=dde_sharp_sigma, strength=dde_sharp_strength)
            else:
                if burst_sr_enabled:
                    burst_sr.push(normalized_small)
                    hr = burst_sr.get_hr()
                    if hr is not None:
                        normalized = hr
                    else:
                        normalized = cv2.resize(normalized_small, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
                elif ibp_upscale_enabled:
                    n_iters = 3 + (ibp_strength - 1) * 3
                    lam = 0.3 + (ibp_strength - 1) * 0.12
                    normalized = ibp_upscale(normalized_small, SCALE, n_iters=n_iters, lam=lam)
                else:
                    normalized = cv2.resize(normalized_small, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)

            # Apply selected palette
            palette_name, colormap_flag = PALETTES[palette_idx]
            to_map = (255 - normalized) if palette_invert else normalized
            colormap = cv2.applyColorMap(to_map, colormap_flag)

            # --- RGB camera blend ---
            if rgb_cam_enabled and _rgb_cam[0] is not None:
                rgb_frame_raw = _rgb_cam[0].get_frame()
                if rgb_frame_raw is not None:
                    # Gradient/edge-based alignment: update every N frames (ECC on gradient maps)
                    rgb_aligner.update(rgb_frame_raw, colormap)
                    if rgb_aligner.is_tracking:
                        rgb_w = rgb_aligner.warp(rgb_frame_raw, (WIDTH, HEIGHT))
                        if rgb_blend_mode == 'full':
                            colormap = cv2.addWeighted(
                                rgb_w, rgb_alpha,
                                colormap, 1.0 - rgb_alpha,
                                0,
                            )
                        else:  # hot_mask: RGB is base, thermal overlaid at hot pixels
                            hot = (temp_celsius >= rgb_hot_min).astype(np.uint8) * 255
                            hot_up = cv2.resize(hot, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
                            mask = cv2.GaussianBlur(hot_up, (21, 21), 0).astype(np.float32) / 255.0
                            mask3 = mask[:, :, np.newaxis]
                            colormap = (
                                colormap.astype(np.float32) * mask3
                                + rgb_w.astype(np.float32) * (1.0 - mask3)
                            ).astype(np.uint8)

            if isotherm_enabled:
                draw_isotherms(colormap, temp_celsius, range_min, range_max, SCALE, use_fahrenheit)

            if anomaly_enabled:
                draw_anomaly_overlay(colormap, temp_celsius, 2.0, SCALE, use_fahrenheit)

            # Get temperature at mouse cursor
            if 0 <= mouse_x < WIDTH and 0 <= mouse_y < HEIGHT:
                cursor_temp = temp_celsius[mouse_y // SCALE, mouse_x // SCALE]
            else:
                cursor_temp = 0.0
        
            if show_min_max:
                _sd = screen_down_in_logical()
                _su = (-_sd[0], -_sd[1])
                # MAX is labelled above the point, so its body sits above and the
                # tip points down at it; MIN is labelled below and points up.
                draw_pin_marker(colormap, (max_x, max_y), (40, 60, 255), _sd, size=PIN_SIZE)
                draw_pin_marker(colormap, (min_x, min_y), (255, 140, 40), _su, size=PIN_SIZE)
        
            if trend_enabled:
                trend_data.append(cursor_temp)

            # Draw crosshair at cursor (text drawn on display later)
            cv2.drawMarker(colormap, (mouse_x, mouse_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
            cursor_label = format_temp(cursor_temp, use_fahrenheit)
            if delta_display is not None:
                d = delta_display[mouse_y // SCALE, mouse_x // SCALE]
                cursor_label += f" ({'+' if d >= 0 else ''}{d:.1f})"
        
            roi_stats = None   # (x1, y1, roi_min, roi_avg, roi_max) for the label below
            if roi_state['start'] and roi_state['end']:
                x1, y1 = min(roi_state['start'][0], roi_state['end'][0]), min(roi_state['start'][1], roi_state['end'][1])
                x2, y2 = max(roi_state['start'][0], roi_state['end'][0]), max(roi_state['start'][1], roi_state['end'][1])
                x1, x2 = max(0, x1), min(WIDTH, x2)
                y1, y2 = max(0, y1), min(HEIGHT, y2)
                if x2 > x1 and y2 > y1:
                    sx1, sy1 = x1 // SCALE, y1 // SCALE
                    sx2, sy2 = max(sx1 + 1, x2 // SCALE), max(sy1 + 1, y2 // SCALE)
                    roi_temps = temp_celsius[sy1:sy2, sx1:sx2]
                    roi_stats = (x1, y1, float(np.min(roi_temps)),
                                 float(np.mean(roi_temps)), float(np.max(roi_temps)))
                    cv2.rectangle(colormap, (x1, y1), (x2, y2), (0, 255, 255), 2)
            if alarm_enabled and max_temp >= alarm_threshold:
                cv2.rectangle(colormap, (0, 0), (WIDTH - 1, HEIGHT - 1), (0, 0, 255), 4)

            # FPS: only new frames are timestamped above; (n-1)/elapsed is the unbiased rate
            fps = compute_fps(fps_times)
            fps_text = f"{fps:.1f} FPS"

            if line_profile_mode and line_profile['start'] and line_profile['end']:
                if line_profile.get('dragging'):
                    cv2.line(colormap, line_profile['start'], line_profile['end'], (0, 255, 255), 2, cv2.LINE_AA)
                else:
                    draw_line_profile_markers(colormap, line_profile['start'], line_profile['end'])

            display_img = apply_rotate_flip(colormap)
            dh, dw = display_img.shape[:2]

            for p in panels.values():
                p.clamp(dw, dh)

            if line_profile_mode and line_profile['start'] and line_profile['end'] and not line_profile.get('dragging'):
                draw_line_profile_graph(display_img, temp_celsius, line_profile['start'], line_profile['end'], SCALE, use_fahrenheit, panels['line_profile'])

            if trend_enabled:
                draw_trend_graph(display_img, trend_data, use_fahrenheit, panels['trend'])

            if histogram_enabled:
                draw_histogram(display_img, temp_celsius, use_fahrenheit, panels['histogram'])

            # --- All text/overlay drawn on display_img in SCREEN coords (always readable, fixed positions) ---

            # Cursor label follows mouse in screen space
            cmx, cmy = logical_to_display(mouse_x, mouse_y)
            cmx, cmy = max(0, min(cmx, dw - 1)), max(0, min(cmy, dh - 1))
            draw_measure_label(display_img, cursor_label, (cmx, cmy), (0, 255, 0), True, dw, dh, scale=0.55)

            if show_min_max:
                draw_measure_label(display_img, f"MAX {format_temp(max_temp, use_fahrenheit)}",
                                   logical_to_display(max_x, max_y), (120, 150, 255), True, dw, dh,
                                   clear=PIN_SIZE)
                draw_measure_label(display_img, f"MIN {format_temp(min_temp, use_fahrenheit)}",
                                   logical_to_display(min_x, min_y), (255, 210, 130), False, dw, dh,
                                   clear=PIN_SIZE)

            # ROI label follows ROI in screen space (stats computed once, above)
            if roi_stats is not None:
                rx, ry, roi_min, roi_avg, roi_max = roi_stats
                dr = logical_to_display(rx, ry)
                draw_measure_label(display_img,
                                   f"ROI: {format_temp(roi_min, use_fahrenheit)} / {format_temp(roi_avg, use_fahrenheit)} / {format_temp(roi_max, use_fahrenheit)}",
                                   dr, (0, 255, 255), True, dw, dh, scale=0.45, pad=6)

            # --- Fixed HUD: always at screen edges, using dw/dh ---
            if delta_display is not None:
                d_min, d_max = float(np.min(delta_display)), float(np.max(delta_display))
                cv2.putText(display_img, f"DELTA-T: {d_min:+.1f} to {d_max:+.1f} C  (V to reset)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

            if alarm_enabled and max_temp >= alarm_threshold:
                cv2.putText(display_img, f" HOT! {format_temp(max_temp, use_fahrenheit)} ", (dw // 2 - 80, dh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

            if show_controls_overlay:
                overlay = display_img.copy()
                cv2.rectangle(overlay, (10, 10), (dw - 10, dh - 10), (30, 30, 30), -1)
                cv2.addWeighted(overlay, 0.88, display_img, 0.12, 0, display_img)
                line_h, y0 = 24, 35
                font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.5
                cv2.putText(display_img, "--- CONTROLS (I to close) ---", (20, y0), font, 0.55, (200, 255, 200), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, "Q - Quit   P - Next palette   M - Min/Max markers   D - Debug", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, "SPACE - Freeze frame   S - Snapshot   F - Celsius/Fahrenheit", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, "1-9 !@#$%^& - Palette 1-16   P - Next   Z - Invert   0 - Temp range: Auto / Room / Wide", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, "R - ROI: draw rectangle (min/avg/max)   A - High temp alarm", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, "H - High quality: DDE + Sharpening", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, "[ ] - DDE Contrast   } - Mode: gamma/percentile/CLAHE   - = - Sharpness", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, "C - FPN correction (running calibration)", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, "X - One-frame FPN: capture current frame, apply to all; X again = reset", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, "G - De-noise (EGLM)  < > (or , .) - denoise strength", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, "Y - IBP upscale   J / K - IBP strength (1-10)", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, "E - Burst SR (multi-frame from subpixel shifts, Samsung-style)", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, "L - Line profile   T - Isotherms   V - Delta-T mode", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, "W - Temp trend   B - Histogram   N - Anomaly detection", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, "Left/Right - Rotate 90 deg   Up/Down - Flip V/H", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, "U - Virtual webcam (OBS) for Zoom/Telegram/Teams", (20, y0), font, fs, (220, 220, 220), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, "O - Toggle RGB camera blend   / - Reset gradient alignment", (20, y0), font, fs, (255, 200, 100), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, "; - Blend mode   ` / ~ - alpha +/-   ' \\ - hot thresh   [ ] - RGB zoom (when O on)", (20, y0), font, fs, (255, 200, 100), 1, cv2.LINE_AA); y0 += line_h
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
                cv2.putText(display_img, f"frame_size (bytes): {FRAME_SIZE}", (15, y0), font, fs, (200, 255, 200), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, f"SCALE: {SCALE}  WIDTH: {WIDTH}  HEIGHT: {HEIGHT}", (15, y0), font, fs, (200, 255, 200), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, f"palette_idx: {palette_idx}  palette: {palette_name}", (15, y0), font, fs, (200, 255, 200), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, f"show_min_max: {show_min_max}", (15, y0), font, fs, (200, 255, 200), 1, cv2.LINE_AA); y0 += line_h
                _rr = reader.running if reader is not None else False
                cv2.putText(display_img, f"reader.running: {_rr}  reconnecting: {reconnecting}", (15, y0), font, fs, (200, 255, 200), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, f"raw temp (before clip) min: {raw_min_c:.2f} C  max: {raw_max_c:.2f} C", (15, y0), font, fs, (255, 255, 200), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, f"display temp min: {min_temp:.2f} C  max: {max_temp:.2f} C", (15, y0), font, fs, (255, 255, 200), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, f"min_pt: ({min_x}, {min_y})  max_pt: ({max_x}, {max_y})", (15, y0), font, fs, (255, 255, 200), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, f"cursor: ({mouse_x}, {mouse_y})  cursor_temp: {cursor_temp:.2f} C", (15, y0), font, fs, (255, 255, 200), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, f"bottom_half shape: {bottom_half.shape}  dtype: {bottom_half.dtype}", (15, y0), font, fs, (200, 200, 255), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, f"temp_celsius shape: {temp_celsius.shape}", (15, y0), font, fs, (200, 200, 255), 1, cv2.LINE_AA); y0 += line_h
                cv2.putText(display_img, f"high_quality_mode: {high_quality_mode} (DDE Sharpness)", (15, y0), font, fs, (200, 255, 200), 1, cv2.LINE_AA); y0 += line_h
                rgb_dbg = (f"RGB cam: {'ON' if rgb_cam_enabled else 'OFF'}"
                           f"  tracking: {rgb_aligner.is_tracking}"
                           f"  mode: {rgb_blend_mode}"
                           f"  alpha: {rgb_alpha:.2f}"
                           f"  hot_min: {rgb_hot_min:.0f}C  {rgb_aligner.status_text}")
                cv2.putText(display_img, rgb_dbg, (15, y0), font, fs, (255, 200, 100), 1, cv2.LINE_AA)

                # Gradient debug: show what we match (thermal vs RGB Sobel magnitude)
                if rgb_cam_enabled and hasattr(rgb_aligner, 'get_debug_gradients'):
                    g_t, g_r = rgb_aligner.get_debug_gradients()
                    if g_t is not None and g_r is not None:
                        box_w, box_h = 200, 150
                        pad = 8
                        x0 = dw - 2 * box_w - 2 * pad - 10
                        y0 = 55
                        # Thermal gradient (left)
                        gt_bgr = cv2.cvtColor(g_t, cv2.COLOR_GRAY2BGR)
                        gt_show = cv2.resize(gt_bgr, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
                        x1, y1 = x0, y0
                        display_img[y1:y1 + box_h, x1:x1 + box_w] = gt_show
                        cv2.rectangle(display_img, (x1, y1), (x1 + box_w, y1 + box_h), (0, 255, 0), 1)
                        cv2.putText(display_img, "Thermal grad", (x1, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                        # RGB gradient (right)
                        gr_bgr = cv2.cvtColor(g_r, cv2.COLOR_GRAY2BGR)
                        gr_show = cv2.resize(gr_bgr, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
                        x2 = x0 + box_w + pad
                        display_img[y1:y1 + box_h, x2:x2 + box_w] = gr_show
                        cv2.rectangle(display_img, (x2, y1), (x2 + box_w, y1 + box_h), (255, 200, 0), 1)
                        cv2.putText(display_img, "RGB grad", (x2, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)
                        cv2.putText(display_img, "ECC matches these (Sobel magnitude)", (x0, y1 + box_h + 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

            # RGB camera HUD (bottom-left, when active)
            if rgb_cam_enabled:
                if not rgb_aligner.is_tracking:
                    rgb_hud = f"RGB | {rgb_aligner.status_text}  (/ to reset)"
                    hud_color = (0, 180, 255)
                elif rgb_blend_mode == 'full':
                    rgb_hud = f"RGB | FULL a:{rgb_alpha:.2f}  {rgb_aligner.status_text}"
                    hud_color = (100, 255, 100)
                else:
                    rgb_hud = f"RGB | HOT >={rgb_hot_min:.0f}C  {rgb_aligner.status_text}"
                    hud_color = (100, 200, 255)
                cv2.putText(display_img, rgb_hud, (10, dh - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display_img, rgb_hud, (10, dh - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1, cv2.LINE_AA)

            if virtual_cam_enabled:
                if _virtual_cam[0] is None:
                    try:
                        import pyvirtualcam
                        _virtual_cam[0] = pyvirtualcam.Camera(_VCAM_SIZE[0], _VCAM_SIZE[1], fps=25)
                        print("Virtual cam: opened device =", getattr(_virtual_cam[0], 'device', '?'), "| size 640x480 @ 25 fps")
                        _vcam_send_count[0] = 0
                    except Exception as e:
                        print("Virtual cam: failed.", e)
                        err_lower = str(e).lower()
                        if "not installed" in err_lower or "obs" in err_lower:
                            print("  -> macOS 13+ and OBS 30+: pyvirtualcam 0.14+ required (needs Python 3.9+).")
                            print("     Use Python 3.9+ then: pip install -U pyvirtualcam")
                            print("  -> Or: Recovery mode -> Terminal -> system-override legacy-camera-plugins-without-sw-camera-indication=on, reboot; use OBS 29.")
                            print("  -> Then: OBS -> Start Virtual Camera -> Stop -> close OBS; press U again.")
                        else:
                            print("  -> macOS: install OBS, start Virtual Camera once, then press U again.")
                        virtual_cam_enabled = False
                if _virtual_cam[0] is not None:
                    if not _send_to_virtual_cam(display_img):
                        virtual_cam_enabled = False

            _tb_display_dims[0], _tb_display_dims[1] = dw, dh
            cont_label = "Off" if contrast_mode == 'off' or dde_clip_limit <= 0 else ("CLAHE" if contrast_mode == 'clahe' else contrast_mode.capitalize())
            status_state = {
                'palette_name': palette_name, 'high_quality_mode': high_quality_mode,
                'cont_label': cont_label, 'dde_clip_limit': dde_clip_limit, 'dde_sharp_strength': dde_sharp_strength,
                'single_frame_fpn_enabled': single_frame_fpn_enabled, 'fpn_enabled': fpn_enabled,
                'denoise_enabled': denoise_enabled, 'denoiser_strength': denoiser.strength if denoise_enabled else 0,
                'ibp_upscale_enabled': ibp_upscale_enabled, 'ibp_strength': ibp_strength,
                'burst_sr_enabled': burst_sr_enabled, 'burst_frame_count': burst_sr.frame_count if burst_sr_enabled else 0,
                'delta_t_enabled': delta_t_enabled, 'isotherm_enabled': isotherm_enabled, 'anomaly_enabled': anomaly_enabled,
                'fps_text': fps_text,
            }
            tb_states = {
                'minmax': show_min_max, 'freeze': frozen, 'fahr': use_fahrenheit,
                'invert': palette_invert, 'roi': roi_state['active'], 'line': line_profile_mode,
                'iso': isotherm_enabled, 'hist': histogram_enabled,
                'trend': trend_enabled, 'anomaly': anomaly_enabled,
                'delta': delta_t_enabled, 'alarm': alarm_enabled,
                'dde': high_quality_mode, 'fpn': fpn_enabled,
                'fpn1': single_frame_fpn_enabled, 'dnz': denoise_enabled,
                'ibp': ibp_upscale_enabled, 'burst': burst_sr_enabled,
                'flip_v': flip_v, 'flip_h': flip_h, 'vcam': virtual_cam_enabled,
                'rgb': rgb_cam_enabled,
                'debug': debug_mode, 'help': show_controls_overlay,
            }
            total_h = dh + STATUS_BAR_H + TOOLBAR_H
            if _ui_canvas is None or _ui_canvas.shape[0] != total_h or _ui_canvas.shape[1] != dw:
                _ui_canvas = np.empty((total_h, dw, 3), dtype=np.uint8)
            _ui_canvas[:dh] = display_img
            _ui_canvas[dh:dh + STATUS_BAR_H] = 0
            _ui_canvas[dh + STATUS_BAR_H:] = 0
            draw_status_bar(_ui_canvas, dh, dw, status_state)
            draw_toolbar(_ui_canvas, dh + STATUS_BAR_H, dw, tb_states, _tb_hover_id[0])
            last_display_img = _ui_canvas
            last_image_h = dh
            _frozen_cache['sig'] = None   # a fresh live frame invalidates the frozen canvas
            cv2.imshow(window_name, _ui_canvas)

            poll_keys(1)
            if window_closed():
                break
            if not apply_live_keys():
                break

        except KeyboardInterrupt:
            break
        except Exception as e:
            # A single bad frame (driver hiccup, virtual-cam glitch, odd geometry)
            # must not kill the session and lose the unsaved settings. Bail out only
            # if errors keep arriving faster than once a second, i.e. we are spinning.
            now_err = time.monotonic()
            loop_errors = loop_errors + 1 if now_err - last_loop_error < 1.0 else 1
            last_loop_error = now_err
            print(f"Frame error ({loop_errors}/{MAX_LOOP_ERRORS}): {type(e).__name__}: {e}")
            if loop_errors >= MAX_LOOP_ERRORS:
                traceback.print_exc()
                print("Too many consecutive frame errors - exiting.")
                break
            poll_keys(50)
            if window_closed():
                break

    # --- Shutdown: settings first, they are what the user would miss ---
    save_app_settings({
        'palette_idx': palette_idx,
        'palette_invert': palette_invert,
        'contrast_mode': contrast_mode,
        'dde_clip_limit': dde_clip_limit,
        'dde_sharp_strength': dde_sharp_strength,
        'dde_sharp_sigma': dde_sharp_sigma,
        'high_quality_mode': high_quality_mode,
        'use_fahrenheit': use_fahrenheit,
        'range_preset': range_preset,
        'rotation_deg': rotation_deg,
        'flip_h': flip_h,
        'flip_v': flip_v,
        'denoise_strength': denoiser.strength,
        'ibp_strength': ibp_strength,
        'alarm_threshold': alarm_threshold,
        'rgb_blend_mode': rgb_blend_mode,
        'rgb_alpha': rgb_alpha,
        'rgb_hot_min': rgb_hot_min,
        'rgb_zoom': rgb_aligner.get_zoom(),
    })
    save_panel_layout(panels)

    if _rgb_cam[0] is not None:
        _rgb_cam[0].stop()
        _rgb_cam[0] = None
    if _virtual_cam[0] is not None:
        try:
            _virtual_cam[0].close()
        except Exception:
            pass
        _virtual_cam[0] = None

    stop_thermal_capture(process, reader)
    # An open attempt may still be in flight; do not leak the ffmpeg it spawns.
    if capture_opener is not None:
        capture_opener.abandon()

    cv2.destroyAllWindows()
    if not _tray_stopped[0]:
        _tray_stopped[0] = True
        try:
            tray_icon.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()
