"""
Detect the Infiray P2 Pro on the USB bus even when AVFoundation does not
publish it as a camera.

Typical macOS failure: the device enumerates as Generic "USB Camera"
(VID 0x0BDA / PID 0x5830) but AppleUSBHostCompositeDevice stays unmatched
(no UVC interfaces). Chrome/Brave WebUSB user-clients on the device are a
common cause — they open the USB device before the UVC stack claims it.
"""

from __future__ import annotations

import re
import subprocess
from typing import Optional


P2PRO_VID = 3034      # 0x0BDA Realtek
P2PRO_PID = 22576     # 0x5830 Infiray P2 Pro in UVC mode


def probe_p2pro_usb(timeout=2.0) -> dict:
    """
    Snapshot of the P2 Pro on the IOKit USB plane.

    Keys:
      present (bool), product, vendor, holders (list[str]),
      has_interfaces (bool), composite_unmatched (bool)
    """
    empty = {
        'present': False,
        'product': None,
        'vendor': None,
        'holders': [],
        'has_interfaces': False,
        'composite_unmatched': False,
    }
    try:
        out = subprocess.check_output(
            ['ioreg', '-l', '-w', '0'],
            text=True, errors='replace', timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError, subprocess.CalledProcessError):
        return empty

    lines = out.splitlines()
    idx = _find_device_line(lines)
    if idx is None:
        return empty

    pos = lines[idx].find('+-o')
    block = [lines[idx]]
    for line in lines[idx + 1:]:
        p = line.find('+-o')
        if p != -1 and (pos == -1 or p <= pos):
            break
        block.append(line)
    text = '\n'.join(block)

    holders = []
    for m in re.finditer(r'"IOUserClientCreator"\s*=\s*"pid \d+, ([^"]+)"', text):
        name = m.group(1).strip()
        if name and name not in holders:
            holders.append(name)

    return {
        'present': True,
        'product': _ioreg_str(text, 'USB Product Name') or 'USB Camera',
        'vendor': _ioreg_str(text, 'USB Vendor Name') or 'Generic',
        'holders': holders,
        'has_interfaces': 'IOUSBHostInterface' in text,
        'composite_unmatched': bool(
            re.search(r'AppleUSBHostCompositeDevice.*!registered, !matched', text)
        ),
    }


def format_unavailable_hint(info: Optional[dict] = None) -> str:
    """One or two short lines for the reconnect overlay / log."""
    if info is None:
        info = probe_p2pro_usb()
    if not info.get('present'):
        return "P2 Pro not on USB — plug it in"
    if info.get('has_interfaces'):
        return "P2 Pro is on USB; waiting for macOS camera list"
    holders = info.get('holders') or []
    if holders:
        who = ', '.join(holders[:3])
        return f"On USB, not a Mac camera. Close {who} (WebUSB holds it)"
    if info.get('composite_unmatched'):
        return "On USB, UVC did not bind — unplug 5 s, then plug back in"
    return "On USB, but macOS did not publish it as a camera"


def _find_device_line(lines):
    vid_pat = re.compile(r'"idVendor"\s*=\s*%d\b' % P2PRO_VID)
    pid_pat = re.compile(r'"idProduct"\s*=\s*%d\b' % P2PRO_PID)
    # Prefer the host-device node, not a random property dump.
    for i, line in enumerate(lines):
        if 'USB Camera@' in line and 'IOUSBHostDevice' in line:
            return i
    # Fallback: a node whose following properties are the P2 Pro VID/PID.
    for i, line in enumerate(lines):
        if '+-o ' not in line:
            continue
        window = '\n'.join(lines[i:i + 40])
        if vid_pat.search(window) and pid_pat.search(window):
            return i
    return None


def _ioreg_str(text: str, key: str):
    m = re.search(r'"%s"\s*=\s*"([^"]*)"' % re.escape(key), text)
    return m.group(1) if m else None
