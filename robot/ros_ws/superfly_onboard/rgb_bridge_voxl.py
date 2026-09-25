#!/usr/bin/env python3
"""VOXL2-side RGB bridge: hires camera -> 224x224 uint8 RGB -> SFRG/UDP.

This is the drone half of the ground-control deployment described in
`docs/3-deploy-ground-control.md` (§3 the RGB bridge, §4 camera geometry).
It runs on the VOXL2's Ubuntu userspace under plain `python3`, with **no ROS
and no `superfly` package installed** -- exactly the shape of Daniel's ToF
path (`tof_udp_stream.cpp`, DIFFAERO.md §10.2), which is a standalone
libmodal-pipe client, not a ROS node.

    voxl-camera-server                 (MPA pipe /run/mpa/hires_front/)
        |  camera_image_metadata_t + NV12/RAW8/RGB payload
        v
    rgb_bridge_voxl.py
        |  decimate 30 -> 15 Hz
        |  [fisheye only] undistort to a 91 deg pinhole view, 4:3
        |  anisotropic resize 4:3 -> 224x224, cv2.INTER_LINEAR, RGB, row 0 = up
        |  zlib level 3, fragment into 60000-byte datagrams
        v
    UDP :15003  ->  superfly.common.rgb_transport.RGBSubscriber (ground)

Because the ground side already speaks the `SFRG` wire format, this script
**vendors** the publisher rather than inventing a second format; see the
"vendored wire format" section below.

Self-contained on purpose: the only hard dependency is numpy. OpenCV (`cv2`)
is used when importable -- for the fisheye undistortion (required) and for the
resize (preferred; a numpy bilinear fallback exists and is within ~1 LSB, but
the sim published frames through `cv2.INTER_LINEAR`, so cv2 is what reproduces
training exactly). See `docs/rgb_bridge_voxl.md` for install and systemd.

PYTHON 3.6 IS THE TARGET, NOT 3.10
----------------------------------
The ModalAI voxl-suite userspace is Ubuntu 18.04: `python3` there is **3.6.9**
with **numpy 1.13**, and this file must run under it unchanged. So, deliberately
and permanently:

  * no `from __future__ import annotations`, no variable/function annotations,
    no `X | Y` unions, no builtin generics (`list[int]`), no `dataclasses`,
    no walrus `:=`, no f-string `=` specifier -- every one of those is 3.7 or
    later;
  * no `time.monotonic_ns` (3.7+): `_monotonic_ns()` below is the shim, on the
    same CLOCK_MONOTONIC and with far more precision than a camera frame
    interval needs;
  * numpy stays on the 1.13 surface -- `np.frombuffer`, `np.repeat`, `np.clip`,
    `np.resize`, fancy indexing, integer shifts. Nothing newer (`np.divmod`,
    `np.stack`, the 1.17+ random Generator, NEP-50 casting) appears.

f-strings themselves ARE 3.6 and are used freely. `python3 -m compileall` on the
board is the real check; on a dev box `ast.parse(src, feature_version=(3, 6))`
catches every syntax-level regression and `tests/test_rgb_bridge_voxl.py`
catches the behavioural ones.

Examples
--------
    # on the drone, real camera, 87 deg rectilinear lens, ground PC at .50
    ./rgb_bridge_voxl.py --source mpa --pipe hires_front \
        --lens rectilinear87 --fps 15 --host 192.168.1.50 --port 15003 --stats

    # bench, no drone: replay a sim video into a local subscriber
    ./rgb_bridge_voxl.py --source file --input rgb.mp4 --lens none \
        --host 127.0.0.1 --port 15003 --fps 15 --stats
"""

import argparse
import errno
import os
import random
import re
import select
import signal
import socket
import struct
import sys
import time
import zlib

import numpy as np

try:                                    # optional; see module docstring
    import cv2                          # noqa: F401
except Exception:                       # pragma: no cover - depends on host
    cv2 = None


# ---------------------------------------------------------------------------
# 1. Vendored wire format
# ---------------------------------------------------------------------------
# CANONICAL SOURCE: src/superfly/common/rgb_transport.py (class RGBPublisher).
# This file is a byte-compatible copy so the bridge can run on a VOXL2 that
# has no `superfly` package installed. Any change to the canonical module's
# MAGIC / VERSION / _HEADER / compression MUST be mirrored here, and
# tests/test_rgb_bridge_voxl.py fails the moment the two disagree (it decodes
# this publisher's datagrams with the real RGBSubscriber).
#
# Wire format (little-endian), one header per fragment:
#     "SFRG" | uint8 version | 3 pad | uint32 seq | uint32 h | uint32 w
#           | uint32 part | uint32 n_parts | zlib(uint8 HxWx3) chunk

MAGIC = b"SFRG"
VERSION = 1
_HEADER = struct.Struct("<4sB3xIIIII")
DEFAULT_CHUNK = 60000                   # rgb_transport._CHUNK
DEFAULT_ZLIB_LEVEL = 3                  # rgb_transport zlib.compress(..., 3)
DEFAULT_RGB_PORT = 15003                # rgb_transport.RGB_PORT at slot 0


class RGBPublisher:
    """Send uint8 RGB frames losslessly, fragmenting above the UDP limit.

    Same wire bytes as `superfly.common.rgb_transport.RGBPublisher`. Two knobs
    the canonical class hard-codes are flags here, because doc 3 §3 asks for
    them over Wi-Fi: `chunk` (60000 B is 41 IP fragments at a 1500 B MTU --
    ~1400 keeps the kernel's reassembly buffers out of the picture) and
    `level` (drop to 1 if zlib is the bottleneck on the Kryo core).

    send() never raises: losing one RGB frame must never kill the bridge.
    """

    def __init__(self, host="127.0.0.1", port=DEFAULT_RGB_PORT,
                 chunk=DEFAULT_CHUNK, level=DEFAULT_ZLIB_LEVEL):
        self._addr = (host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._chunk = int(chunk)
        self._level = int(level)
        self._seq = 0
        self.send_errors = 0
        self.frames_sent = 0
        self.bytes_sent = 0

    def send(self, rgb):
        frame = np.asarray(rgb)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"RGB frame must be HxWx3, got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        h, w, _ = frame.shape
        body = zlib.compress(frame.tobytes(), self._level)
        n_parts = max(1, -(-len(body) // self._chunk))
        try:
            for part in range(n_parts):
                chunk = body[part * self._chunk:(part + 1) * self._chunk]
                packet = _HEADER.pack(
                    MAGIC, VERSION, self._seq, h, w, part, n_parts
                ) + chunk
                self._sock.sendto(packet, self._addr)
            self.frames_sent += 1
            self.bytes_sent += len(body)
        except OSError as exc:
            self.send_errors += 1
            if self.send_errors < 3 or self.send_errors % 100 == 0:
                _log(f"send failed ({exc}); {self.send_errors} frames dropped so far")
        self._seq += 1

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 2. Modal Pipe Architecture (MPA) camera client
# ---------------------------------------------------------------------------
# No Python binding ships with libmodal-pipe (voxl-suite installs the C
# library `/usr/lib64/libmodal_pipe.so` and the C headers only), so the raw
# pipe protocol is implemented here. It is small and stable; verified against
# libmodal-pipe master, library/src/client.c:
#
#   * the server publishes in a directory `/run/mpa/<name>/` holding a FIFO
#     named `request` (client.c:920-922 builds `req_path = dir + "request"`);
#   * a client picks a unique name `<client_name><8 random digits>`, writes it
#     NUL-terminated to `request` (client.c:433-441), and the server then
#     creates a FIFO `/run/mpa/<name>/<that name>` which the client opens
#     read-only (client.c:404-407, 462-468);
#   * the stream is then, forever: one packed `camera_image_metadata_t`
#     (40 bytes) followed by exactly `size_bytes` of image
#     (client.c:691-693 reads sizeof(meta), _check_cam_meta gives size_bytes).
#
# camera_image_metadata_t, quoted verbatim from libmodal-pipe master,
# library/include/pipe_interfaces/camera_image_metadata_t.h:
#
#     typedef struct camera_image_metadata_t
#     {
#         uint32_t magic_number; ///< set to CAMERA_MAGIC_NUMBER
#         int64_t timestamp_ns;  ///< timestamp in apps-proc clock-monotonic of beginning of exposure
#         int32_t frame_id;      ///< iterator from 0++ starting from first frame when server starts on boot
#         int16_t width;         ///< image width in pixels
#         int16_t height;        ///< image height in bytes
#         int32_t size_bytes;    ///< size of the image, for stereo this is the size of both L&R together
#         int32_t stride;        ///< bytes per row
#         int32_t exposure_ns;   ///< exposure in nanoseconds
#         int16_t gain;          ///< ISO gain (100, 200, 400, etc..)
#         int16_t format;        ///< raw8, nv12, etc
#         int16_t framerate;     ///< expected framerate hz
#         int16_t reserved;      ///< extra reserved bytes
#     } __attribute__((packed)) camera_image_metadata_t;
#
# `__attribute__((packed))` => no padding => struct "<Iqihhiiihhhh", 40 bytes.
# NOTE the magic number is 0x564F584C ("VOXL"), from
# library/include/pipe_interfaces/magic_number.h:
#     #define CAMERA_MAGIC_NUMBER         (0x564F584C)

MPA_BASE_DIR = "/run/mpa/"
CAMERA_MAGIC_NUMBER = 0x564F584C
_CAM_META = struct.Struct("<Iqihhiiihhhh")
assert _CAM_META.size == 40, "camera_image_metadata_t must be 40 packed bytes"

_META_FIELDS = ("magic_number", "timestamp_ns", "frame_id", "width", "height",
                "size_bytes", "stride", "exposure_ns", "gain", "format",
                "framerate", "reserved")

# from camera_image_metadata_t.h (only the ones this bridge decodes)
IMAGE_FORMAT_RAW8 = 0
IMAGE_FORMAT_NV12 = 1
IMAGE_FORMAT_NV21 = 6
IMAGE_FORMAT_YUV420 = 9
IMAGE_FORMAT_RGB = 10

FORMAT_NAMES = {
    IMAGE_FORMAT_RAW8: "RAW8", IMAGE_FORMAT_NV12: "NV12",
    IMAGE_FORMAT_NV21: "NV21", IMAGE_FORMAT_YUV420: "YUV420",
    IMAGE_FORMAT_RGB: "RGB",
}


def _log(msg):
    print(f"[rgb_bridge] {msg}", file=sys.stderr, flush=True)


def _monotonic_ns():
    """`time.monotonic_ns` is 3.7+ and the board is 3.6.9 (see the module
    docstring). Same clock either way, and a float64 second count still
    resolves far below a nanosecond at any plausible uptime, so nothing
    measurable is lost against the camera's own `timestamp_ns`."""
    ns = getattr(time, "monotonic_ns", None)
    if ns is not None:
        return ns()
    return int(time.monotonic() * 1e9)


def parse_meta(raw):
    """Unpack a packed camera_image_metadata_t into a dict."""
    return dict(zip(_META_FIELDS, _CAM_META.unpack(raw)))


class MPASource:
    """libmodal-pipe camera client, raw protocol, auto-reconnecting.

    `read()` returns `(meta, payload_bytes)` or None if nothing arrived within
    the timeout. It never raises on a camera-server restart: the pipe is
    re-requested on the next call (doc 3 §3 robustness).
    """

    name = "mpa"

    def __init__(self, pipe, client_name="rgb_bridge", reconnect_s=0.5,
                 max_frame_bytes=64 << 20):
        # accept either a bare pipe name ("hires_front") or a full path
        pipe = str(pipe)
        self.pipe_dir = pipe if pipe.startswith("/") else MPA_BASE_DIR + pipe
        if not self.pipe_dir.endswith("/"):
            self.pipe_dir += "/"
        self.req_path = self.pipe_dir + "request"
        self.client_name = client_name
        self.reconnect_s = float(reconnect_s)
        self.max_frame_bytes = int(max_frame_bytes)
        self._fd = None
        self._data_path = None
        self._buf = bytearray()
        self._next_try = 0.0
        self.reconnects = 0

    # -- connection ---------------------------------------------------------
    @property
    def connected(self):
        return self._fd is not None

    def connect(self):
        """One connection attempt. Returns True on success."""
        now = time.monotonic()
        if now < self._next_try:
            return False
        self._next_try = now + self.reconnect_s
        if not os.path.exists(self.req_path):
            return False                       # server not running yet
        newname = "%s%08d" % (self.client_name[:24], random.randint(0, 99999999))
        data_path = self.pipe_dir + newname
        if os.path.exists(data_path):
            return False
        try:
            req_fd = os.open(self.req_path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno != errno.ENXIO:       # ENXIO = server left stale pipes
                _log(f"cannot open request pipe {self.req_path}: {exc}")
            return False
        try:
            os.write(req_fd, newname.encode() + b"\x00")
        except OSError as exc:
            _log(f"request write failed: {exc}")
            os.close(req_fd)
            return False
        os.close(req_fd)
        # the server creates our FIFO in response; wait up to 1 s like client.c
        for _ in range(500):
            if os.path.exists(data_path):
                break
            time.sleep(0.002)
        try:
            fd = os.open(data_path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            _log(f"cannot open data pipe {data_path}: {exc}")
            return False
        self._fd = fd
        self._data_path = data_path
        self._buf = bytearray()
        _log(f"connected to {self.pipe_dir} as {newname}")
        return True

    def disconnect(self, why=""):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        if self._data_path:
            try:
                os.remove(self._data_path)     # client.c:1426 does the same
            except OSError:
                pass
        if self._fd is not None:
            self.reconnects += 1
            _log(f"disconnected{(': ' + why) if why else ''}; will retry")
        self._fd = None
        self._data_path = None
        self._buf = bytearray()

    close = disconnect

    # -- reading ------------------------------------------------------------
    def _fill(self, timeout):
        """Wait for and append one chunk of pipe data. False = disconnected."""
        try:
            r, _, _ = select.select([self._fd], [], [], timeout)
        except (OSError, ValueError):
            self.disconnect("select failed")
            return False
        if not r:
            return True                        # just idle
        try:
            data = os.read(self._fd, 1 << 20)
        except BlockingIOError:
            return True
        except OSError as exc:
            self.disconnect(f"read failed ({exc})")
            return False
        if not data:                           # writer closed => server gone
            self.disconnect("pipe closed by server")
            return False
        self._buf += data
        return True

    def _resync(self):
        """Scan forward to the next CAMERA_MAGIC_NUMBER after a pipe overflow."""
        magic = struct.pack("<I", CAMERA_MAGIC_NUMBER)
        idx = self._buf.find(magic, 1)
        if idx < 0:
            keep = 3                           # a magic may straddle the edge
            del self._buf[:max(0, len(self._buf) - keep)]
            return False
        _log(f"resynced, dropped {idx} bytes (pipe overflow / fell behind)")
        del self._buf[:idx]
        return True

    def read(self, timeout=0.2):
        """Return (meta, payload) or None. Never raises."""
        if self._fd is None:
            self.connect()
            if self._fd is None:
                time.sleep(min(timeout, self.reconnect_s))
                return None
        deadline = time.monotonic() + timeout
        while True:
            if len(self._buf) >= _CAM_META.size:
                meta = parse_meta(bytes(self._buf[:_CAM_META.size]))
                if meta["magic_number"] != CAMERA_MAGIC_NUMBER:
                    if not self._resync():
                        if time.monotonic() >= deadline:
                            return None
                        if not self._fill(max(0.0, deadline - time.monotonic())):
                            return None
                    continue
                n = meta["size_bytes"]
                if n <= 0 or n > self.max_frame_bytes:
                    self.disconnect(f"implausible size_bytes={n}")
                    return None
                need = _CAM_META.size + n
                if len(self._buf) >= need:
                    payload = bytes(self._buf[_CAM_META.size:need])
                    del self._buf[:need]
                    return meta, payload
            if time.monotonic() >= deadline:
                return None
            if not self._fill(max(0.0, deadline - time.monotonic())):
                return None


class FileSource:
    """Bench source: a video file or an OpenCV device index (v4l2).

    Feeds the identical downstream pipeline on any machine, so stage 0 of
    doc 3 §5 can be rehearsed without a drone. Frames come out as RGB uint8
    with a synthetic `camera_image_metadata_t`-shaped meta, paced at
    `source_fps` to imitate the 30 fps hires stream.
    """

    name = "file"

    def __init__(self, spec, source_fps=None, loop=False):
        if cv2 is None:
            raise RuntimeError("--source file/v4l2 needs OpenCV (cv2)")
        self.spec = int(spec) if str(spec).isdigit() else str(spec)
        self.loop = bool(loop)
        self._cap = cv2.VideoCapture(self.spec)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video source {self.spec!r}")
        fps = source_fps
        if fps is None:
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
            if not (1.0 <= fps <= 240.0):
                fps = 30.0
        self.source_fps = float(fps)
        self._period = 1.0 / self.source_fps
        self._next = None
        self._frame_id = 0
        self.reconnects = 0
        self.ended = False                 # EOF on a non-looping file

    @property
    def connected(self):
        return self._cap is not None and self._cap.isOpened()

    def read(self, timeout=0.2):
        now = time.monotonic()
        if self._next is None:
            self._next = now
        if now < self._next:                   # pace like a real camera
            time.sleep(min(timeout, self._next - now))
            return None
        ok, bgr = self._cap.read()
        if not ok:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, bgr = self._cap.read()
            if not ok:
                self.ended = True          # main() exits rather than spinning
                return None
        self._next += self._period
        if self._next < time.monotonic():
            self._next = time.monotonic() + self._period
        rgb = bgr[:, :, ::-1]
        h, w = rgb.shape[:2]
        meta = dict(magic_number=CAMERA_MAGIC_NUMBER,
                    timestamp_ns=_monotonic_ns(), frame_id=self._frame_id,
                    width=w, height=h, size_bytes=int(rgb.size),
                    stride=w * 3, exposure_ns=0, gain=0,
                    format=IMAGE_FORMAT_RGB, framerate=int(self.source_fps),
                    reserved=0)
        self._frame_id += 1
        return meta, np.ascontiguousarray(rgb)

    def disconnect(self, why=""):
        pass

    def close(self, why=""):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ---------------------------------------------------------------------------
# 3. Colour conversion (MPA payload -> RGB uint8, row 0 = up)
# ---------------------------------------------------------------------------
# Fixed-point ITU-R BT.601 "video range" coefficients, identical to OpenCV's
# YUV420->RGB path (modules/imgproc/src/color_yuv.simd.hpp), so a frame
# converted here is bit-identical to cv2.cvtColor(..., COLOR_YUV2RGB_NV12).
# That matters: the ground-side encoder is contrastively trained on exact
# uint8, so "close enough" colour is a silent domain shift.
_ITUR_SHIFT = 20
_ITUR_CY = 1220542
_ITUR_CUB = 2116026
_ITUR_CUG = -409993
_ITUR_CVG = -852492
_ITUR_CVR = 1673527


def yuv_to_rgb(y, u, v):
    """(h,w) uint8 luma + full-size chroma -> (h,w,3) uint8 RGB."""
    yy = np.maximum(y.astype(np.int32) - 16, 0) * _ITUR_CY
    ui = u.astype(np.int32) - 128
    vi = v.astype(np.int32) - 128
    half = 1 << (_ITUR_SHIFT - 1)
    out = np.empty(y.shape + (3,), np.uint8)
    out[..., 0] = np.clip((yy + _ITUR_CVR * vi + half) >> _ITUR_SHIFT, 0, 255)
    out[..., 1] = np.clip((yy + _ITUR_CVG * vi + _ITUR_CUG * ui + half) >> _ITUR_SHIFT, 0, 255)
    out[..., 2] = np.clip((yy + _ITUR_CUB * ui + half) >> _ITUR_SHIFT, 0, 255)
    return out


def _upsample2x(c, h, w):
    """Chroma plane -> luma resolution by 2x2 pixel replication (OpenCV's)."""
    return np.repeat(np.repeat(c, 2, axis=0), 2, axis=1)[:h, :w]


def nv12_to_rgb(buf, width, height, stride=None, swap_uv=False):
    """NV12 (Y plane, then interleaved UV at half resolution) -> RGB uint8."""
    stride = int(stride or width)
    a = np.frombuffer(buf, dtype=np.uint8)
    ch = height // 2
    need = stride * (height + ch)
    if a.size < need:
        raise ValueError(f"NV12 payload too small: {a.size} < {need}")
    y = a[:stride * height].reshape(height, stride)[:, :width]
    uv = a[stride * height:need].reshape(ch, stride)[:, :width]
    u = uv[:, 0::2]
    v = uv[:, 1::2]
    if swap_uv:                                # NV21 is V first
        u, v = v, u
    return yuv_to_rgb(y, _upsample2x(u, height, width),
                      _upsample2x(v, height, width))


def yuv420_to_rgb(buf, width, height, stride=None):
    """Planar YUV420 / I420 (Y, then U plane, then V plane) -> RGB uint8."""
    stride = int(stride or width)
    cstride = max(1, stride // 2)
    a = np.frombuffer(buf, dtype=np.uint8)
    ch, cw = height // 2, width // 2
    y_end = stride * height
    u_end = y_end + cstride * ch
    need = u_end + cstride * ch
    if a.size < need:
        raise ValueError(f"YUV420 payload too small: {a.size} < {need}")
    y = a[:y_end].reshape(height, stride)[:, :width]
    u = a[y_end:u_end].reshape(ch, cstride)[:, :cw]
    v = a[u_end:need].reshape(ch, cstride)[:, :cw]
    return yuv_to_rgb(y, _upsample2x(u, height, width),
                      _upsample2x(v, height, width))


def rgb_to_rgb(buf, width, height, stride=None):
    """IMAGE_FORMAT_RGB: 24 bpp, `stride` bytes per row."""
    stride = int(stride or width * 3)
    a = np.frombuffer(buf, dtype=np.uint8)
    need = stride * height
    if a.size < need:
        raise ValueError(f"RGB payload too small: {a.size} < {need}")
    return a[:need].reshape(height, stride)[:, :width * 3].reshape(height, width, 3)


def raw8_to_rgb(buf, width, height, stride=None):
    """IMAGE_FORMAT_RAW8: 8-bit gray, replicated to three channels."""
    stride = int(stride or width)
    a = np.frombuffer(buf, dtype=np.uint8)
    need = stride * height
    if a.size < need:
        raise ValueError(f"RAW8 payload too small: {a.size} < {need}")
    gray = a[:need].reshape(height, stride)[:, :width]
    return np.repeat(gray[:, :, None], 3, axis=2)


def payload_to_rgb(payload, meta):
    """Dispatch on meta['format']; returns a contiguous (h,w,3) uint8 array."""
    if isinstance(payload, np.ndarray):        # FileSource hands RGB straight
        return np.ascontiguousarray(payload)
    w, h = int(meta["width"]), int(meta["height"])
    stride = int(meta["stride"]) or None
    fmt = int(meta["format"])
    if fmt == IMAGE_FORMAT_NV12:
        rgb = nv12_to_rgb(payload, w, h, stride)
    elif fmt == IMAGE_FORMAT_NV21:
        rgb = nv12_to_rgb(payload, w, h, stride, swap_uv=True)
    elif fmt == IMAGE_FORMAT_YUV420:
        rgb = yuv420_to_rgb(payload, w, h, stride)
    elif fmt == IMAGE_FORMAT_RGB:
        rgb = rgb_to_rgb(payload, w, h, stride)
    elif fmt == IMAGE_FORMAT_RAW8:
        rgb = raw8_to_rgb(payload, w, h, stride)
    else:
        raise ValueError(
            f"unsupported MPA image format {fmt} "
            f"({FORMAT_NAMES.get(fmt, 'unknown')}); reconfigure "
            f"voxl-camera-server to publish NV12/YUV420/RGB/RAW8")
    return np.ascontiguousarray(rgb)


# ---------------------------------------------------------------------------
# 4. Geometry: undistort (fisheye only) + anisotropic resize to 224
# ---------------------------------------------------------------------------
# doc 3 §4. Training squashed 640x480 -> 224x224 with cv2.INTER_LINEAR and NO
# centre crop (POLICY_RGB_PUBLISH, src/superfly/sim/px4_sim.py:945-947), so
# the resize here must stay anisotropic; a square crop would rescale every
# object horizontally and silently break the encoder.

DEFAULT_TARGET_HFOV_DEG = 91.0          # AG_FOV_X_DEG, px4_sim.py
DEFAULT_OUT_SIZE = 224                  # AG_NET_SIZE, px4_sim.py


def resize_bilinear(img, out_w, out_h):
    """cv2.INTER_LINEAR when cv2 is present, else a numpy bilinear twin.

    The fallback uses OpenCV's half-pixel-centre mapping and is within ~1 LSB
    of cv2 (cv2 rounds 5-bit fixed-point weights), but it is a fallback: the
    reference frames the policy was trained on came out of cv2.
    """
    if cv2 is not None:
        return cv2.resize(img, (int(out_w), int(out_h)),
                          interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    out_w, out_h = int(out_w), int(out_h)
    xs = (np.arange(out_w) + 0.5) * (w / out_w) - 0.5
    ys = (np.arange(out_h) + 0.5) * (h / out_h) - 0.5
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    ax = (xs - x0)[None, :, None]
    ay = (ys - y0)[:, None, None]
    f = img.astype(np.float32)
    top = f[y0][:, x0] * (1.0 - ax) + f[y0][:, x1] * ax
    bot = f[y1][:, x0] * (1.0 - ax) + f[y1][:, x1] * ax
    return np.clip(top * (1.0 - ay) + bot * ay + 0.5, 0, 255).astype(np.uint8)


def parse_cv_yaml(text):
    """Minimal OpenCV-FileStorage YAML reader (no pyyaml on the VOXL2).

    Handles what `voxl-calibrate-camera` writes to
    /data/modalai/opencv_<cam>_intrinsics.yml: a `%YAML:1.0` header, scalar
    keys (`distortion_model: fisheye`, `width: 640`), and `!!opencv-matrix`
    blocks with rows/cols/dt/data. Values come back as floats, strings, or
    numpy arrays.
    """
    out = {}
    text = text.replace("!!opencv-matrix", " ")
    lines = [ln for ln in text.splitlines()
             if not ln.lstrip().startswith(("%YAML", "---", "#"))]
    i = 0
    key_re = re.compile(r"^(\s*)([A-Za-z_][\w./-]*)\s*:\s*(.*)$")
    while i < len(lines):
        m = key_re.match(lines[i])
        if not m or m.group(1):                # only top-level keys start a block
            i += 1
            continue
        key, rest = m.group(2), m.group(3).strip()
        block = [rest]
        i += 1
        while i < len(lines):
            m2 = key_re.match(lines[i])
            if m2 and not m2.group(1):
                break
            block.append(lines[i])
            i += 1
        blob = "\n".join(block)
        if "data" in blob and "[" in blob:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                              blob[blob.index("["):])
            arr = np.array([float(x) for x in nums], dtype=np.float64)
            rows = re.search(r"rows\s*:\s*(\d+)", blob)
            cols = re.search(r"cols\s*:\s*(\d+)", blob)
            if rows and cols:
                r, c = int(rows.group(1)), int(cols.group(1))
                if arr.size >= r * c:
                    arr = arr[:r * c].reshape(r, c)
            out[key] = arr
        elif rest.startswith("[") or rest.startswith("!"):
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", rest)
            out[key] = np.array([float(x) for x in nums], dtype=np.float64)
        elif rest:
            v = rest.strip().strip('"').strip("'")
            try:
                out[key] = float(v) if re.match(r"^[-+]?[\d.eE+-]+$", v) else v
            except ValueError:
                out[key] = v
    return out


_K_KEYS = ("M", "K", "camera_matrix", "Matrix", "matrix", "intrinsic_matrix",
           "camera_intrinsics", "cameraMatrix")
_D_KEYS = ("D", "distortion_coefficients", "Distortion", "distortion",
           "distCoeffs", "dist_coeffs")


def load_intrinsics(path):
    """Read a ModalAI/OpenCV intrinsics YAML -> dict(K, D, width, height, model).

    Tolerant on purpose: ModalAI's tool has shipped several spellings across
    voxl-suite releases and there is no schema published. Whatever it is
    called, we need a 3x3 K, the distortion vector, and the resolution the
    calibration was taken at (so K can be scaled if the stream differs).
    """
    with open(path, "r") as fh:
        raw = parse_cv_yaml(fh.read())
    K = None
    for k in _K_KEYS:
        v = raw.get(k)
        if isinstance(v, np.ndarray) and v.size >= 9:
            K = np.asarray(v, np.float64).ravel()[:9].reshape(3, 3)
            break
    if K is None:                              # flat fx/fy/cx/cy form
        try:
            K = np.array([[float(raw["fx"]), 0.0, float(raw["cx"])],
                          [0.0, float(raw["fy"]), float(raw["cy"])],
                          [0.0, 0.0, 1.0]])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{path}: no camera matrix found "
                             f"(looked for {_K_KEYS} and fx/fy/cx/cy)")
    D = None
    for k in _D_KEYS:
        v = raw.get(k)
        if isinstance(v, np.ndarray) and v.size >= 1:
            D = np.asarray(v, np.float64).ravel()
            break
    if D is None:
        ks = [raw.get(f"k{i}") for i in (1, 2, 3, 4)]
        if any(x is not None for x in ks):
            D = np.array([float(x or 0.0) for x in ks], np.float64)
        else:
            raise ValueError(f"{path}: no distortion coefficients found")
    model = str(raw.get("distortion_model") or raw.get("Model")
                or raw.get("model") or "fisheye").strip().lower()
    width = int(raw.get("width") or raw.get("Width") or 0)
    height = int(raw.get("height") or raw.get("Height") or 0)
    return dict(K=K, D=D, width=width, height=height, model=model)


class Geometry:
    """Undistort (fisheye only) then anisotropic-resize to the net input.

    | `--lens`         | what happens |
    |------------------|--------------|
    | `rectilinear87`  | nothing but the resize -- doc 3 §4 accepts the 87 vs 91 deg deficit as-is |
    | `none`           | same, for bench sources with no lens model at all |
    | `fisheye120`     | Kannala-Brandt undistort onto a pinhole target sized so `--target-hfov` spans the full width, then the resize |

    The fisheye remap LUT is built once, on the first frame (the source's
    resolution is not known before then), never per frame -- doc 3 §3's
    10 ms undistort+resize budget assumes exactly that.
    """

    def __init__(self, lens="none", intrinsics=None, out_size=DEFAULT_OUT_SIZE,
                 target_hfov_deg=DEFAULT_TARGET_HFOV_DEG, undist_size=(640, 480)):
        self.lens = lens
        self.out_size = int(out_size)
        self.target_hfov_deg = float(target_hfov_deg)
        self.undist_w, self.undist_h = (int(undist_size[0]), int(undist_size[1]))
        self.intr = None
        self._map1 = self._map2 = None
        self._src_shape = None
        if lens == "fisheye120":
            if intrinsics is None:
                raise ValueError("--lens fisheye120 requires --intrinsics")
            if cv2 is None:
                raise RuntimeError("--lens fisheye120 requires OpenCV (cv2)")
            self.intr = (intrinsics if isinstance(intrinsics, dict)
                         else load_intrinsics(intrinsics))

    def target_K(self):
        """Pinhole K whose horizontal FOV is exactly `target_hfov_deg`."""
        fx = (self.undist_w / 2.0) / np.tan(np.radians(self.target_hfov_deg) / 2.0)
        return np.array([[fx, 0.0, self.undist_w / 2.0],
                         [0.0, fx, self.undist_h / 2.0],
                         [0.0, 0.0, 1.0]], np.float64)

    def _build_maps(self, src_h, src_w):
        K = np.array(self.intr["K"], np.float64)
        cal_w = self.intr["width"] or src_w
        cal_h = self.intr["height"] or src_h
        if (cal_w, cal_h) != (src_w, src_h):   # calibrated at another binning
            sx, sy = src_w / float(cal_w), src_h / float(cal_h)
            K = K.copy()
            K[0, :] *= sx
            K[1, :] *= sy
            _log(f"intrinsics scaled {cal_w}x{cal_h} -> {src_w}x{src_h}")
        D = np.asarray(self.intr["D"], np.float64).ravel()
        D = (np.resize(D, 4) if D.size >= 4 else
             np.concatenate([D, np.zeros(4 - D.size)])).reshape(4, 1)
        self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), self.target_K(),
            (self.undist_w, self.undist_h), cv2.CV_16SC2)
        self._src_shape = (src_h, src_w)
        _log(f"fisheye remap LUT built: {src_w}x{src_h} -> "
             f"{self.undist_w}x{self.undist_h} at {self.target_hfov_deg:.1f} deg HFOV")

    def __call__(self, rgb):
        if self.lens == "fisheye120":
            h, w = rgb.shape[:2]
            if self._map1 is None or self._src_shape != (h, w):
                self._build_maps(h, w)
            rgb = cv2.remap(rgb, self._map1, self._map2, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
        n = self.out_size
        if rgb.shape[0] != n or rgb.shape[1] != n:
            rgb = resize_bilinear(rgb, n, n)   # anisotropic on purpose
        return np.ascontiguousarray(rgb)


# ---------------------------------------------------------------------------
# 5. Rate control, stats, PNG dump
# ---------------------------------------------------------------------------

class Decimator:
    """Deadline-based frame decimation, e.g. 30 Hz camera -> 15 Hz link.

    Deadline-based rather than modulo-based so it is right when the camera's
    real rate is not an exact multiple of the target (or drifts): it emits at
    most one frame per 1/fps window and never accumulates debt.
    """

    def __init__(self, fps):
        self.fps = float(fps)
        self.period = 1.0 / self.fps if self.fps > 0 else 0.0
        self._next = None

    def accept(self, t):
        if self.period <= 0.0:
            return True
        if self._next is None:
            self._next = t
        if t >= self._next - 1e-9:
            self._next += self.period
            if self._next <= t:                # fell behind: resync, no burst
                self._next = t + self.period
            return True
        return False


class Stats:
    """Every `interval` s: achieved fps in/out, send errors, frame age."""

    def __init__(self, interval=2.0, enabled=True):
        self.interval = float(interval)
        self.enabled = bool(enabled)
        self.t0 = time.monotonic()
        self.reset(self.t0)
        self.total_in = 0
        self.total_out = 0

    def reset(self, now):
        self._t = now
        self.n_in = 0
        self.n_out = 0
        self.age_sum = 0.0
        self.age_max = 0.0
        self.n_age = 0

    def frame_in(self):
        self.n_in += 1
        self.total_in += 1

    def frame_out(self, age_s=None):
        self.n_out += 1
        self.total_out += 1
        if age_s is not None and age_s >= 0.0:
            self.age_sum += age_s
            self.age_max = max(self.age_max, age_s)
            self.n_age += 1

    def maybe_print(self, now, pub, source, extra=""):
        if not self.enabled or now - self._t < self.interval:
            return
        dt = now - self._t
        mean_age = (self.age_sum / self.n_age * 1e3) if self.n_age else float("nan")
        _log("stats: in %5.1f Hz  out %5.1f Hz  sent %d  send_err %d  "
             "reconnects %d  age mean %.1f ms max %.1f ms%s"
             % (self.n_in / dt, self.n_out / dt, pub.frames_sent if pub else 0,
                pub.send_errors if pub else 0, getattr(source, "reconnects", 0),
                mean_age, self.age_max * 1e3, (" " + extra) if extra else ""))
        self.reset(now)


def write_png(path, rgb):
    """Minimal RGB8 PNG writer -- no cv2 needed, so --dump-dir works on a
    stock VOXL2 image where only numpy is installed."""
    h, w, _ = np.asarray(rgb).shape
    raw = b"".join(b"\x00" + bytes(rgb[i]) for i in range(h))

    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)


# ---------------------------------------------------------------------------
# 6. Main loop
# ---------------------------------------------------------------------------

_RUNNING = True


def _stop(signum, _frame):
    global _RUNNING
    _RUNNING = False
    _log(f"signal {signum} received, shutting down")


def build_source(args):
    if args.source == "mpa":
        return MPASource(args.pipe, client_name=args.client_name)
    spec = args.input
    if spec is None:
        spec = "0" if args.source == "v4l2" else None
    if spec is None:
        raise SystemExit("--source file needs --input <video path>")
    return FileSource(spec, source_fps=args.source_fps, loop=args.loop)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="VOXL2 hires camera -> 224x224 RGB -> SFRG/UDP bridge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_argument_group("source")
    src.add_argument("--source", choices=("mpa", "file", "v4l2"), default="mpa",
                     help="mpa = the VOXL Modal Pipe Architecture camera pipe; "
                          "file/v4l2 = bench replay through the same pipeline")
    src.add_argument("--pipe", default="hires_front",
                     help="MPA pipe name or /run/mpa/<name>/ path "
                          "(enumerate with `voxl-list-pipes`)")
    src.add_argument("--client-name", default="rgb_bridge",
                     help="MPA client name; 8 random digits are appended")
    src.add_argument("--input", default=None,
                     help="video file path, or camera index for --source v4l2")
    src.add_argument("--source-fps", type=float, default=None,
                     help="pace a file source at this rate (default: the file's)")
    src.add_argument("--loop", action="store_true",
                     help="loop a file source at EOF")

    geo = p.add_argument_group("geometry (doc 3 section 4)")
    geo.add_argument("--lens", choices=("rectilinear87", "fisheye120", "none"),
                     default="rectilinear87")
    geo.add_argument("--intrinsics", default="/data/modalai/opipe_hires_intrinsics.yml",
                     help="voxl-calibrate-camera YAML, required for fisheye120")
    geo.add_argument("--target-hfov", type=float, default=DEFAULT_TARGET_HFOV_DEG,
                     help="pinhole HFOV the fisheye is undistorted onto (training FOV)")
    geo.add_argument("--undist-size", default="640x480",
                     help="pinhole target size, 4:3, before the 224 resize")
    geo.add_argument("--out-size", type=int, default=DEFAULT_OUT_SIZE,
                     help="net input edge; the resize is anisotropic")

    net = p.add_argument_group("link")
    net.add_argument("--host", default="127.0.0.1", help="ground station IP")
    net.add_argument("--port", type=int, default=DEFAULT_RGB_PORT)
    net.add_argument("--fps", type=float, default=15.0,
                     help="publish rate; the camera runs at 30 and is decimated")
    net.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK,
                     help="UDP fragment payload; try 1400 to stay under the MTU")
    net.add_argument("--zlib-level", type=int, default=DEFAULT_ZLIB_LEVEL,
                     help="1 if zlib is the bottleneck on the Kryo core")

    dbg = p.add_argument_group("diagnostics")
    dbg.add_argument("--stats", action="store_true",
                     help="print in/out rate, send errors and frame age")
    dbg.add_argument("--stats-interval", type=float, default=2.0)
    dbg.add_argument("--dump-dir", default=None,
                     help="save every Nth published 224 frame as PNG")
    dbg.add_argument("--dump-every", type=int, default=30)
    dbg.add_argument("--dry-run", action="store_true",
                     help="run the whole pipeline but send nothing")
    dbg.add_argument("--duration", type=float, default=0.0,
                     help="exit after this many seconds (0 = forever)")
    dbg.add_argument("--max-frames", type=int, default=0,
                     help="exit after publishing this many frames (0 = forever)")
    return p.parse_args(argv)


def main(argv=None):
    global _RUNNING
    _RUNNING = True
    args = parse_args(argv)
    try:
        uw, uh = (int(x) for x in str(args.undist_size).lower().split("x"))
    except ValueError:
        raise SystemExit(f"--undist-size must look like 640x480, got {args.undist_size!r}")

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    geom = Geometry(lens=args.lens,
                    intrinsics=(args.intrinsics if args.lens == "fisheye120" else None),
                    out_size=args.out_size, target_hfov_deg=args.target_hfov,
                    undist_size=(uw, uh))
    source = build_source(args)
    pub = None if args.dry_run else RGBPublisher(
        args.host, args.port, chunk=args.chunk_bytes, level=args.zlib_level)
    decim = Decimator(args.fps)
    stats = Stats(args.stats_interval, enabled=args.stats)
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

    # No nested f-string here: 3.6 accepts one, but only as long as the inner
    # quotes differ from the outer, which is a trap for the next edit.
    sink = "DRY RUN" if args.dry_run else "%s:%d" % (args.host, args.port)
    have_cv2 = "yes" if cv2 is not None else "NO - numpy fallback"
    _log(f"source={args.source} lens={args.lens} out={args.out_size}x{args.out_size} "
         f"-> {sink} at {args.fps} Hz (cv2 {have_cv2})")

    t_start = time.monotonic()
    n_dumped = 0
    fmt_seen = None
    try:
        while _RUNNING:
            now = time.monotonic()
            if args.duration and now - t_start >= args.duration:
                break
            got = source.read(timeout=0.2)
            stats.maybe_print(time.monotonic(), pub, source)
            if got is None:
                if getattr(source, "ended", False):
                    _log("source exhausted (end of file)")
                    break
                continue
            meta, payload = got
            stats.frame_in()
            t_now = time.monotonic()
            if not decim.accept(t_now):
                continue
            if fmt_seen != meta["format"]:
                fmt_seen = meta["format"]
                _log(f"stream: {meta['width']}x{meta['height']} "
                     f"{FORMAT_NAMES.get(fmt_seen, fmt_seen)} stride={meta['stride']} "
                     f"{meta['framerate']} Hz")
            try:
                rgb = payload_to_rgb(payload, meta)
                frame = geom(rgb)
            except (ValueError, RuntimeError) as exc:
                _log(f"frame dropped: {exc}")
                continue
            if pub is not None:
                pub.send(frame)                # never raises
            # timestamp_ns is the apps-proc CLOCK_MONOTONIC start of exposure,
            # the same clock as _monotonic_ns() -> real glass-to-wire age.
            age = None
            ts = int(meta.get("timestamp_ns") or 0)
            if ts > 0:
                age = (_monotonic_ns() - ts) * 1e-9
                if not (0.0 <= age < 10.0):    # clock mismatch: don't report junk
                    age = None
            stats.frame_out(age)
            every = max(1, args.dump_every)
            if args.dump_dir and (stats.total_out - 1) % every == 0:
                try:
                    write_png(os.path.join(args.dump_dir,
                                           "frame_%06d.png" % stats.total_out), frame)
                    n_dumped += 1
                except OSError as exc:
                    _log(f"dump failed: {exc}")
            if args.max_frames and stats.total_out >= args.max_frames:
                break
    finally:
        source.close("shutdown")
        if pub is not None:
            pub.close()
    dt = max(1e-9, time.monotonic() - t_start)
    _log("exit: %d frames in, %d published (%.2f Hz), %d send errors, "
         "%d reconnects, %d dumped"
         % (stats.total_in, stats.total_out, stats.total_out / dt,
            pub.send_errors if pub else 0, getattr(source, "reconnects", 0),
            n_dumped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
