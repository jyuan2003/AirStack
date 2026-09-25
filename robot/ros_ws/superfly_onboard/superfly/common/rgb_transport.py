"""Lossless fragmented UDP transport for 224x224 RGB policy frames.

The RGB counterpart of superfly.common.transport (depth). The CL4Nav encoder
is contrastively trained on uint8 RGB, so unlike depth there is no lossy
codec here: the publisher zlib-compresses the exact uint8 frame the sim
rendered and fragments it across datagrams (a 224x224x3 frame is 150 KB, well
over the 65507-byte UDP payload cap). The subscriber reassembles by sequence
number and drops incomplete frames, so the policy always sees a byte-exact
frame or none at all.

Wire format (little-endian), one header per fragment:
    "SFRG" | uint8 version | 3 pad | uint32 seq | uint32 h | uint32 w
          | uint32 part | uint32 n_parts | zlib(uint8 HxWx3) chunk

send() never raises on transport errors -- losing one RGB frame must never
kill the sim loop.
"""

from __future__ import annotations

import os
import socket
import struct
import sys
import time
import zlib

import numpy as np

from superfly.common.slot import slot_port


# Offset by this process's trial slot (superfly.common.slot, stride 10) so
# concurrent trials never share a socket; slot 0 keeps the historical 15003.
RGB_PORT = slot_port(15003)

# Interface the subscriber binds by default. Loopback is right whenever the
# sim and the policy share a host (every harness run, and the docker runtime,
# which uses host networking). Set SUPERFLY_RGB_BIND=0.0.0.0 to accept frames
# from another machine -- a container on its own network, or the ground-control
# deployment where Isaac and the policy sit on different hosts.
RGB_BIND_ENV = "SUPERFLY_RGB_BIND"
DEFAULT_RGB_BIND = "127.0.0.1"

# A frame older than this is not worth acting on: at the ~10 Hz the policy
# consumes RGB, 0.25 s is two and a half missed frames, which at 5 m/s means
# the drone has moved more than a metre since the picture was taken. Consumers
# compare `RGBSubscriber.age_s()` against it and hold/abort rather than steer
# on a stale view. Defined here, next to the receive timestamp that feeds it,
# so publisher and consumer cannot disagree about what "stale" means.
RGB_STALE_S = 0.25

MAGIC = b"SFRG"
VERSION = 1
_HEADER = struct.Struct("<4sB3xIIIII")
_CHUNK = 60000


class RGBPublisher:
    """Send uint8 RGB frames losslessly, fragmenting above the UDP limit."""

    def __init__(self, host: str = "127.0.0.1", port: int = RGB_PORT):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0
        self._send_errors = 0

    def send(self, rgb: np.ndarray) -> None:
        frame = np.asarray(rgb)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"RGB frame must be HxWx3, got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        h, w, _ = frame.shape
        body = zlib.compress(frame.tobytes(), 3)
        n_parts = max(1, -(-len(body) // _CHUNK))
        try:
            for part in range(n_parts):
                chunk = body[part * _CHUNK:(part + 1) * _CHUNK]
                packet = _HEADER.pack(
                    MAGIC, VERSION, self._seq, h, w, part, n_parts
                ) + chunk
                self._sock.sendto(packet, self._addr)
        except OSError as exc:
            self._send_errors += 1
            if self._send_errors < 3 or self._send_errors % 100 == 0:
                print(
                    f"[rgb_transport] send failed ({exc}); "
                    f"{self._send_errors} frames dropped so far.",
                    file=sys.stderr,
                    flush=True,
                )
        self._seq += 1


class RGBSubscriber:
    """Drain UDP and return the newest completely reassembled RGB frame.

    Binding
    -------
    `host=None` (the default) binds `$SUPERFLY_RGB_BIND`, itself defaulting to
    `127.0.0.1`. Set that variable to `0.0.0.0` to receive frames from another
    host or from a container on its own network; pass `host=` explicitly to
    override both. The port is `RGB_PORT`, already offset by this process's
    trial slot.

    Frame age
    ---------
    UDP gives no back-pressure, so a subscriber that has stopped receiving
    keeps handing out the last frame it ever saw, forever, and a policy reading
    it cannot tell a live view from a frozen one. Three accessors, all cheap
    and all draining the socket first:

    | Call | Returns |
    |---|---|
    | `latest()` | newest frame, or None -- unchanged, age not considered |
    | `latest_with_age()` | `(frame, age_s)`; `(None, None)` before the first frame |
    | `age_s()` | seconds since the newest frame was received, or None |

    `age_s()` is measured from the receive time in *this* process, so it needs
    no clock agreement with the publisher and counts reassembly latency too. A
    consumer that must not act on a stale view compares it against the module
    constant `RGB_STALE_S`:

        frame, age = sub.latest_with_age()
        if frame is None or age > rgb_transport.RGB_STALE_S:
            ...  # hold the last command / abort, do not steer on this frame
    """

    def __init__(self, host: str | None = None, port: int = RGB_PORT):
        if host is None:
            host = os.environ.get(RGB_BIND_ENV, DEFAULT_RGB_BIND)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.setblocking(False)
        self._last = None
        self._last_t = None      # monotonic receive time of self._last
        self._frame_seq = None   # wire seq of the newest COMPLETE frame
        self._seq = None
        self._shape = None
        self._n_parts = 0
        self._parts = {}

    def latest_with_age(self):
        """(newest frame, seconds since it was received), or (None, None)."""
        frame = self.latest()
        if frame is None or self._last_t is None:
            return None, None
        return frame, time.monotonic() - self._last_t

    def last_frame_id(self):
        """`(seq, monotonic receive time)` of the newest COMPLETE frame.

        `(None, None)` until the first frame is reassembled. Drains the socket
        first, like every other accessor here. A consumer that must advance a
        recurrent policy exactly once per rendered frame -- the control loop
        runs on the wall clock, the renderer on the simulation clock, and the
        two only agree at realtime factor 1 -- compares this tuple against the
        one it stepped on last:

            fid = sub.last_frame_id()
            if fid != last_fid:          # a NEW frame, step the policy
                ...
        """
        self.latest()
        return self._frame_seq, self._last_t

    def age_s(self):
        """Seconds since the newest frame arrived, or None if none ever has.
        Drains the socket first, so an arriving frame resets it immediately."""
        self.latest()
        if self._last_t is None:
            return None
        return time.monotonic() - self._last_t

    def latest(self):
        while True:
            try:
                data = self._sock.recv(65535)
            except BlockingIOError:
                break
            if len(data) < _HEADER.size:
                continue
            magic, version, seq, h, w, part, n_parts = _HEADER.unpack_from(data)
            if (magic != MAGIC or version != VERSION or h == 0 or w == 0
                    or n_parts == 0 or part >= n_parts):
                continue
            if seq != self._seq:
                self._seq = seq
                self._shape = (h, w)
                self._n_parts = n_parts
                self._parts = {}
            if self._shape != (h, w) or self._n_parts != n_parts:
                continue
            self._parts[part] = data[_HEADER.size:]
            if len(self._parts) != self._n_parts:
                continue
            try:
                packed = b"".join(self._parts[i] for i in range(self._n_parts))
                raw = zlib.decompress(packed)
            except (KeyError, zlib.error):
                self._parts = {}
                continue
            if len(raw) != h * w * 3:
                self._parts = {}
                continue
            self._last = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            self._last_t = time.monotonic()
            self._frame_seq = seq
            self._parts = {}
        return self._last
