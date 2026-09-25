"""Trial-slot index: the one knob that makes every per-trial resource unique.

The comparison harness can fly several trials at once (`run_comparison.py
--parallel N`). Every resource a trial binds -- PX4's three UDP/TCP ports, the
depth/RGB/debug UDP ports, the sentinel files, the acados codegen dir -- is a
function of a single integer `k`, the *slot*, exported to every child process
as `$SUPERFLY_SLOT`.

Slot 0 is the historical single-trial configuration: every port and every file
name it produces is byte-identical to what the harness used before slots
existed, so a `--parallel 1` run (and any manual invocation with the variable
unset) behaves exactly as it always did.

Who reads it:

| Module | What it offsets |
|---|---|
| `superfly.common.sentinels` | `/tmp/superfly_*` file names |
| `superfly.common.transport` | depth UDP port 15001+10k |
| `superfly.common.agile_debug_transport` | debug UDP port 15002+10k |
| `superfly.common.rgb_transport` | RGB UDP port 15003+10k |
| `superfly.sim.px4_sim` | PX4 simulator TCP port 4560+k (`--slot`) |
| `superfly.compare.runner` | PX4 instance `-i k`, MAVLink 14550+k |

The sensor ports step by `SLOT_PORT_STRIDE`, not by 1. Their bases sit one
apart, so a stride of 1 would alias across slots -- slot 1's debug port
(15002+1) is slot 0's RGB port (15003+0), which is exactly how run
`f2_probe_p2_a` died with `Errno 98` on the RGB bind. The stride must exceed
the span of the base block; 10 leaves room for a fourth sensor and still fits
ten slots below 15100. PX4's ports keep a stride of 1 because that is what
`px4 -i k` itself does, and PX4's own bases are hundreds apart.

The value is read once at import: a process belongs to exactly one slot for
its whole life, and the transports bake their port into a module constant.
"""

import os

SLOT_ENV = "SUPERFLY_SLOT"

# Spacing between consecutive slots' sensor UDP ports (see the table above).
SLOT_PORT_STRIDE = 10


def slot_index() -> int:
    """This process's slot `k` (`$SUPERFLY_SLOT`, default 0). Never raises:
    an unparseable value falls back to 0 rather than killing a flight."""
    try:
        return max(0, int(os.environ.get(SLOT_ENV, "0")))
    except (TypeError, ValueError):
        return 0


def slot_suffix(slot: int | None = None) -> str:
    """Name suffix for slot `k`: "" for slot 0 (historical names kept exactly),
    "_slot{k}" otherwise."""
    k = slot_index() if slot is None else int(slot)
    return "" if k == 0 else f"_slot{k}"


def slot_port(base: int, slot: int | None = None) -> int:
    """Sensor port for slot `k`: `base + SLOT_PORT_STRIDE*k`, so slot 0 keeps
    the historical port and no two slots can alias onto each other (see the
    module docstring for why the stride is not 1)."""
    k = slot_index() if slot is None else int(slot)
    return int(base) + SLOT_PORT_STRIDE * k
