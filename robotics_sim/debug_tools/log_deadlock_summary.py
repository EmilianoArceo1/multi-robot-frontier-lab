"""
Parse a simulator console log and summarize multi-robot HOLD/deadlock symptoms.

Usage:
    python debug_tools/log_deadlock_summary.py "Pasted text.txt"
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

STATE_RE = re.compile(r"R(?P<robot>\d+) state=(?P<state>[A-Z_]+); reason=(?P<reason>.*)")
MOVE_RE = re.compile(
    r"R(?P<robot>\d+) move @ t=(?P<t>[\d.]+)s: pos=\((?P<x>[-\d.]+), (?P<y>[-\d.]+)\).*target=\((?P<tx>[-\d.]+), (?P<ty>[-\d.]+)\)"
)
ROUTE_RE = re.compile(
    r"R(?P<robot>\d+) route assigned: start=\((?P<x>[-\d.]+), (?P<y>[-\d.]+)\), target=\((?P<tx>[-\d.]+), (?P<ty>[-\d.]+)\), waypoints=(?P<w>\d+)"
)


def main(path: str) -> int:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    states = Counter()
    reasons = Counter()
    last_pos = {}
    first_pos = {}
    route_targets = defaultdict(list)

    for line in text.splitlines():
        if m := STATE_RE.search(line):
            robot = int(m.group("robot"))
            state = m.group("state")
            reason = m.group("reason")
            states[(robot, state)] += 1
            reasons[(robot, reason)] += 1
        if m := MOVE_RE.search(line):
            robot = int(m.group("robot"))
            item = (
                float(m.group("t")),
                float(m.group("x")),
                float(m.group("y")),
                float(m.group("tx")),
                float(m.group("ty")),
            )
            first_pos.setdefault(robot, item)
            last_pos[robot] = item
        if m := ROUTE_RE.search(line):
            robot = int(m.group("robot"))
            route_targets[robot].append((float(m.group("tx")), float(m.group("ty")), int(m.group("w"))))

    print("STATE COUNTS")
    for (robot, state), count in sorted(states.items()):
        print(f"  R{robot} {state}: {count}")

    print("\nDISTANCE MOVED")
    for robot in sorted(last_pos):
        t0, x0, y0, _, _ = first_pos[robot]
        t1, x1, y1, tx, ty = last_pos[robot]
        moved = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        remaining = ((tx - x1) ** 2 + (ty - y1) ** 2) ** 0.5
        print(f"  R{robot}: moved={moved:.3f} m over {t1-t0:.2f}s; last_remaining={remaining:.3f} m")

    print("\nROUTE TARGETS")
    for robot, targets in sorted(route_targets.items()):
        print(f"  R{robot}: {targets}")

    print("\nTOP HOLD REASONS")
    for (robot, reason), count in reasons.most_common(10):
        if "frontier" in reason.lower() or "hold" in reason.lower():
            print(f"  R{robot}: {count}x {reason}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__.strip())
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
