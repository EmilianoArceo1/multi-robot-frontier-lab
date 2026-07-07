from __future__ import annotations

import math


MIN_EDITOR_OBSTACLE_SIZE = 0.2

ObstacleRect = tuple[float, float, float, float]
Point2D = tuple[float, float]


def create_rect_obstacle_from_drag(
    start_xy: Point2D,
    end_xy: Point2D,
    min_size: float = MIN_EDITOR_OBSTACLE_SIZE,
) -> ObstacleRect | None:
    """Create a normalized axis-aligned rectangle obstacle from a drag gesture."""
    x0, y0 = start_xy
    x1, y1 = end_xy

    width = abs(x1 - x0)
    height = abs(y1 - y0)

    if width < min_size and height < min_size:
        return None

    left = min(x0, x1)
    bottom = min(y0, y1)
    width = max(width, min_size)
    height = max(height, min_size)

    return normalize_obstacle((left, bottom, width, height))


def create_free_draw_obstacles_from_path(
    points: list[Point2D],
    brush_size: float = 0.2,
) -> list[ObstacleRect]:
    """Create circular-brush obstacle stamps along a free-draw stroke.

    The simulator still stores obstacles as axis-aligned bounding rectangles
    because the runtime collision/planning stack expects ``(x, y, w, h)``.
    The editor renderer interprets these dense square stamps as circular brush
    marks and draws their visual union, so the stroke looks smooth and behaves
    like one object without destructive bounding-box merges.
    """
    if brush_size <= 0.0 or not points:
        return []

    brush = float(brush_size)
    half = brush / 2.0

    # Keep spacing below the brush radius so adjacent circular stamps overlap.
    # That makes the rendered union look continuous instead of bead-like.
    spacing = max(brush * 0.38, 0.02)

    sampled_points: list[Point2D] = []
    previous: Point2D | None = None

    for raw_point in points:
        current = (float(raw_point[0]), float(raw_point[1]))
        if previous is None:
            sampled_points.append(current)
            previous = current
            continue

        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        distance = math.hypot(dx, dy)
        steps = max(1, int(math.ceil(distance / spacing)))
        for step in range(1, steps + 1):
            t = step / steps
            sampled_points.append((previous[0] + dx * t, previous[1] + dy * t))
        previous = current

    # Quantize centers to avoid hundreds of nearly identical stamps from dense
    # mouse events. Do not merge stamps into rectangles: merging is visual only.
    quantized: dict[tuple[int, int], Point2D] = {}
    quant = max(spacing * 0.75, 0.015)
    for x, y in sampled_points:
        key = (int(round(x / quant)), int(round(y / quant)))
        quantized[key] = (x, y)

    return [
        normalize_obstacle((x - half, y - half, brush, brush))
        for x, y in quantized.values()
    ]


def create_square_obstacle_from_drag(
    start_xy: Point2D,
    end_xy: Point2D,
    min_size: float = MIN_EDITOR_OBSTACLE_SIZE,
) -> ObstacleRect | None:
    """Create a normalized square obstacle from a drag gesture."""
    rect = create_rect_obstacle_from_drag(start_xy, end_xy, min_size=min_size)
    if rect is None:
        return None

    left, bottom, width, height = rect
    size = max(width, height)

    return normalize_obstacle((left, bottom, size, size))


def normalize_obstacle(obstacle: ObstacleRect) -> ObstacleRect:
    """Return an obstacle with positive width and height."""
    x, y, width, height = obstacle
    x = float(x)
    y = float(y)
    width = float(width)
    height = float(height)

    if width < 0.0:
        x += width
        width = abs(width)
    if height < 0.0:
        y += height
        height = abs(height)

    return (float(x), float(y), float(width), float(height))


def normalize_obstacles(obstacles: list[ObstacleRect]) -> list[ObstacleRect]:
    """Normalize and drop degenerate obstacle rectangles."""
    normalized: list[ObstacleRect] = []
    for obstacle in obstacles:
        x, y, width, height = normalize_obstacle(obstacle)
        if width <= 0.0 or height <= 0.0:
            continue
        normalized.append((x, y, width, height))
    return normalized


def find_obstacle_at(
    obstacles: list[ObstacleRect],
    point_xy: Point2D,
) -> int | None:
    """Return the index of the topmost obstacle containing the point, if any."""
    px, py = point_xy

    # Reverse order matches visual stacking: the last drawn obstacle is easiest
    # to select when rectangles overlap.
    for index in range(len(obstacles) - 1, -1, -1):
        ox, oy, ow, oh = obstacles[index]

        if ox <= px <= ox + ow and oy <= py <= oy + oh:
            return index

    return None


def _rectangles_touch_or_overlap(
    first: ObstacleRect,
    second: ObstacleRect,
    tolerance: float = 1.0e-9,
) -> bool:
    """Return True when two obstacle rectangles overlap or touch."""
    ax, ay, aw, ah = normalize_obstacle(first)
    bx, by, bw, bh = normalize_obstacle(second)

    return not (
        ax + aw < bx - tolerance
        or bx + bw < ax - tolerance
        or ay + ah < by - tolerance
        or by + bh < ay - tolerance
    )


def connected_obstacle_indices(
    obstacles: list[ObstacleRect],
    start_index: int,
) -> list[int]:
    """Return all obstacle indices connected to ``start_index``.

    Connectivity is based on touching/overlapping bounding rectangles. This is
    intentionally non-destructive: connected obstacles may render and move as
    one visual object without modifying the underlying obstacle list.
    """
    if start_index < 0 or start_index >= len(obstacles):
        return []

    visited: set[int] = set()
    pending = [int(start_index)]

    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)

        for index, obstacle in enumerate(obstacles):
            if index in visited:
                continue
            if _rectangles_touch_or_overlap(obstacles[current], obstacle):
                pending.append(index)

    return sorted(visited)


def find_obstacle_group_at(
    obstacles: list[ObstacleRect],
    point_xy: Point2D,
) -> list[int]:
    """Return the connected visual object under ``point_xy``.

    The topmost hit rectangle is used as the seed, then all touching/overlapping
    rectangles are returned. This makes free-draw strokes and manually joined
    objects move together.
    """
    index = find_obstacle_at(obstacles, point_xy)
    if index is None:
        return []
    return connected_obstacle_indices(obstacles, index)


def move_obstacles_by(
    obstacles: list[ObstacleRect],
    indices: list[int],
    delta_xy: Point2D,
) -> bool:
    """Move several obstacles by a delta while preserving their sizes."""
    if not indices:
        return False

    dx, dy = float(delta_xy[0]), float(delta_xy[1])
    changed = False
    unique_indices = sorted(set(int(index) for index in indices))

    for index in unique_indices:
        if index < 0 or index >= len(obstacles):
            continue
        x, y, width, height = normalize_obstacle(obstacles[index])
        obstacles[index] = normalize_obstacle((x + dx, y + dy, width, height))
        changed = True

    return changed


def move_obstacle_to(
    obstacles: list[ObstacleRect],
    index: int,
    left_bottom_xy: Point2D,
) -> bool:
    """Move one obstacle while preserving its size."""
    if index < 0 or index >= len(obstacles):
        return False

    _, _, width, height = normalize_obstacle(obstacles[index])
    left, bottom = left_bottom_xy
    obstacles[index] = normalize_obstacle((float(left), float(bottom), width, height))
    return True


def remove_obstacle_at(
    obstacles: list[ObstacleRect],
    point_xy: Point2D,
) -> bool:
    """Remove the topmost obstacle containing the point.

    Returns True when an obstacle was removed.
    """
    index = find_obstacle_at(obstacles, point_xy)

    if index is None:
        return False

    del obstacles[index]
    return True


def _intervals_touch_or_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return not (a1 < b0 or b1 < a0)


def _almost_equal(first: float, second: float, tolerance: float = 1.0e-9) -> bool:
    return abs(float(first) - float(second)) <= tolerance


def _can_merge_without_changing_shape(first: ObstacleRect, second: ObstacleRect) -> bool:
    """Return True only when the union remains a rectangle.

    This avoids the destructive behavior where two diagonal/offset rectangles
    are replaced by a large bounding box that fills space the user did not draw.
    """
    ax, ay, aw, ah = normalize_obstacle(first)
    bx, by, bw, bh = normalize_obstacle(second)

    a_right = ax + aw
    a_top = ay + ah
    b_right = bx + bw
    b_top = by + bh

    same_x_span = _almost_equal(ax, bx) and _almost_equal(a_right, b_right)
    same_y_span = _almost_equal(ay, by) and _almost_equal(a_top, b_top)

    if same_x_span and _intervals_touch_or_overlap(ay, a_top, by, b_top):
        return True
    if same_y_span and _intervals_touch_or_overlap(ax, a_right, bx, b_right):
        return True

    # Containment is safe because the merged shape is the containing rectangle.
    first_contains_second = ax <= bx and ay <= by and a_right >= b_right and a_top >= b_top
    second_contains_first = bx <= ax and by <= ay and b_right >= a_right and b_top >= a_top
    return bool(first_contains_second or second_contains_first)


def _merge_pair(
    first: ObstacleRect,
    second: ObstacleRect,
) -> ObstacleRect:
    """Merge two compatible axis-aligned rectangles into one rectangle."""
    ax, ay, aw, ah = normalize_obstacle(first)
    bx, by, bw, bh = normalize_obstacle(second)

    left = min(ax, bx)
    bottom = min(ay, by)
    right = max(ax + aw, bx + bw)
    top = max(ay + ah, by + bh)

    return normalize_obstacle((left, bottom, right - left, top - bottom))


def merge_obstacles(
    obstacles: list[ObstacleRect],
) -> list[ObstacleRect]:
    """Merge only when doing so preserves the exact rectangular union.

    Offset overlaps, diagonal free-draw stamps, and L-shaped compositions are
    kept as separate rectangles. That preserves what the user drew instead of
    replacing it with a larger bounding rectangle.
    """
    pending = normalize_obstacles(obstacles)
    if not pending:
        return []

    merged: list[ObstacleRect] = []
    for obstacle in pending:
        current = obstacle
        changed = True
        while changed:
            changed = False
            remaining: list[ObstacleRect] = []
            for existing in merged:
                if _can_merge_without_changing_shape(current, existing):
                    current = _merge_pair(current, existing)
                    changed = True
                else:
                    remaining.append(existing)
            merged = remaining
        merged.append(current)

    return merged
