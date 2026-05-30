import time

import math
from typing import Optional

class FPSCounter:
    def __init__(self, buffer_size: int = 15):
        self.time_buffer = [time.time()] * buffer_size
        self.buffer_size = buffer_size

    def get_fps(self) -> float:
        self.time_buffer.append(time.time())
        return (self.buffer_size - 1) / (self.time_buffer[-1] - self.time_buffer.pop(0))


def get_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return 0
    cos_theta = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_theta))


def _add_anomaly(anomalies: dict[str, list[int]], name: str, frame_index: int) -> None:
    anomalies.setdefault(name, []).append(frame_index)


def _bbox_metrics(bbox: tuple[int, int, int, int]) -> tuple[float, float, float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    area = width * height
    aspect_ratio = width / height
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return width, height, area, aspect_ratio, center_x, center_y


def detect_motion_anomalies(
    points: list[tuple[int, int]],
    boxes: Optional[list[tuple[int, int, int, int]]] = None,
    speed_thresh: float = 20.0,
    accel_thresh: float = 15.0,
    angle_thresh: float = 60.0,
    min_turn_speed: float = 4.0,
    sudden_stop_ratio: float = 0.35,
    bbox_area_change_thresh: float = 0.45,
    bbox_aspect_change_thresh: float = 0.55,
    bbox_height_drop_thresh: float = 0.35,
    fallen_aspect_ratio_thresh: float = 1.15,
    min_box_area: float = 64.0,
) -> dict[str, list[int]]:
    """Find motion and bbox-shape signs that can point to a crash/fall.

    Returned values are frame indexes inside the passed history window.
    """
    anomalies: dict[str, list[int]] = {}

    for i in range(2, len(points)):
        p0, p1, p2 = points[i-2], points[i-1], points[i]

        # Speed is the distance between points
        speed1 = get_distance(p0, p1)
        speed2 = get_distance(p1, p2)

        # High Speed
        if speed2 > speed_thresh:
            _add_anomaly(anomalies, "high_speed", i)

        # Acceleration spike
        if abs(speed2 - speed1) > accel_thresh:
            _add_anomaly(anomalies, "acceleration_spike", i)

        # After impact the person often goes from fast movement to almost no movement.
        if speed1 > speed_thresh and speed2 < speed1 * sudden_stop_ratio:
            _add_anomaly(anomalies, "sudden_stop", i)

        # Sharp change of direction (Sharp turn)
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        angle = angle_between(v1, v2)

        if angle > angle_thresh and min(speed1, speed2) > min_turn_speed:
            _add_anomaly(anomalies, "sharp_turn", i)

        if angle > 120 and min(speed1, speed2) > min_turn_speed:
            _add_anomaly(anomalies, "direction_reversal", i)

    if boxes is None:
        return anomalies

    for i in range(1, len(boxes)):
        prev_width, prev_height, prev_area, prev_ratio, _, prev_center_y = _bbox_metrics(boxes[i-1])
        width, height, area, ratio, _, center_y = _bbox_metrics(boxes[i])

        if prev_area < min_box_area or area < min_box_area:
            continue

        area_change = abs(area - prev_area) / prev_area
        if area_change > bbox_area_change_thresh:
            _add_anomaly(anomalies, "bbox_size_change", i)

        aspect_change = abs(ratio - prev_ratio) / max(prev_ratio, 0.01)
        if aspect_change > bbox_aspect_change_thresh:
            _add_anomaly(anomalies, "bbox_aspect_change", i)

        height_drop = (prev_height - height) / prev_height
        width_growth = (width - prev_width) / prev_width
        center_drop = center_y - prev_center_y
        if height_drop > bbox_height_drop_thresh:
            _add_anomaly(anomalies, "bbox_height_drop", i)

        if (
            ratio > fallen_aspect_ratio_thresh
            and prev_ratio < fallen_aspect_ratio_thresh
            and (height_drop > bbox_height_drop_thresh or width_growth > bbox_aspect_change_thresh)
        ):
            _add_anomaly(anomalies, "fall_transition", i)

        if ratio > fallen_aspect_ratio_thresh:
            _add_anomaly(anomalies, "fallen_aspect_ratio", i)

        if height_drop > bbox_height_drop_thresh * 0.6 and center_drop > prev_height * 0.15:
            _add_anomaly(anomalies, "vertical_collapse", i)

    return anomalies
