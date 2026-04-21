from dataclasses import dataclass, field

import numpy as np

@dataclass
class Person:
    points: list[tuple[int, int]] = field(default_factory=list)
    l_points: list[tuple[int, int]] = field(default_factory=list)
    r_points: list[tuple[int, int]] = field(default_factory=list)

    num_disappearances: int = 0
    num_dangers_frames: int = 0

    crash: bool = False


@dataclass
class Car:
    box: tuple[int, int, int, int]
    points: list[tuple[int, int]] = field(default_factory=list)

    num_disappearances: int = 0
    has_mov: bool = False
