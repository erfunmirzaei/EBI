from typing import NamedTuple
import numpy as np

class Trajectory(NamedTuple):
    ang: np.ndarray
    dist: np.ndarray
    pos: np.ndarray