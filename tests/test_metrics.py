import numpy as np

from neoskidrl.metrics import action_delta_l1, path_length_increment
from neoskidrl.scripts.read_metrics import _format_float


def test_action_delta_l1():
    a0 = np.array([0.0, 0.5], dtype=np.float32)
    a1 = np.array([1.0, -0.5], dtype=np.float32)
    assert action_delta_l1(None, a1) == 0.0
    assert np.isclose(action_delta_l1(a0, a1), 2.0)


def test_path_length_increment():
    p0 = np.array([0.0, 0.0], dtype=np.float32)
    p1 = np.array([3.0, 4.0], dtype=np.float32)
    assert path_length_increment(None, p1) == 0.0
    assert np.isclose(path_length_increment(p0, p1), 5.0)


def test_format_float():
    assert _format_float(1.23456, width=8, prec=2).strip() == "1.23"
