from neoskidrl.logging.run_logger import RollingWindow


def test_rolling_window_means_and_rates():
    window = RollingWindow(3)
    window.push({"success": True, "final_dist": 1.0, "reward_contrib_sum": {"progress": 2.0}})
    window.push({"success": False, "final_dist": 3.0, "reward_contrib_sum": {"progress": -1.0}})
    window.push({"success": True, "final_dist": 2.0, "reward_contrib_sum": {"progress": 1.0}})

    assert window.count() == 3
    assert window.rate("success") == 2.0 / 3.0
    assert window.mean("final_dist") == 2.0
    contrib_mean = window.dict_mean("reward_contrib_sum")
    assert contrib_mean["progress"] == (2.0 - 1.0 + 1.0) / 3.0


def test_rolling_window_handles_missing():
    window = RollingWindow(2)
    window.push({"success": True})
    window.push({"final_dist": 2.0})
    assert window.rate("success") == 1.0
    assert window.mean("final_dist") == 2.0
