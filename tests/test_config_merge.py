from neoskidrl.config import merge_config


def test_merge_config_deep():
    base = {"a": 1, "b": {"c": 2, "d": 3}, "e": [1, 2]}
    override = {"b": {"c": 99}, "e": [9]}
    merged = merge_config(base, override)
    assert merged["a"] == 1
    assert merged["b"]["c"] == 99
    assert merged["b"]["d"] == 3
    assert merged["e"] == [9]
