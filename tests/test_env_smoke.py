import numpy as np

from neoskidrl.envs import NeoSkidNavEnv


def test_env_reset_step_smoke():
    env = NeoSkidNavEnv(config_path="config/default.yml", render_mode=None)
    obs, info = env.reset(seed=123)
    assert obs.shape[0] == env.rays + 5
    assert "goal_xy" in info
    assert "base_xy" in info

    for _ in range(8):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        assert np.isfinite(reward)
        assert obs.shape[0] == env.rays + 5
        assert info["base_xy"].shape == (2,)
        if term or trunc:
            break

    env.close()
