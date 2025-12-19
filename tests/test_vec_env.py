import numpy as np
from gymnasium.vector import SyncVectorEnv

from neoskidrl.envs import NeoSkidNavEnv


def _make_env(rank: int):
    def _init():
        env = NeoSkidNavEnv(config_path="config/default.yml", render_mode=None)
        env.reset(seed=100 + rank)
        return env

    return _init


def test_vector_env_headless():
    envs = SyncVectorEnv([_make_env(i) for i in range(4)])
    obs, info = envs.reset()
    assert obs.shape[0] == 4
    action = np.stack([envs.single_action_space.sample() for _ in range(4)], axis=0)
    obs, reward, term, trunc, info = envs.step(action)
    assert obs.shape[0] == 4
    assert reward.shape[0] == 4
    envs.close()
