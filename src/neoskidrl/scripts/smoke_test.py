import numpy as np
from neoskidrl.envs import NeoSkidNavEnv

if __name__ == "__main__":
    env = NeoSkidNavEnv(config_path="config/default.yml", render_mode=None)
    obs, info = env.reset()
    print("obs shape:", obs.shape, "goal:", info["goal_xy"], "goal_yaw:", info["goal_yaw"])

    for i in range(200):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        if (i % 20) == 0:
            print(i, "r:", round(r, 3), "dist:", round(info["dist"], 3), "collision:", info["collision"])
        if term or trunc:
            print("done:", info)
            break
    env.close()
