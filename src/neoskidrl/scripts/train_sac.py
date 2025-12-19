from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from neoskidrl.envs import NeoSkidNavEnv

def make_env():
    return NeoSkidNavEnv(config_path="config/default.yml", render_mode=None)

if __name__ == "__main__":
    env = DummyVecEnv([make_env])

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="runs/tb",
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
    )

    model.learn(total_timesteps=300_000)
    model.save("runs/skidnav_sac.zip")
