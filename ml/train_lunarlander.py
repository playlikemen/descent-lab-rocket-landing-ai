"""Train PPO on Gymnasium LunarLander-v3 with periodic evaluation."""
import argparse
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("artifacts/lunarlander"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([lambda: Monitor(gym.make("LunarLander-v3"))])
    ev = DummyVecEnv([lambda: Monitor(gym.make("LunarLander-v3"))])
    cb = EvalCallback(ev, best_model_save_path=str(a.out / "best"),
        log_path=str(a.out / "logs"), eval_freq=10_000, n_eval_episodes=10,
        deterministic=True, render=False)
    model = PPO("MlpPolicy", env, verbose=1, seed=a.seed,
        tensorboard_log=str(a.out / "logs"), n_steps=1024, batch_size=64)
    model.learn(total_timesteps=a.timesteps, callback=cb)
    model.save(a.out / "final_model")
    env.close()
    ev.close()

if __name__ == "__main__":
    main()
