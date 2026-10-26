"""Evaluate a trained PPO model on fixed, separate episodes."""
import argparse, json
from pathlib import Path
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--out", type=Path, default=Path("artifacts/lunarlander/evaluation.json"))
    a = p.parse_args()
    model = PPO.load(str(a.model))
    env = gym.make("LunarLander-v3")
    rewards, lengths = [], []
    try:
        for i in range(a.episodes):
            obs, _ = env.reset(seed=a.seed + i)
            done, total, steps = False, 0.0, 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward); steps += 1
                done = terminated or truncated
            rewards.append(total); lengths.append(steps)
    finally:
        env.close()
    out = {"environment":"LunarLander-v3","episodes":a.episodes,"seed_start":a.seed,
        "mean_reward":float(np.mean(rewards)),"std_reward":float(np.std(rewards)),
        "min_reward":float(np.min(rewards)),"max_reward":float(np.max(rewards)),
        "reward_200_fraction":float(np.mean(np.asarray(rewards) >= 200)),
        "mean_episode_length":float(np.mean(lengths)),
        "note":"Reward >= 200 is a benchmark proxy, not a task-independent physical landing-success label."}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
