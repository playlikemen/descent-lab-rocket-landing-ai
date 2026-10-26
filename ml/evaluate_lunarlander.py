"""Evaluate a saved PPO model on LunarLander-v3 using deterministic actions."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a Stable-Baselines3 PPO .zip model.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--render", action="store_true", help="Render evaluation episodes (slow).")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        parser.error(f"Model does not exist: {model_path}")

    model = PPO.load(str(model_path), device="auto")
    rewards: list[float] = []
    lengths: list[int] = []
    terminated_count = 0
    truncated_count = 0
    env = gym.make("LunarLander-v3", render_mode="human" if args.render else None)
    try:
        for i in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + i)
            total_reward = 0.0
            steps = 0
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                steps += 1
                if terminated or truncated:
                    terminated_count += int(terminated)
                    truncated_count += int(truncated)
                    break
            rewards.append(total_reward)
            lengths.append(steps)
    finally:
        env.close()

    result = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "environment": "Gymnasium LunarLander-v3",
        "episodes": args.episodes,
        "seed_start": args.seed,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "episodes_with_reward_at_least_200": int(sum(r >= 200 for r in rewards)),
        "terminated_episodes": terminated_count,
        "truncated_episodes": truncated_count,
        "episode_rewards": rewards,
        "episode_lengths": lengths,
        "note": "Reward >= 200 is reported as a benchmark threshold, not a guaranteed safe-landing percentage.",
    }
    print(json.dumps({k: v for k, v in result.items() if k not in {"episode_rewards", "episode_lengths"}}, indent=2))
    output = Path(args.output).expanduser() if args.output else model_path.parent / "evaluation-results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Full episode results saved to: {output}")


if __name__ == "__main__":
    main()
