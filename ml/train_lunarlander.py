"""Train PPO on Gymnasium LunarLander-v3 and save reproducible artifacts.

Run from the repository root:
    python -m pip install -r ml/requirements.txt
    python ml/train_lunarlander.py --timesteps 500000
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def make_env(seed: int, render_mode: str | None = None):
    def _init():
        env = gym.make("LunarLander-v3", render_mode=render_mode)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total environment steps (default: 500000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for training and evaluation.")
    parser.add_argument("--run-name", default=None, help="Optional artifact subfolder name.")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel vectorized environments.")
    parser.add_argument("--eval-freq", type=int, default=25_000, help="Evaluation interval in environment steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps <= 0 or args.n_envs <= 0 or args.eval_freq <= 0:
        raise SystemExit("timesteps, n-envs and eval-freq must all be positive integers")

    run_name = args.run_name or datetime.now(timezone.utc).strftime("ppo-%Y%m%d-%H%M%S")
    run_dir = ARTIFACTS / "lunarlander" / run_name
    checkpoints = run_dir / "checkpoints"
    best_dir = run_dir / "best_model"
    logs_dir = run_dir / "tensorboard"
    for path in (checkpoints, best_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)
    train_env = VecMonitor(DummyVecEnv([make_env(args.seed + i) for i in range(args.n_envs)]))
    eval_env = DummyVecEnv([make_env(args.seed + 10_000)])
    # Callback frequencies are counted in vectorized calls, so divide by n_envs
    # to keep the intended cadence approximately aligned with environment steps.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.eval_freq // args.n_envs, 1),
        save_path=str(checkpoints),
        name_prefix="ppo-lunarlander",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(run_dir / "evaluation"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(logs_dir),
        device="auto",
    )
    metadata = {
        "environment": "Gymnasium LunarLander-v3",
        "algorithm": "Stable-Baselines3 PPO",
        "requested_timesteps": args.timesteps,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "note": "Training configuration, not evidence of a solved environment. Evaluate the saved model separately.",
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        model.learn(total_timesteps=args.timesteps, callback=[checkpoint_callback, eval_callback], progress_bar=False, tb_log_name=run_name)
        final_path = run_dir / "final_model"
        model.save(str(final_path))
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["model_path"] = str(final_path.with_suffix(".zip").relative_to(ROOT))
        (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"\nTraining run saved to: {run_dir}")
        print(f"Final model: {final_path}.zip")
        print(f"Best evaluated model (if available): {best_dir / 'best_model.zip'}")
        print("Next: python ml/evaluate_lunarlander.py --model " + str(final_path) + ".zip")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
