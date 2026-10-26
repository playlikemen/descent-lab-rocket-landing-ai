# Descent Lab — Rocket Landing AI

Interactive browser lab for learning rocket landings with tabular Q-learning, plus an optional Python PPO benchmark using Gymnasium LunarLander-v3.

## Run the browser demo

No build step or npm packages are required. Open `index.html`, or run:

```bash
python3 -m http.server 8000
```

Visit http://localhost:8000.

## Features

- Canvas-based 2D rocket simulator and live telemetry.
- Manual, heuristic baseline, and learned Q-policy pilot modes.
- Train batches of episodes in-browser; policy and history persist in local storage.
- Reward curve, recent success/failure rates and greedy evaluation.
- Export the learned policy to JSON.
- Separate Stable-Baselines3 PPO scripts for the Gymnasium LunarLander-v3 task.
- GitHub Pages workflow for the static demo.

## Train the optional PPO benchmark

Use Python 3.10+; Box2D dependencies may differ by OS.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt
python ml/train_lunarlander.py --timesteps 300000
python ml/evaluate_lunarlander.py --model artifacts/lunarlander/best/best_model.zip --episodes 100
```

The scripts do not include a pretrained model; training and evaluation must be run locally. Results are saved under `artifacts/`.

## Important interpretation

The browser toy environment and Gymnasium LunarLander are different tasks. Do not compare their raw rewards. Report sample size, mean reward, and success/failure rates. A reward-200 benchmark proxy is not equivalent to a verified real-world landing label.

This is an educational demo, not real rocket guidance software or a validated flight-dynamics model. See [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).
