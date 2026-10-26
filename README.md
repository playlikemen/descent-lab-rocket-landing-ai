# Descent Lab — Rocket Landing AI

A local-first, interactive reinforcement-learning laboratory for a simplified 2D rocket landing task. The project includes a zero-build browser application with a tabular Q-learning agent, a heuristic baseline, an episode evaluator, reward charts, policy import/export, automated core tests, and a separate PPO pipeline for Gymnasium's LunarLander-v3 benchmark.

> **Scope and safety:** This is an educational simulation, not a flight dynamics package or a flight-certified control system. The browser environment is a simplified model created for this project. The Python benchmark uses Gymnasium's established LunarLander environment. Training outcomes are not pre-guaranteed; run evaluation and use recorded results when reporting performance.

## Get started in under a minute

**Requirements:** Python 3 for serving the static files. No JavaScript build step or npm install is required.

```bash
./run_local.sh
```

Open [http://localhost:8000](http://localhost:8000). Alternatively, run `python3 -m http.server 8000` from the repository root.

### First experiment

1. Launch **Heuristic baseline** to see a flight with a simple rules-based controller.
2. Choose a training batch (start with 100 episodes) and press **Train agent**.
3. The local tabular Q-learning agent updates action values from simulated rewards. The reward chart tracks the observed episode returns.
4. Choose **Learned Q-policy**, launch a fresh episode, and compare the landing report with the baseline.
5. Press **Evaluate 30** to test the selected policy on a fixed set of seeds. Evaluation does not update the Q-table.
6. Export the Q-policy as JSON to preserve it, or import a previously exported policy.

The **Learned Q-policy** is conservatively gated: it falls back to the heuristic baseline for states/actions without adequate evidence and only overrides the baseline when a sampled alternative's estimated Q-value is more than 5 reward points higher. This reduces regressions while the table is sparse, but it is not a formal safety guarantee. Unseen state bins should not be mistaken for a fully learned policy. The agent and data are held in memory for this browser session; export a policy before refreshing if you want to keep it.

## What is included

| Path | Purpose |
|---|---|
| `index.html` | Responsive dashboard and canvas simulator |
| `src/core.js` | Deterministic physics, state discretization, Q-learning table and baseline controller |
| `src/main.js` | UI interactions, training batches, evaluation, rendering and policy import/export |
| `src/styles.css` | Responsive dark mission-control design |
| `ml/train_lunarlander.py` | Train PPO on Gymnasium LunarLander-v3, save checkpoints and best evaluated model |
| `ml/evaluate_lunarlander.py` | Deterministic multi-episode model evaluation with JSON results |
| `tests/core.test.cjs` | Physics, determinism, Q-learning and policy serialization tests |
| `docs/architecture.md` | System boundaries and design decisions |
| `docs/roadmap.md` | Milestones and completion criteria |

## Run the tests

Node.js 20+ is recommended; no packages are required for the browser core tests.

```bash
node --test tests/core.test.cjs
node --check src/core.js
node --check src/main.js
```

## Optional: train PPO on Gymnasium LunarLander-v3

Use a dedicated Python 3.11 or 3.12 virtual environment for the Box2D-based environment and install the machine-learning dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r ml/requirements.txt
python ml/train_lunarlander.py --timesteps 500000 --seed 42
```

To evaluate a produced model over 100 episodes:

```bash
python ml/evaluate_lunarlander.py --model artifacts/lunarlander/<run-name>/final_model.zip --episodes 100 --seed 12345
```

The scripts save the final model, periodic checkpoints, the best evaluated checkpoint (when produced), TensorBoard logs, and a JSON evaluation report. Open training curves with:

```bash
tensorboard --logdir artifacts/lunarlander
```

The Python PPO model is a **separate benchmark path** and is not automatically loaded into the browser's tabular Q-policy. Bridging those systems requires an explicit observation/action adapter, model inference endpoint or compatible browser inference export. Do not conflate training results from the two environments.

## Important model details

The browser environment's observation state is discretized from six values: horizontal offset, altitude, horizontal velocity, vertical velocity, attitude angle and angular velocity. The action set contains six combinations of main-engine thrust on/off and rotation torque. A sparse Q-table stores values for visited state bins. The reward includes small step costs, potential-based shaping and terminal rewards/penalties for safe landing, hard landing, leaving the flight area or timing out.

A success requires the vehicle to touch down within the configured position, speed, attitude and angular-velocity thresholds. This success definition is for the local toy environment only. Reward improvements do not imply success-rate improvements, so use fixed-seed evaluation and inspect both metrics.

## Known limitations

- The local simulator is simplified and does not model orbital mechanics, aerodynamic loads, slosh, realistic engine dynamics or certified guidance controls.
- Tabular Q-learning has limited generalization between discretized states; training quality depends on parameters and coverage.
- The heuristic fallback for unseen states makes early learned-policy flights usable but means the policy is hybrid until coverage improves.
- Browser state is session-scoped unless a Q-table is exported.
- The PPO benchmark is a separate environment and has not been trained just because this code exists.
- Performance and learning curves depend on the number of training episodes and runtime; no particular score is promised.

## License

MIT — see `LICENSE`.
