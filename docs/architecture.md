# Architecture

## Browser MVP

The static frontend is intentionally build-free. `src/core.js` exposes a small, dependency-free API for simulation state transitions, reward calculation, state encoding, action selection, Q-learning updates and policy serialization. It uses a deterministic seeded random-number generator for reproducible environment resets and test cases.

`src/main.js` owns DOM access, rendering, UI state and long-running task orchestration. Training is divided into short asynchronous batches so the interface can refresh between batches. Evaluation uses deterministic seed values and does not modify the model. The canvas renderer is a view of the environment state; it does not drive the physics.

### Browser control loop

1. Reset the environment to a new initial state.
2. Pick an action using the selected policy.
3. Apply one environment transition and receive reward/terminal information.
4. During training, update Q(s,a) toward `r + gamma * max Q(s', a')` (or toward `r` on a terminal transition).
5. Record episode reward and outcome; evaluation uses fixed seeds without updating Q-values.

The sparse table uses a packed integer key for discretized six-dimensional observations. Q-values are stored as a small `Float32Array` per visited state. A baseline controller is used for side-by-side experiments and as an explicitly disclosed fallback when the Q-table lacks evidence. The live conservative policy only overrides the baseline when both actions have been sampled and the learned action value exceeds the baseline action value by at least the configured 5-point margin. This is a heuristic guard, not a formal safety guarantee.

## Python benchmark path

`ml/train_lunarlander.py` trains Stable-Baselines3 PPO in Gymnasium's `LunarLander-v3`. It saves periodic checkpoints, evaluation snapshots, TensorBoard logs and a final model. `ml/evaluate_lunarlander.py` evaluates that saved model deterministically and saves complete episode-level measurements.

The Python and browser agents intentionally remain separate. Their environments, observations, action spaces and policy representations differ. A future adapter must explicitly map those interfaces before a Python PPO model can control the custom browser simulator.

## Trustworthy reporting

- Episode reward and safe landing rate are separate measurements.
- Evaluation seeds are separate from browser training episodes.
- A chart displays collected observations rather than hand-authored performance data.
- Training scripts are provided, but no model performance is asserted until a run has been completed and evaluated.
