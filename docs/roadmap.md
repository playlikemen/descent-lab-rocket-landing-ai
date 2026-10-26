# Roadmap and acceptance checklist

## Milestone 1 — Runnable simulator
- [x] Static browser dashboard.
- [x] Simplified rocket dynamics and collision/termination states.
- [x] Live telemetry and landing report.
- [x] Heuristic and random comparison controllers.
- [x] Unit tests for core environment operations.

## Milestone 2 — Reinforcement learning
- [x] Sparse tabular Q-learning implementation.
- [x] Exploration schedule, learning rate and discount factor.
- [x] Reward history and local training batches.
- [x] Import/export for the Q-table.
- [x] Fixed-seed evaluation without policy updates.
- [ ] Establish a repeatable benchmark showing improvement over a heuristic baseline.
- [ ] Tune reward shaping and discretization with documented ablations.

## Milestone 3 — Standard benchmark
- [x] PPO training script for Gymnasium LunarLander-v3.
- [x] Periodic checkpoints and best-model evaluation callback.
- [x] Deterministic multi-episode evaluation script.
- [ ] Install optional dependencies and execute an end-to-end training run.
- [ ] Attach evaluation metrics produced by an actual run.

## Milestone 4 — Production polish
- [ ] Persist browser models intentionally using user-selected downloads or IndexedDB.
- [ ] Add training cancellation between short, safe checkpoints.
- [ ] Add train-vs-baseline comparison table with confidence intervals.
- [ ] Add configurable wind/disturbance experiments and out-of-distribution evaluation.
- [ ] Add keyboard accessibility review and manual mobile-browser QA.
- [ ] Add deployment workflow after selecting a hosting target.

Checkboxes describe the repository implementation as supplied, not independent QA results. Benchmark and optional-training items remain unchecked until actually executed.
