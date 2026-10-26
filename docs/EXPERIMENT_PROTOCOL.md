# Experiment protocol

## Browser Q-learning task
1. Record browser version, commit SHA, number of episodes and starting exploration value.
2. Train in batches; retain reward history and the 50-episode summary.
3. Use greedy policy actions for evaluation, with no Q-table updates.
4. Report evaluation sample size, mean return, safe-landing rate and failure rate.
5. Repeat with fresh scenarios before making performance claims.

## Gymnasium PPO benchmark
- Task: `LunarLander-v3`.
- Record seed, package versions, hyperparameters, total timesteps and model checkpoint.
- Use a separate environment with deterministic policy actions for evaluation.
- For performance claims, evaluate at least 100 seeded episodes and report mean/std reward and episode length.
- Reward >= 200 is a common benchmark proxy; it is not equivalent to a universal binary landing-success label.

The browser toy and the Gymnasium benchmark are different tasks. Do not compare raw scores across them. This project is educational, not flight-qualified software.
