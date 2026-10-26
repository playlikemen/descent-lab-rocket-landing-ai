# Descent Lab — Rocket Landing AI

<div align="center">

**A local-first reinforcement-learning lab for learning to land a virtual rocket.**

Train a tabular Q-learning agent, watch the flight, compare policies, and evaluate landing outcomes—all from a browser.

[Open the repository](https://github.com/playlikemen/descent-lab-rocket-landing-ai) · [Run locally](#quick-start) · [Report a bug](https://github.com/playlikemen/descent-lab-rocket-landing-ai/issues)

</div>

---

## What is Descent Lab?

Descent Lab is an interactive, educational sandbox for exploring the reinforcement-learning loop: observe a state, choose an action, receive a reward, and update a policy. It pairs a browser-based 2D landing environment with a separate Python benchmark using Gymnasium's LunarLander-v3.

The project is deliberately transparent about what is simulated. The browser environment is a simplified teaching model; it is **not** a physically validated rocket flight simulator or hardware-control system.

## Features

### Browser flight deck
- Animated Canvas simulation with a plotted trajectory, landing zone, and live flight state.
- Real-time telemetry for horizontal/vertical velocity, attitude, angular velocity, and engine state.
- Launch, pause/resume, reset, and post-flight outcome reporting.
- Responsive dashboard, accessible status messages, and keyboard-friendly controls.

### Reinforcement learning
- Tabular Q-learning with configurable episode batches and a visible exploration schedule.
- Three policies to compare: heuristic baseline, random pilot, and a conservative learned Q-policy.
- Conservative policy gating: the learned controller avoids overriding the baseline when relevant state-action experience is missing or the estimated reward advantage is too small.
- Episode reward chart with a moving-average trend.
- Export and import of the learned policy as JSON.

### Evaluation and reproducibility
- A separate fixed-seed evaluation batch with no Q-table updates.
- Landing success rate, mean reward, and touchdown report.
- Python scripts to train and evaluate PPO on Gymnasium LunarLander-v3.
- GitHub Actions checks for the JavaScript sources, core tests, and Python syntax.

## Quick start

Requirements for the browser app: a modern browser. To run the project locally, use Python 3 or Node.js.

### Option A — Python static server

```bash
git clone https://github.com/playlikemen/descent-lab-rocket-landing-ai.git
cd descent-lab-rocket-landing-ai
python3 -m http.server 8000
```

Open [http://localhost:8000](http://localhost:8000).

### Option B — npm scripts

Node.js 20 or newer is recommended.

```bash
npm run serve
```

### Train your first policy

1. Open the **Training lab** panel.
2. Start with 200 episodes to get a quick baseline.
3. Watch the reward curve and the visited-state count update.
4. Select **Learned Q-policy (conservative)** under Pilot selection.
5. Launch a flight, then use **Evaluate policy** for a fixed-seed batch.
6. Export the policy JSON to keep a portable copy.

The best-looking reward curve does not automatically mean the agent is landing reliably. Use the evaluation success rate and touchdown metrics as well.

## Python PPO benchmark (optional)

The Python benchmark is a separate environment and algorithm path from the custom browser demo. It requires Python 3.10+; Box2D setup can vary by operating system.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt

python ml/train_lunarlander.py --timesteps 300000
python ml/evaluate_lunarlander.py \
  --model artifacts/lunarlander/best/best_model.zip \
  --episodes 100
```

On Windows, activate the virtual environment with `.venv\\Scripts\\activate`.

Training can take time. No pretrained PPO checkpoint is included. The evaluation script writes a JSON summary under `artifacts/lunarlander/`; inspect the saved output before making performance claims.

## How the browser agent works

The browser environment exposes a compact state made from horizontal position, altitude, horizontal velocity, vertical velocity, angle, and angular velocity. Continuous observations are discretized into bins so the policy can store action values in a table.

The action space includes coast, rotate, main thrust, and combined thrust/rotation actions. The update follows the standard Q-learning form:

```text
Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') − Q(s, a)]
```

The learned controller is intentionally conservative: it uses the heuristic baseline when the table has insufficient evidence for a safer override. This can make the first few learned flights look similar to the baseline; it is not evidence that a pretrained model exists.

## Project structure

```text
.
├── index.html                  # App shell and page layout
├── src/
│   ├── styles.css              # Responsive interface styling
│   ├── core.js                 # Physics environment, state encoding, Q-agent
│   └── main.js                 # Canvas rendering, controls, training/evaluation
├── tests/
│   └── core.test.cjs           # Core simulation and agent tests
├── ml/
│   ├── train_lunarlander.py    # PPO training entry point
│   ├── evaluate_lunarlander.py # Fixed-seed evaluation entry point
│   └── requirements.txt
├── docs/
│   └── EXPERIMENT_PROTOCOL.md  # Reproducible experiment guidance
└── .github/workflows/
    └── pages.yml               # Validation and optional Pages deployment
```

## Development commands

```bash
npm run check  # Syntax-check JavaScript modules and run core tests
npm test       # Run the Node.js test suite
npm run serve  # Serve the website at localhost:8000
```

The browser application has no frontend package-install/build step. The npm scripts are convenience commands for serving and testing.

## Deployment with GitHub Pages

The repository workflow validates the app and packages the static site on each push to `main`. The deployment job is deliberately gated because a workflow token cannot necessarily turn on Pages for an organization-owned repository.

To publish:
1. In repository **Settings → Pages**, configure **GitHub Actions** as the build and deployment source.
2. In **Settings → Secrets and variables → Actions → Variables**, create `PAGES_ENABLED` with value `true`.
3. Push a commit or rerun the workflow from the Actions tab.

If the organization restricts the Pages API, an owner/admin may need to perform the Pages setup in repository settings before the deployment job can publish.

## Evaluation guidelines

For honest comparisons, record the algorithm, environment, seed, episode count, mean reward, success rate, and failure reasons. Evaluate against the same fixed seeds when comparing policies. Repeat with additional seeds to understand variability.

The browser environment and LunarLander-v3 are different tasks; **their raw rewards are not directly comparable**. A reward threshold such as 200 on LunarLander is a benchmark convention, not a universal or physically verified landing-success label.

See [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) for the experiment checklist.

## Limitations and safety

- The browser environment uses intentionally simplified 2D dynamics.
- Tabular state discretization trades precision for simplicity and may limit generalization.
- Training happens locally in the browser and uses CPU time; progress is lost if not exported when leaving the session, depending on browser behavior.
- The PPO pipeline is optional and separate; no trained PPO model is bundled.
- This project is for education and experimentation only. Do not use it to control real rockets, aircraft, or other physical systems.

## Contributing

Bug reports, usability improvements, tests, and reproducible experiments are welcome. Please include steps to reproduce issues and clearly distinguish measured results from hypotheses.

## License

MIT. See [LICENSE](LICENSE).
