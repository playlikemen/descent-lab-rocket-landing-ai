# Build status

Created as a local MVP on 2026-10-26.

## Verified in this build
- JavaScript syntax checks pass for `src/core.js` and `src/main.js`.
- 11 core unit tests pass with `node --test tests/core.test.cjs`.
- Python training and evaluation scripts compile with `python -m py_compile`.
- A local Node-based simulation smoke experiment ran 5,000 Q-learning episodes and 100 fixed-seed evaluation episodes through the conservative policy wrapper without runtime errors.

## Not yet verified
- Manual browser/phone UI QA (the available headless Chromium process did not complete in this environment).
- End-to-end PPO training, since the optional Gymnasium/Box2D and Stable-Baselines3 dependencies were not installed for this build.
- A demonstrated Q-learning improvement over the heuristic baseline. The baseline fallback intentionally preserves behavior where the learned table does not provide enough evidence; an evaluation report should be inspected before claiming the model is solved.
- Hosting or GitHub deployment. No destination repository was provided.
