const test = require('node:test');
const assert = require('node:assert/strict');
const RL = require('../src/core.js');

test('seeded environments reset deterministically', () => {
  const a = RL.createEnvironment(123).reset();
  const b = RL.createEnvironment(123).reset();
  assert.deepEqual(a, b);
});

test('reset state starts above the landing surface', () => {
  const s = RL.createEnvironment(8).reset();
  assert.ok(s.y >= 80 && s.y <= 95);
  assert.equal(s.done, false);
});

test('coasting decreases vertical velocity under gravity', () => {
  const env = RL.createEnvironment(1);
  const s = env.reset({ x: 0, y: 60, vx: 0, vy: 0, angle: 0, omega: 0 });
  const result = env.step(s, 1);
  assert.ok(result.state.vy < 0);
  assert.ok(result.state.y < s.y);
});

test('straight main thrust counters gravity and raises vertical velocity', () => {
  const env = RL.createEnvironment(1);
  const s = env.reset({ x: 0, y: 60, vx: 0, vy: 0, angle: 0, omega: 0 });
  const result = env.step(s, 4);
  assert.ok(result.state.vy > 0);
  assert.equal(result.state.lastThrust, true);
});

test('state encoder returns deterministic integer bins', () => {
  const s = RL.createEnvironment(12).reset();
  assert.equal(RL.encodeState(s), RL.encodeState({ ...s }));
  assert.ok(Number.isInteger(RL.encodeState(s)));
});

test('Q update creates state values and changes selected action', () => {
  const agent = new RL.QAgent({ seed: 1 });
  const s = RL.createEnvironment(5).reset();
  const next = { ...s, y: s.y - 40 };
  agent.update(s, 4, 10, next, false);
  const q = agent.values(s, false);
  assert.equal(agent.q.size, 2);
  assert.ok(q[4] > 0);
});

test('Q-table exports and imports without changing values', () => {
  const source = new RL.QAgent({ seed: 3 });
  const s = RL.createEnvironment(7).reset();
  source.update(s, 3, 4, s, true);
  source.episodes = 19;
  const data = JSON.parse(JSON.stringify(source.exportJSON()));
  const target = new RL.QAgent();
  assert.equal(target.importJSON(data), source.q.size);
  assert.equal(target.episodes, 19);
  assert.deepEqual(Array.from(target.values(s, false)), Array.from(source.values(s, false)));
});

test('unsafe touchdown terminates as a crash', () => {
  const env = RL.createEnvironment(1);
  const s = env.reset({ x: 0, y: 0.01, vx: 0, vy: -4, angle: 0, omega: 0 });
  const result = env.step(s, 1);
  assert.equal(result.done, true);
  assert.equal(result.landed, false);
  assert.equal(result.info.outcome, 'crashed');
});

test('heuristic returns a valid discrete action', () => {
  const s = RL.createEnvironment(8).reset();
  assert.ok(RL.ACTIONS.some((_, index) => index === RL.heuristicAction(s)));
});


test('conservative Q policy falls back when there is no Q evidence', () => {
  const agent = new RL.QAgent({ seed: 9 });
  const s = RL.createEnvironment(31).reset();
  assert.equal(RL.conservativeQAction(s, agent), RL.heuristicAction(s));
});

test('conservative Q policy never selects an unvisited Q action', () => {
  const agent = new RL.QAgent({ seed: 9 });
  const s = RL.createEnvironment(31).reset();
  const baseline = RL.heuristicAction(s);
  agent.update(s, baseline, 1, { ...s, y: s.y - 5 }, false);
  assert.equal(RL.conservativeQAction(s, agent), baseline);
});
