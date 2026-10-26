(function (root) {
  'use strict';
  const ACTIONS = [
    { name: 'COAST / ROTATE LEFT', thrust: false, torque: -1 },
    { name: 'COAST', thrust: false, torque: 0 },
    { name: 'COAST / ROTATE RIGHT', thrust: false, torque: 1 },
    { name: 'THRUST / ROTATE LEFT', thrust: true, torque: -1 },
    { name: 'MAIN THRUST', thrust: true, torque: 0 },
    { name: 'THRUST / ROTATE RIGHT', thrust: true, torque: 1 }
  ];
  const CONFIG = Object.freeze({ gravity: 2.15, thrust: 5.25, torque: 3.4, dt: 0.1, maxSteps: 700, xLimit: 47, safeX: 4.5, safeVx: 1.5, safeVy: 2.2, safeAngle: 0.24, safeOmega: 0.6 });
  function clamp(n, lo, hi) { return Math.max(lo, Math.min(hi, n)); }
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function createEnvironment(seed) {
    let random = mulberry32(seed == null ? Math.floor(Math.random() * 0xffffffff) : seed);
    let nextSeed = seed == null ? null : seed;
    function reset(options) {
      options = options || {};
      if (options.seed != null) { random = mulberry32(options.seed); nextSeed = options.seed; }
      const state = {
        x: options.x == null ? (random() * 2 - 1) * 18 : options.x,
        y: options.y == null ? 80 + random() * 15 : options.y,
        vx: options.vx == null ? (random() * 2 - 1) * 1.2 : options.vx,
        vy: options.vy == null ? -random() * 1.1 : options.vy,
        angle: options.angle == null ? (random() * 2 - 1) * 0.16 : options.angle,
        omega: options.omega == null ? (random() * 2 - 1) * 0.08 : options.omega,
        steps: 0, done: false, landed: false, outcome: null, lastThrust: false
      };
      return state;
    }
    function step(state, actionIndex) {
      if (state.done) return { state, reward: 0, done: true, landed: state.landed, info: { outcome: state.outcome } };
      const action = ACTIONS[clamp(Math.floor(actionIndex), 0, ACTIONS.length - 1)];
      const previousPotential = potential(state);
      const s = { ...state };
      const dt = CONFIG.dt;
      s.lastThrust = action.thrust;
      if (action.thrust) {
        s.vx += CONFIG.thrust * Math.sin(s.angle) * dt;
        s.vy += (CONFIG.thrust * Math.cos(s.angle) - CONFIG.gravity) * dt;
      } else {
        s.vy -= CONFIG.gravity * dt;
      }
      s.omega = clamp((s.omega + action.torque * CONFIG.torque * dt) * 0.965, -2.5, 2.5);
      s.angle = clamp(s.angle + s.omega * dt, -0.9, 0.9);
      s.x += s.vx * dt;
      s.y += s.vy * dt;
      s.steps += 1;
      let reward = -0.018 + 1.25 * (potential(s) - previousPotential);
      let done = false;
      let landed = false;
      let outcome = null;
      if (s.y <= 0) {
        s.y = 0;
        const safe = Math.abs(s.x) <= CONFIG.safeX && Math.abs(s.vx) <= CONFIG.safeVx && s.vy >= -CONFIG.safeVy && Math.abs(s.angle) <= CONFIG.safeAngle && Math.abs(s.omega) <= CONFIG.safeOmega;
        done = true; landed = safe; outcome = safe ? 'landed' : 'crashed';
        reward += safe ? 120 - Math.abs(s.x) * 2 - Math.abs(s.vx) * 4 - Math.abs(s.vy) * 4 - Math.abs(s.angle) * 18 - Math.abs(s.omega) * 4 : -100;
      } else if (Math.abs(s.x) > CONFIG.xLimit) {
        done = true; outcome = 'out_of_bounds'; reward -= 100;
      } else if (s.steps >= CONFIG.maxSteps) {
        done = true; outcome = 'timeout'; reward -= 35;
      }
      s.done = done; s.landed = landed; s.outcome = outcome;
      return { state: s, reward, done, landed, info: { outcome, action: action.name } };
    }
    return { reset, step, get seed() { return nextSeed; } };
  }
  function potential(s) {
    const altitude = Math.max(0, s.y);
    const desiredDescent = -Math.min(4.0, 0.52 * Math.sqrt(altitude));
    return -0.030 * Math.abs(s.x) - 0.19 * Math.abs(s.vx) - 0.07 * Math.abs(s.vy - desiredDescent) - 0.72 * Math.abs(s.angle) - 0.13 * Math.abs(s.omega);
  }
  function quantize(v, min, max, buckets) { return clamp(Math.floor((clamp(v, min, max) - min) / (max - min) * buckets), 0, buckets - 1); }
  function encodeState(s) {
    const bins = [
      quantize(s.x, -45, 45, 7), quantize(s.y, 0, 100, 6),
      quantize(s.vx, -6, 6, 5), quantize(s.vy, -8, 8, 5),
      quantize(s.angle, -0.9, 0.9, 5), quantize(s.omega, -2.5, 2.5, 3)
    ];
    let key = 0; const dims = [7, 6, 5, 5, 5, 3];
    for (let i = 0; i < bins.length; i++) key = key * dims[i] + bins[i];
    return key;
  }
  class QAgent {
    constructor(options) {
      options = options || {};
      this.alpha = options.alpha == null ? 0.15 : options.alpha;
      this.gamma = options.gamma == null ? 0.99 : options.gamma;
      this.minEpsilon = options.minEpsilon == null ? 0.08 : options.minEpsilon;
      this.episodes = options.episodes || 0;
      this.q = new Map();
      this.visits = new Map();
      this.rng = mulberry32(options.seed == null ? 74129 : options.seed);
    }
    epsilon() { return Math.max(this.minEpsilon, 1 - this.episodes / 2500); }
    values(state, create) {
      const key = encodeState(state);
      let values = this.q.get(key);
      if (!values && create) { values = new Float32Array(ACTIONS.length); this.q.set(key, values); }
      return values || null;
    }
    visitCounts(state, create) {
      const key = encodeState(state);
      let counts = this.visits.get(key);
      if (!counts && create) { counts = new Uint32Array(ACTIONS.length); this.visits.set(key, counts); }
      return counts || null;
    }
    hasVisited(state) {
      const counts = this.visitCounts(state, false);
      return !!counts && counts.some(count => count > 0);
    }
    choose(state, epsilon, options) {
      options = options || {};
      const key = encodeState(state);
      const values = this.q.get(key);
      const counts = this.visits.get(key);
      const explore = this.rng() < (epsilon || 0);
      const untried = [];
      if (counts) for (let i = 0; i < ACTIONS.length; i++) if (counts[i] === 0) untried.push(i);
      if (explore) {
        if (untried.length && options.exploreUntried !== false) return untried[Math.floor(this.rng() * untried.length)];
        return Math.floor(this.rng() * ACTIONS.length);
      }
      // Use the controller as a prior for unseen bins, then let Q-values take over
      // when the learner has evidence. Exploration still samples actions at epsilon.
      if (!values || !counts) return heuristicAction(state);
      let best = -1; let bestValue = -Infinity;
      for (let i = 0; i < values.length; i++) {
        // Do not treat unvisited actions' initialized zero values as learned evidence.
        if (counts[i] > 0 && values[i] > bestValue) { best = i; bestValue = values[i]; }
      }
      if (best >= 0) return best;
      return heuristicAction(state);
    }
    update(state, action, reward, nextState, done) {
      const key = encodeState(state);
      const values = this.values(state, true);
      const counts = this.visitCounts(state, true);
      counts[action] = Math.min(0xffffffff, counts[action] + 1);
      const nextValues = done ? null : this.values(nextState, true);
      let nextMax = 0;
      if (nextValues) {
        const nextCounts = this.visitCounts(nextState, false);
        let found = false;
        if (nextCounts) for (let i = 0; i < nextValues.length; i++) if (nextCounts[i] > 0 && (!found || nextValues[i] > nextMax)) { nextMax = nextValues[i]; found = true; }
        if (!found) nextMax = 0;
      }
      const target = reward + (done ? 0 : this.gamma * nextMax);
      values[action] += this.alpha * (target - values[action]);
    }
    exportJSON() {
      const entries = []; const visits = [];
      this.q.forEach((values, key) => entries.push([key, Array.from(values)]));
      this.visits.forEach((counts, key) => visits.push([key, Array.from(counts)]));
      return { format: 'descent-lab-qtable-v1', episodes: this.episodes, alpha: this.alpha, gamma: this.gamma, minEpsilon: this.minEpsilon, entries, visits };
    }
    importJSON(data) {
      if (!data || data.format !== 'descent-lab-qtable-v1' || !Array.isArray(data.entries)) throw new Error('This file is not a supported Descent Lab Q-policy.');
      const next = new Map();
      const nextVisits = new Map();
      for (const row of data.entries) {
        if (!Array.isArray(row) || row.length !== 2 || !Number.isInteger(row[0]) || !Array.isArray(row[1]) || row[1].length !== ACTIONS.length || row[1].some(v => !Number.isFinite(v))) throw new Error('The policy file contains an invalid Q-table entry.');
        next.set(row[0], Float32Array.from(row[1]));
      }
      if (data.visits != null) {
        if (!Array.isArray(data.visits)) throw new Error('The policy visit-count table is invalid.');
        for (const row of data.visits) {
          if (!Array.isArray(row) || row.length !== 2 || !Number.isInteger(row[0]) || !Array.isArray(row[1]) || row[1].length !== ACTIONS.length || row[1].some(v => !Number.isSafeInteger(v) || v < 0)) throw new Error('The policy contains an invalid action visit count.');
          nextVisits.set(row[0], Uint32Array.from(row[1]));
        }
      }
      this.q = next; this.visits = nextVisits; this.episodes = Math.max(0, Number(data.episodes) || 0);
      this.alpha = Number(data.alpha) || 0.15; this.gamma = Number(data.gamma) || 0.99; this.minEpsilon = Number(data.minEpsilon) || 0.08;
      return this.q.size;
    }
  }
  function heuristicAction(s) {
    const desiredAngle = clamp(-0.075 * s.x - 0.19 * s.vx, -0.52, 0.52);
    const angleError = desiredAngle - s.angle;
    const desiredOmega = clamp(angleError * 1.8, -0.8, 0.8);
    const torque = s.omega < desiredOmega - 0.08 ? 1 : s.omega > desiredOmega + 0.08 ? -1 : 0;
    const desiredVy = -Math.min(4.0, 0.52 * Math.sqrt(Math.max(0, s.y)));
    const thrust = s.vy < desiredVy - 0.25;
    if (thrust) return torque < 0 ? 3 : torque > 0 ? 5 : 4;
    return torque < 0 ? 0 : torque > 0 ? 2 : 1;
  }
  function randomAction(random) { return Math.floor((random || Math.random)() * ACTIONS.length); }

  function conservativeQAction(state, agent, minAdvantage) {
    const advantage = minAdvantage == null ? 5 : minAdvantage;
    const fallback = heuristicAction(state);
    const values = agent.values(state, false);
    const counts = agent.visitCounts(state, false);
    // Avoid overriding the known baseline using a Q-value with no matching experience.
    if (!values || !counts || counts[fallback] === 0) return fallback;
    const candidate = agent.choose(state, 0, { exploreUntried: false });
    if (candidate !== fallback && counts[candidate] > 0 && values[candidate] > values[fallback] + advantage) return candidate;
    return fallback;
  }
  const api = { ACTIONS, CONFIG, clamp, createEnvironment, encodeState, QAgent, heuristicAction, conservativeQAction, randomAction, mulberry32 };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.RocketRL = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
