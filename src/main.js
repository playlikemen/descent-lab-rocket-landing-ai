(function () {
  'use strict';
  const RL = window.RocketRL;
  const $ = (id) => document.getElementById(id);
  const canvas = $('sim-canvas');
  const ctx = canvas.getContext('2d');
  const chart = $('reward-chart');
  const chartCtx = chart.getContext('2d');
  const agent = new RL.QAgent({ seed: 20261026 });
  let env = RL.createEnvironment();
  let state = env.reset();
  let mode = 'baseline';
  let running = false;
  let jobBusy = false;
  let trainingStopRequested = false;
  let trainProgress = 0;
  let trainTotal = 0;
  let episodeHistory = [];
  let evalInfo = null;
  let lastReport = null;
  let lastAction = 1;
  let episodeReward = 0;
  let episodeStart = 0;
  let simTime = 0;
  let lastTick = 0;
  let tickAccumulator = 0;
  let flightPath = [];
  let stars = [];
  let randomGen = RL.mulberry32(19671);
  const SKY = { width: 1000, height: 500 };

  function notice(message, isError) {
    $('action-notice').textContent = message;
    $('action-notice').style.color = isError ? 'var(--red)' : '';
  }
  function formatSigned(n, digits) { return (n >= 0 ? '+' : '−') + Math.abs(n).toFixed(digits); }
  function policyAction(s, chosenMode) {
    if (chosenMode === 'random') return RL.randomAction(randomGen);
    if (chosenMode === 'baseline') return RL.heuristicAction(s);
    // Conservative policy: only override the baseline when the sampled Q-value has a clear advantage.
    return RL.conservativeQAction(s, agent, 5);
  }
  function modeName() {
    if (mode === 'baseline') return 'Heuristic baseline';
    if (mode === 'random') return 'Random pilot';
    return agent.q.size ? 'Conservative Q-policy + baseline gate' : 'Q-policy (baseline fallback)';
  }
  function setMode(value) {
    mode = value;
    $('control-mode').textContent = modeName();
    $('policy-hint').textContent = mode === 'learned'
      ? (agent.q.size ? 'Q actions override the baseline only when sampled Q-values show a +5 reward advantage.' : 'No learned table yet; the heuristic baseline remains in control while training gathers evidence.')
      : mode === 'baseline' ? 'Rules-based autopilot for a baseline comparison.' : 'Random actions provide a deliberately weak comparison policy.';
    $('launch-btn').innerHTML = '▶ <span>' + (mode === 'learned' ? 'Launch AI' : mode === 'baseline' ? 'Launch baseline' : 'Launch random') + '</span>';
  }
  function resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    SKY.width = rect.width; SKY.height = rect.height;
    stars = Array.from({ length: Math.max(55, Math.floor(rect.width * rect.height / 4600)) }, () => ({ x: Math.random() * rect.width, y: Math.random() * rect.height * 0.79, r: Math.random() * 1.15 + 0.25, a: Math.random() * .45 + .13 }));
    draw();
  }
  function roundRect(context, x, y, w, h, r) {
    context.beginPath(); context.roundRect(x, y, w, h, r);
  }
  function draw() {
    const w = SKY.width, h = SKY.height;
    if (!w || !h) return;
    const groundY = h - 48;
    const altScale = (h - 102) / 110;
    const xScale = w * .42 / 48;
    const px = x => w / 2 + x * xScale;
    const py = y => groundY - y * altScale;
    const bg = ctx.createLinearGradient(0, 0, 0, h);
    bg.addColorStop(0, '#0a1527'); bg.addColorStop(.64, '#122238'); bg.addColorStop(1, '#1a2a3a');
    ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h);
    stars.forEach((star, i) => {
      ctx.globalAlpha = star.a * (.7 + Math.sin(performance.now() / 1700 + i) * .2);
      ctx.fillStyle = i % 8 === 0 ? '#9df4d8' : '#cad7ea';
      ctx.beginPath(); ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2); ctx.fill();
    }); ctx.globalAlpha = 1;
    // Horizon glow and plotted descent guides.
    const glow = ctx.createRadialGradient(w * .5, groundY - 6, 8, w * .5, groundY - 6, w * .55);
    glow.addColorStop(0, '#1c514b35'); glow.addColorStop(1, '#13233700');
    ctx.fillStyle = glow; ctx.fillRect(0, 0, w, h);
    ctx.save(); ctx.strokeStyle = '#91a7c010'; ctx.lineWidth = 1; ctx.setLineDash([3, 7]);
    for (let y = groundY - 60; y > 35; y -= 55) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
    for (let x = w / 2 - 240; x < w; x += 80) { ctx.beginPath(); ctx.moveTo(x, 24); ctx.lineTo(x, groundY); ctx.stroke(); }
    ctx.restore();
    // Terrain
    ctx.beginPath(); ctx.moveTo(0, groundY + 2);
    for (let x = 0; x <= w; x += 14) ctx.lineTo(x, groundY + 2 + Math.sin(x * .043) * 3 + Math.sin(x * .011) * 2);
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
    const terrain = ctx.createLinearGradient(0, groundY, 0, h); terrain.addColorStop(0, '#26364a'); terrain.addColorStop(1, '#101a2b'); ctx.fillStyle = terrain; ctx.fill();
    ctx.strokeStyle = '#52667c'; ctx.lineWidth = 1.1; ctx.beginPath(); ctx.moveTo(0, groundY + 1); ctx.lineTo(w, groundY + 1); ctx.stroke();
    // Landing zone at x=0
    const padX = px(0), padW = Math.min(98, w * .15);
    ctx.fillStyle = '#f2bd79'; ctx.globalAlpha = .09; ctx.fillRect(padX - padW / 2, groundY - 7, padW, 22); ctx.globalAlpha = 1;
    ctx.strokeStyle = '#e9b674'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(padX - padW / 2, groundY - 4); ctx.lineTo(padX + padW / 2, groundY - 4); ctx.stroke();
    ctx.strokeStyle = '#d6a869'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(padX - padW / 2 + 8, groundY - 4); ctx.lineTo(padX - padW / 2 + 8, groundY + 3); ctx.moveTo(padX + padW / 2 - 8, groundY - 4); ctx.lineTo(padX + padW / 2 - 8, groundY + 3); ctx.stroke();
    ctx.fillStyle = '#cfac79'; ctx.font = '9px "DM Mono", monospace'; ctx.textAlign = 'center'; ctx.fillText('LANDING ZONE', padX, groundY + 17);
    // Flight trail
    if (flightPath.length > 1) {
      ctx.save(); ctx.beginPath(); flightPath.forEach((p, i) => { const x = px(p.x), y = py(p.y); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      ctx.strokeStyle = '#7df0c477'; ctx.lineWidth = 1.3; ctx.setLineDash([3, 5]); ctx.stroke(); ctx.restore();
    }
    // Rocket, rendered in world coordinates.
    if (state && state.y > -1) {
      const rx = px(state.x), ry = py(Math.max(0, state.y));
      ctx.save(); ctx.translate(rx, ry); ctx.rotate(-state.angle); // screen y-axis points down
      // engine flame
      if (state.lastThrust && running) {
        const flame = 11 + Math.random() * 7;
        ctx.globalAlpha = .9; ctx.fillStyle = '#ffbc75'; ctx.beginPath(); ctx.moveTo(-5, 10); ctx.lineTo(0, 10 + flame); ctx.lineTo(5, 10); ctx.closePath(); ctx.fill();
        ctx.fillStyle = '#e9fff5'; ctx.beginPath(); ctx.moveTo(-2, 10); ctx.lineTo(0, 10 + flame * .55); ctx.lineTo(2, 10); ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
      }
      ctx.fillStyle = '#dce9f5'; ctx.strokeStyle = '#9cb4c9'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, -15); ctx.lineTo(7, -4); ctx.lineTo(5, 9); ctx.lineTo(-5, 9); ctx.lineTo(-7, -4); ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#4d806f'; ctx.beginPath(); ctx.moveTo(-5, 3); ctx.lineTo(-11, 11); ctx.lineTo(-5, 9); ctx.closePath(); ctx.fill(); ctx.beginPath(); ctx.moveTo(5, 3); ctx.lineTo(11, 11); ctx.lineTo(5, 9); ctx.closePath(); ctx.fill();
      ctx.fillStyle = '#18283a'; ctx.beginPath(); ctx.arc(0, -5, 3.1, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = '#a1f2d8'; ctx.stroke();
      ctx.strokeStyle = '#8fe9c9'; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(-4, 9); ctx.lineTo(-7, 13); ctx.moveTo(4, 9); ctx.lineTo(7, 13); ctx.stroke();
      ctx.restore();
      // target tracking line
      ctx.save(); ctx.strokeStyle = '#7df0c445'; ctx.setLineDash([2, 6]); ctx.beginPath(); ctx.moveTo(rx, ry + 18); ctx.lineTo(rx, groundY - 9); ctx.stroke(); ctx.restore();
    }
    // crosshair at center
    ctx.strokeStyle = '#a6b7cb50'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(w / 2 - 5, 25); ctx.lineTo(w / 2 + 5, 25); ctx.moveTo(w / 2, 20); ctx.lineTo(w / 2, 30); ctx.stroke();
  }
  function updateMeters() {
    $('hud-x').textContent = formatSigned(state.x, 1);
    $('hud-alt').textContent = Math.max(0, state.y).toFixed(1).padStart(5, '0');
    $('sim-time').textContent = 'T+' + simTime.toFixed(1).padStart(4, '0') + 's';
    $('tele-vx').innerHTML = formatSigned(state.vx, 2) + ' <small>m/s</small>';
    $('tele-vy').innerHTML = formatSigned(state.vy, 2) + ' <small>m/s</small>';
    $('tele-angle').innerHTML = formatSigned(state.angle * 180 / Math.PI, 1) + ' <small>deg</small>';
    $('tele-omega').innerHTML = formatSigned(state.omega, 2) + ' <small>rad/s</small>';
    $('tele-thrust').textContent = state.lastThrust ? 'ACTIVE' : 'OFF';
    $('bar-vx').style.width = Math.min(100, Math.abs(state.vx) / 6 * 100) + '%';
    $('bar-vy').style.width = Math.min(100, Math.abs(state.vy) / 8 * 100) + '%';
    $('bar-angle').style.width = Math.min(100, Math.abs(state.angle) / .9 * 100) + '%';
    $('bar-omega').style.width = Math.min(100, Math.abs(state.omega) / 2.5 * 100) + '%';
    $('bar-thrust').style.width = state.lastThrust ? '100%' : '0%';
  }
  function updateMetrics() {
    $('episodes-value').innerHTML = agent.episodes.toLocaleString() + ' <span>EP</span>';
    $('states-value').textContent = agent.q.size.toLocaleString() + ' states';
    $('epsilon-value').textContent = agent.epsilon().toFixed(3);
    $('success-value').innerHTML = evalInfo ? evalInfo.successRate.toFixed(0) + '<span>%</span>' : '—<span>%</span>';
    $('eval-sample').textContent = evalInfo ? evalInfo.n + ' episodes' : 'Not tested';
    $('reward-value').textContent = lastReport ? lastReport.reward.toFixed(1) : '—';
    $('last-outcome').textContent = lastReport ? (lastReport.landed ? 'Safe touchdown' : labelOutcome(lastReport.outcome)) : 'No flight yet';
    $('quality-value').textContent = lastReport ? (lastReport.landed ? 'PASS' : 'FAIL') : '—';
    $('quality-foot').textContent = lastReport ? (lastReport.landed ? 'Within safe limits' : 'Landing limits exceeded') : 'Awaiting telemetry';
    $('agent-status').textContent = jobBusy ? 'TRAINING IN PROGRESS' : agent.episodes >= 1 ? 'POLICY TRAINED' : 'BASELINE POLICY';
    $('agent-status-sub').textContent = jobBusy ? 'Updating Q-values locally' : agent.q.size.toLocaleString() + ' visited state bins';
    $('control-mode').textContent = modeName();
  }
  function labelOutcome(outcome) { return ({ crashed: 'Hard landing', out_of_bounds: 'Out of bounds', timeout: 'Flight timeout' })[outcome] || 'Flight ended'; }
  function startFlight() {
    if (jobBusy) return;
    env = RL.createEnvironment(Math.floor(Math.random() * 0xffffffff));
    state = env.reset(); running = true; tickAccumulator = 0; simTime = 0; episodeReward = 0; episodeStart = performance.now(); flightPath = [{ x: state.x, y: state.y }];
    lastReport = null; $('report-empty').classList.remove('hidden'); $('report-data').classList.add('hidden');
    $('flight-overlay').classList.add('hidden'); $('flight-overlay').classList.remove('failed');
    $('flight-state').textContent = 'IN FLIGHT'; $('pause-btn').disabled = false; $('pause-btn').innerHTML = 'Ⅱ <span>Pause</span>';
    $('launch-btn').disabled = true; $('agent-status-sub').textContent = 'Episode running';
    setMode($('policy-select').value); notice('Flight launched using ' + modeName().toLowerCase() + '.'); draw(); updateMeters();
  }
  function pauseFlight() {
    if (!running) return; running = false; $('pause-btn').innerHTML = '▶ <span>Resume</span>'; $('flight-state').textContent = 'PAUSED';
  }
  function resumeFlight() {
    if (running) return; running = true; lastTick = performance.now(); $('pause-btn').innerHTML = 'Ⅱ <span>Pause</span>'; $('flight-state').textContent = 'IN FLIGHT';
  }
  function finishFlight(result) {
    running = false; $('pause-btn').disabled = true; $('launch-btn').disabled = false;
    const s = result.state;
    lastReport = { landed: result.landed, outcome: result.info.outcome, reward: episodeReward, vx: s.vx, vy: s.vy, angle: s.angle, steps: s.steps, x: s.x, omega: s.omega };
    $('flight-state').textContent = result.landed ? 'LANDED' : 'FLIGHT ENDED';
    $('overlay-symbol').textContent = result.landed ? '✓' : '×';
    $('overlay-title').textContent = result.landed ? 'Touchdown confirmed' : labelOutcome(result.info.outcome);
    $('overlay-description').textContent = result.landed ? 'The vehicle reached the pad inside the configured safety limits.' : 'The episode did not meet the configured landing safety limits.';
    $('flight-overlay').classList.remove('hidden'); $('flight-overlay').classList.toggle('failed', !result.landed);
    $('report-empty').classList.add('hidden'); $('report-data').classList.remove('hidden');
    $('report-result-text').textContent = result.landed ? 'SAFE TOUCHDOWN' : labelOutcome(result.info.outcome).toUpperCase();
    $('report-reward').textContent = episodeReward.toFixed(1) + ' reward';
    $('report-dot').parentElement.classList.toggle('failed', !result.landed);
    $('report-vx').textContent = formatSigned(s.vx, 2) + ' m/s'; $('report-vy').textContent = formatSigned(s.vy, 2) + ' m/s';
    $('report-angle').textContent = formatSigned(s.angle * 180 / Math.PI, 1) + '°'; $('report-steps').textContent = s.steps;
    $('reward-value').textContent = episodeReward.toFixed(1); $('last-outcome').textContent = result.landed ? 'Safe touchdown' : labelOutcome(result.info.outcome);
    $('quality-value').textContent = result.landed ? 'PASS' : 'FAIL'; $('quality-foot').textContent = result.landed ? 'Within safe limits' : 'Landing limits exceeded';
    notice(result.landed ? 'Flight complete: safe touchdown recorded.' : 'Flight complete: ' + labelOutcome(result.info.outcome).toLowerCase() + '.');
    updateMetrics();
  }
  function animationFrame(now) {
    if (running) {
      tickAccumulator += Math.min(100, now - (lastTick || now));
      while (tickAccumulator >= 32 && running) {
        tickAccumulator -= 32;
        const action = policyAction(state, mode); lastAction = action;
        const result = env.step(state, action); state = result.state; episodeReward += result.reward; simTime += RL.CONFIG.dt;
        if (state.steps % 3 === 0) { flightPath.push({ x: state.x, y: state.y }); if (flightPath.length > 130) flightPath.shift(); }
        updateMeters();
        if (result.done) finishFlight(result);
      }
    }
    lastTick = now; draw(); requestAnimationFrame(animationFrame);
  }
  function episode(seed, learning) {
    const e = RL.createEnvironment(seed); let s = e.reset(); let total = 0; let steps = 0;
    while (!s.done && steps < RL.CONFIG.maxSteps) {
      const action = learning ? agent.choose(s, agent.epsilon()) : evaluationAction(s, 'learned');
      const result = e.step(s, action);
      if (learning) agent.update(s, action, result.reward, result.state, result.done);
      total += result.reward; s = result.state; steps++;
    }
    return { reward: total, landed: s.landed, outcome: s.outcome, steps, state: s };
  }
  function evaluationAction(s, policy) {
    if (policy === 'baseline') return RL.heuristicAction(s);
    if (policy === 'random') return RL.randomAction(randomGen);
    return RL.conservativeQAction(s, agent, 5);
  }
  function waitFrame() { return new Promise(resolve => setTimeout(resolve, 0)); }
  async function train() {
    if (jobBusy) return;
    running = false; trainTotal = Number($('train-count').value); trainProgress = 0; trainingStopRequested = false; jobBusy = true;
    $('train-btn').disabled = true; $('stop-train-btn').disabled = false; $('evaluate-btn').disabled = true; $('launch-btn').disabled = true; $('training-indicator-text').textContent = 'TRAINING';
    $('run-count').textContent = trainTotal.toLocaleString(); $('progress-bar').style.width = '0%'; $('progress-label').textContent = 'Initializing exploration...'; $('progress-percent').textContent = '0%';
    $('flight-state').textContent = 'TRAINING'; $('agent-status').textContent = 'TRAINING IN PROGRESS'; notice('Training started. The browser will work in small batches to keep the page responsive.');
    try {
      for (let i = 0; i < trainTotal; i++) {
        if (trainingStopRequested) break;
        const result = episode(Math.floor(Math.random() * 0xffffffff), true);
        agent.episodes++;
        episodeHistory.push({ episode: agent.episodes, reward: result.reward, landed: result.landed, steps: result.steps });
        if (episodeHistory.length > 3000) episodeHistory.shift();
        trainProgress++;
        if (trainProgress % 5 === 0 || trainProgress === trainTotal) {
          const pct = trainProgress / trainTotal * 100;
          $('progress-bar').style.width = pct + '%'; $('progress-percent').textContent = Math.round(pct) + '%';
          $('progress-label').textContent = 'Episode ' + agent.episodes.toLocaleString() + ' · ' + (result.landed ? 'safe landing' : labelOutcome(result.outcome).toLowerCase());
          $('epsilon-value').textContent = agent.epsilon().toFixed(3);
          $('episodes-value').innerHTML = agent.episodes.toLocaleString() + ' <span>EP</span>'; $('states-value').textContent = agent.q.size.toLocaleString() + ' states';
          drawChart(); await waitFrame();
        }
        if (trainProgress % 25 === 0) await waitFrame();
      }
      $('progress-label').textContent = trainingStopRequested ? 'Stopped at episode ' + agent.episodes.toLocaleString() : 'Batch complete · ' + agent.episodes.toLocaleString() + ' episodes total';
      notice(trainingStopRequested ? 'Training stopped. Your current Q-table is retained in memory.' : 'Training batch complete. Evaluate the policy to measure landing performance.');
    } catch (err) { console.error(err); notice('Training error: ' + err.message, true); }
    finally {
      jobBusy = false; $('train-btn').disabled = false; $('stop-train-btn').disabled = true; $('evaluate-btn').disabled = false; $('launch-btn').disabled = false;
      $('training-indicator-text').textContent = 'IDLE'; $('flight-state').textContent = 'STANDBY'; updateMetrics(); drawChart();
    }
  }
  async function evaluate() {
    if (jobBusy) return;
    running = false; jobBusy = true; $('evaluate-btn').disabled = true; $('train-btn').disabled = true; $('launch-btn').disabled = true;
    const n = 30; let success = 0, totalReward = 0, steps = 0; const results = [];
    $('training-indicator-text').textContent = 'EVALUATING'; $('progress-label').textContent = 'Evaluating 30 fixed-seed episodes...'; $('progress-bar').style.width = '0%'; $('progress-percent').textContent = '0%';
    notice('Evaluating policy on deterministic seed values without updating Q-values.');
    try {
      const evalMode = $('policy-select').value;
      for (let i = 0; i < n; i++) {
        const e = RL.createEnvironment(80000 + i); let s = e.reset(); let reward = 0; let count = 0;
        while (!s.done && count < RL.CONFIG.maxSteps) {
          let action;
          if (evalMode === 'baseline') action = RL.heuristicAction(s);
          else if (evalMode === 'random') action = RL.randomAction(RL.mulberry32(4500 + i * 701 + count));
          else action = evaluationAction(s, 'learned');
          const result = e.step(s, action); s = result.state; reward += result.reward; count++;
        }
        if (s.landed) success++; totalReward += reward; steps += count; results.push({ landed: s.landed, reward });
        $('progress-bar').style.width = ((i + 1) / n * 100) + '%'; $('progress-percent').textContent = Math.round((i + 1) / n * 100) + '%';
        if ((i + 1) % 3 === 0) { $('progress-label').textContent = 'Evaluated ' + (i + 1) + ' / ' + n + ' episodes'; await waitFrame(); }
      }
      evalInfo = { n, successRate: success / n * 100, meanReward: totalReward / n, meanSteps: steps / n, successes: success };
      $('success-value').innerHTML = evalInfo.successRate.toFixed(0) + '<span>%</span>'; $('eval-sample').textContent = n + ' episodes';
      $('progress-label').textContent = 'Evaluation complete · ' + success + '/' + n + ' safe landings';
      $('chart-trend').textContent = 'EVAL MEAN ' + evalInfo.meanReward.toFixed(1).toUpperCase();
      notice('Evaluation complete: ' + success + ' safe landings in ' + n + ' fixed-seed episodes. Mean reward ' + evalInfo.meanReward.toFixed(1) + '.');
    } catch (err) { console.error(err); notice('Evaluation error: ' + err.message, true); }
    finally { jobBusy = false; $('evaluate-btn').disabled = false; $('train-btn').disabled = false; $('launch-btn').disabled = false; $('training-indicator-text').textContent = 'IDLE'; updateMetrics(); }
  }
  function drawChart() {
    const rect = chart.getBoundingClientRect(); const dpr = Math.min(window.devicePixelRatio || 1, 2);
    chart.width = Math.max(1, Math.round(rect.width * dpr)); chart.height = Math.max(1, Math.round(rect.height * dpr));
    chartCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = rect.width, h = rect.height; chartCtx.clearRect(0, 0, w, h);
    const data = episodeHistory.slice(-180); $('chart-empty').classList.toggle('hidden', data.length > 0);
    $('chart-episodes').textContent = episodeHistory.length.toLocaleString() + ' EPISODES RECORDED';
    if (data.length < 1 || w < 10 || h < 10) return;
    const values = data.map(d => d.reward); const min = Math.min(-110, ...values); const max = Math.max(130, ...values); const pad = { l: 30, r: 8, t: 10, b: 23 }; const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;
    const y = val => pad.t + (max - val) / (max - min) * ch;
    chartCtx.font = '9px DM Mono, monospace'; chartCtx.textAlign = 'right'; chartCtx.textBaseline = 'middle'; chartCtx.fillStyle = '#647791';
    [-100, 0, 100].forEach(v => { const yy = y(v); chartCtx.strokeStyle = '#253348'; chartCtx.setLineDash([3, 5]); chartCtx.beginPath(); chartCtx.moveTo(pad.l, yy); chartCtx.lineTo(w - pad.r, yy); chartCtx.stroke(); chartCtx.setLineDash([]); chartCtx.fillText(String(v), pad.l - 7, yy); });
    const x = i => pad.l + (data.length <= 1 ? 0 : i / (data.length - 1) * cw);
    chartCtx.beginPath(); values.forEach((v, i) => { if (i === 0) chartCtx.moveTo(x(i), y(v)); else chartCtx.lineTo(x(i), y(v)); }); chartCtx.strokeStyle = '#7ca7ff'; chartCtx.lineWidth = 1.15; chartCtx.stroke();
    const avg = values.map((_, i) => { const slice = values.slice(Math.max(0, i - 19), i + 1); return slice.reduce((a, b) => a + b, 0) / slice.length; });
    chartCtx.beginPath(); avg.forEach((v, i) => { if (i === 0) chartCtx.moveTo(x(i), y(v)); else chartCtx.lineTo(x(i), y(v)); }); chartCtx.strokeStyle = '#7df0c4'; chartCtx.lineWidth = 1.8; chartCtx.stroke();
    if (avg.length > 1) { const diff = avg[avg.length - 1] - avg[Math.max(0, avg.length - 21)]; $('chart-trend').textContent = (diff >= 0 ? '↗ ' : '↘ ') + '20-EP TREND ' + (diff >= 0 ? '+' : '') + diff.toFixed(1); }
  }
  function exportPolicy() {
    const data = agent.exportJSON(); const blob = new Blob([JSON.stringify(data)], { type: 'application/json' }); const url = URL.createObjectURL(blob); const a = document.createElement('a');
    a.href = url; a.download = 'descent-lab-q-policy.json'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    notice('Exported ' + agent.q.size.toLocaleString() + ' state entries and ' + agent.episodes.toLocaleString() + ' training episodes.');
  }
  async function importPolicy(file) {
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text()); const count = agent.importJSON(parsed);
      episodeHistory = []; evalInfo = null; updateMetrics(); setMode($('policy-select').value); drawChart();
      notice('Imported policy with ' + count.toLocaleString() + ' state entries from ' + agent.episodes.toLocaleString() + ' episodes.');
    } catch (err) { notice('Could not import policy: ' + err.message, true); }
  }
  $('launch-btn').addEventListener('click', startFlight);
  $('pause-btn').addEventListener('click', () => running ? pauseFlight() : resumeFlight());
  $('reset-btn').addEventListener('click', () => { running = false; env = RL.createEnvironment(); state = env.reset(); episodeReward = 0; simTime = 0; flightPath = []; $('flight-overlay').classList.add('hidden'); $('flight-state').textContent = 'STANDBY'; $('pause-btn').disabled = true; $('launch-btn').disabled = false; updateMeters(); draw(); notice('Environment reset. No policy or training data was changed.'); });
  $('overlay-retry').addEventListener('click', startFlight);
  $('policy-select').addEventListener('change', e => setMode(e.target.value));
  $('train-count').addEventListener('change', e => $('run-count').textContent = Number(e.target.value).toLocaleString());
  $('train-btn').addEventListener('click', train);
  $('stop-train-btn').addEventListener('click', () => { trainingStopRequested = true; $('stop-train-btn').disabled = true; notice('Stopping after the current episode...'); });
  $('evaluate-btn').addEventListener('click', evaluate);
  $('export-btn').addEventListener('click', exportPolicy);
  $('import-file').addEventListener('change', e => importPolicy(e.target.files && e.target.files[0]));
  window.addEventListener('resize', () => { resizeCanvas(); drawChart(); });
  setMode($('policy-select').value); resizeCanvas(); drawChart(); updateMeters(); updateMetrics(); requestAnimationFrame(animationFrame);
})();
