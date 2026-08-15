/* EpsilonZero frontend */
"use strict";

const $ = (id) => document.getElementById(id);
const api = (url, opts) => fetch(url, opts).then((r) => r.json());
const post = (url, body) =>
  api(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });

let STATE = null; // latest trainer state
let CFG = null;

/* ---------------- tabs ---------------- */
document.querySelectorAll("nav button").forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll("nav button").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $("tab-" + b.dataset.tab).classList.add("active");
    if (b.dataset.tab === "charts") refreshCharts();
    if (b.dataset.tab === "games") loadGamesList();
    if (b.dataset.tab === "control") { fillConfig(); loadCheckpoints(); }
    if (b.dataset.tab === "play") resizeBoard();
  };
});

/* ---------------- live status (WebSocket) ---------------- */
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/status`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "state") updateState(msg.data);
  };
  ws.onclose = () => setTimeout(connectWS, 2000);
}
connectWS();

const STATUS_CN = {
  idle: "空闲", selfplay: "自我对弈中", training: "梯度训练中",
  evaluating: "评估中", pretraining: "棋谱预训练中",
};

function updateState(s) {
  STATE = s; CFG = s.config;
  $("statusDot").className = "status-dot" + (s.running || s.status !== "idle" ? " running" : "");
  $("statusText").textContent = (STATUS_CN[s.status] || s.status) +
    (s.device === "cuda" ? " · GPU 加速" : " · CPU");
  $("cStatus").textContent = STATUS_CN[s.status] || s.status;
  $("cIter").textContent = s.iteration;
  $("cGames").textContent = s.total_games;
  $("cBuffer").textContent = s.buffer_size;
  $("cElo").textContent = s.elo;
  $("cParams").textContent = (s.params / 1e6).toFixed(2) + "M";
  $("cBoard").textContent = `${s.config.board_size}×${s.config.board_size}`;
  $("cSims").textContent = s.config.mcts_simulations;
  const p = s.progress || {};
  $("progressLabel").textContent = p.label || "空闲";
  $("progressFill").style.width = p.total ? `${(100 * p.done / p.total).toFixed(1)}%` : "0%";
}

/* ---------------- records info ---------------- */
function loadRecordsInfo() {
  api("/api/records/stats").then((d) => {
    const leagues = Object.entries(d.by_league || {})
      .map(([k, v]) => `<span class="badge">${k}: ${v}</span>`).join(" ");
    const sizes = Object.entries(d.by_size || {})
      .map(([k, v]) => `<span class="badge">${k}×${k}: ${v}局</span>`).join(" ");
    $("recordsInfo").innerHTML = d.games
      ? `共 <b>${d.games}</b> 局基础棋谱 (Gomocup 顶级 AI 比赛)<br>来源: ${leagues}<br>棋盘: ${sizes}`
      : `暂无棋谱数据 — 请到「训练控制」页下载或生成`;
  });
}
loadRecordsInfo();

/* ---------------- board rendering ---------------- */
function drawBoard(canvas, size, cells, opts = {}) {
  const px = canvas.width;
  const margin = px * 0.055;
  const cell = (px - 2 * margin) / (size - 1);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, px, px);
  ctx.strokeStyle = "#7a5a20"; ctx.lineWidth = 1;
  for (let i = 0; i < size; i++) {
    ctx.beginPath(); ctx.moveTo(margin, margin + i * cell); ctx.lineTo(px - margin, margin + i * cell); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(margin + i * cell, margin); ctx.lineTo(margin + i * cell, px - margin); ctx.stroke();
  }
  // star points
  const star = size >= 13 ? [3, Math.floor(size / 2), size - 4] : [Math.floor(size / 2)];
  ctx.fillStyle = "#7a5a20";
  star.forEach((r) => star.forEach((c) => {
    ctx.beginPath(); ctx.arc(margin + c * cell, margin + r * cell, px * 0.006, 0, 7); ctx.fill();
  }));
  // stones
  const last = opts.lastMove;
  cells.forEach((v, i) => {
    if (!v) return;
    const r = Math.floor(i / size), c = i % size;
    const x = margin + c * cell, y = margin + r * cell;
    const rad = cell * 0.44;
    const grad = ctx.createRadialGradient(x - rad / 3, y - rad / 3, rad / 6, x, y, rad);
    if (v === 1) { grad.addColorStop(0, "#666"); grad.addColorStop(1, "#000"); }
    else { grad.addColorStop(0, "#fff"); grad.addColorStop(1, "#bbb"); }
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(x, y, rad, 0, 7); ctx.fill();
    if (i === last) {
      ctx.strokeStyle = "#f85149"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, rad * 0.5, 0, 7); ctx.stroke();
    }
  });
  return { margin, cell };
}

/* ---------------- play vs AI ---------------- */
let game = null;   // {session, size, human, cells, lastMove, over}
let boardGeom = null;

function resizeBoard() {
  const el = $("board");
  const w = Math.min(640, el.parentElement.clientWidth - 24);
  el.width = w; el.height = w;
  if (game) renderGameBoard();
}

function renderGameBoard() {
  boardGeom = drawBoard($("board"), game.size, game.cells, { lastMove: game.lastMove });
}

function winnerText(w) {
  if (w === 1) return "黑棋胜";
  if (w === -1) return "白棋胜";
  if (w === 2) return "平局";
  return "进行中";
}

function updateGameInfo(payload, aiInfo) {
  game.cells = payload.board;
  game.lastMove = payload.history?.length ? payload.history[payload.history.length - 1] : -1;
  renderGameBoard();
  let html = `棋盘: <b>${payload.size}×${payload.size}</b> · 手数: <b>${payload.history.length}</b><br>` +
    `状态: <b>${winnerText(payload.winner)}</b>`;
  if (payload.winner && payload.winner !== 2) {
    const humanWon = payload.winner === game.human;
    html += humanWon ? ` <span style="color:var(--green)">你赢了!</span>`
                     : ` <span style="color:var(--red)">AI 获胜</span>`;
  }
  if (aiInfo) {
    html += `<br>AI 局面评估: <b>${(aiInfo.value * 100).toFixed(1)}%</b> (AI 视角胜率)`;
    $("topMoves").innerHTML = (aiInfo.top_moves || [])
      .map((m) => `<div><span>(${m.row + 1}, ${m.col + 1})</span><span>${m.visits}</span></div>`)
      .join("") || "-";
  }
  $("gameInfo").innerHTML = html;
}

$("btnNewGame").onclick = async () => {
  const human = parseInt($("playColor").value);
  const sims = parseInt($("playSims").value);
  $("gameInfo").textContent = "AI 思考中…";
  const d = await post("/api/game/new", { human_color: human, ai_simulations: sims });
  game = { session: d.session, size: d.size, human, cells: d.board, lastMove: -1 };
  resizeBoard();
  updateGameInfo({ board: d.board, winner: 0, size: d.size,
                   history: d.ai_first_move ? [d.ai_first_move.move] : [] },
                 d.ai_first_move);
};

$("board").onclick = async (ev) => {
  if (!game || !boardGeom) return;
  const rect = ev.target.getBoundingClientRect();
  const scale = ev.target.width / rect.width;
  const x = (ev.clientX - rect.left) * scale, y = (ev.clientY - rect.top) * scale;
  const { margin, cell } = boardGeom;
  const col = Math.round((x - margin) / cell), row = Math.round((y - margin) / cell);
  if (row < 0 || col < 0 || row >= game.size || col >= game.size) return;
  $("gameInfo").textContent = "AI 思考中…";
  const d = await post("/api/game/move", { session: game.session, row, col });
  if (!d.ok) { updateGameInfo(d); return; }
  updateGameInfo(d, d.ai_move);
};

$("btnUndo").onclick = async () => {
  if (!game) return;
  const d = await post("/api/game/undo?session=" + game.session);
  if (d.ok) { $("topMoves").textContent = "-"; updateGameInfo(d); }
};

/* ---------------- custom canvas charts ---------------- */
const COLORS = ["#58a6ff", "#3fb950", "#f85149", "#d29922", "#bc8cff", "#39c5cf"];

function lineChart(canvas, series, opts = {}) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width = canvas.clientWidth * 2;
  const H = canvas.height = canvas.clientHeight * 2;
  ctx.clearRect(0, 0, W, H);
  const padL = 70, padR = 20, padT = 30, padB = 50;
  const all = series.flatMap((s) => s.data);
  if (!all.length) {
    ctx.fillStyle = "#8b949e"; ctx.font = "24px sans-serif";
    ctx.fillText("暂无数据", W / 2 - 60, H / 2); return;
  }
  let ymin = opts.ymin ?? Math.min(...all.map((p) => p.y));
  let ymax = opts.ymax ?? Math.max(...all.map((p) => p.y));
  if (ymax - ymin < 1e-9) { ymax += 1; ymin -= 1; }
  const pad = (ymax - ymin) * 0.08; ymin -= pad; ymax += pad;
  const xmin = opts.xmin ?? Math.min(...all.map((p) => p.x));
  const xmax = opts.xmax ?? Math.max(...all.map((p) => p.x));
  const X = (v) => padL + ((v - xmin) / Math.max(1e-9, xmax - xmin)) * (W - padL - padR);
  const Y = (v) => H - padB - ((v - ymin) / (ymax - ymin)) * (H - padT - padB);
  // grid + y labels
  ctx.strokeStyle = "#2d333b"; ctx.fillStyle = "#8b949e"; ctx.font = "20px sans-serif";
  for (let i = 0; i <= 4; i++) {
    const v = ymin + (i * (ymax - ymin)) / 4;
    ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(W - padR, Y(v)); ctx.stroke();
    ctx.fillText(fmtNum(v), 6, Y(v) + 7);
  }
  // x labels (a few)
  const nTicks = Math.min(6, Math.max(2, Math.floor(xmax - xmin + 1)));
  for (let i = 0; i < nTicks; i++) {
    const v = xmin + ((xmax - xmin) * i) / (nTicks - 1);
    ctx.fillText(fmtNum(v), X(v) - 15, H - 18);
  }
  if (opts.yline != null) {
    ctx.strokeStyle = "#444"; ctx.setLineDash([6, 6]);
    ctx.beginPath(); ctx.moveTo(padL, Y(opts.yline)); ctx.lineTo(W - padR, Y(opts.yline)); ctx.stroke();
    ctx.setLineDash([]);
  }
  series.forEach((s, si) => {
    ctx.strokeStyle = COLORS[si % COLORS.length]; ctx.lineWidth = 3;
    ctx.beginPath();
    s.data.forEach((p, i) => (i ? ctx.lineTo(X(p.x), Y(p.y)) : ctx.moveTo(X(p.x), Y(p.y))));
    ctx.stroke();
    // legend
    ctx.fillStyle = COLORS[si % COLORS.length];
    ctx.fillRect(padL + si * 170, 6, 18, 10);
    ctx.fillStyle = "#8b949e";
    ctx.fillText(s.name, padL + 24 + si * 170, 16);
  });
}

function barChart(canvas, items, opts = {}) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width = canvas.clientWidth * 2;
  const H = canvas.height = canvas.clientHeight * 2;
  ctx.clearRect(0, 0, W, H);
  if (!items.length) {
    ctx.fillStyle = "#8b949e"; ctx.font = "24px sans-serif";
    ctx.fillText("暂无数据", W / 2 - 60, H / 2); return;
  }
  const padL = 60, padB = 60, padT = 20;
  const ymax = Math.max(...items.map((i) => i.value)) * 1.15 || 1;
  const bw = (W - padL - 30) / items.length;
  ctx.font = "20px sans-serif";
  items.forEach((it, i) => {
    const h = (it.value / ymax) * (H - padT - padB);
    ctx.fillStyle = COLORS[i % COLORS.length];
    ctx.fillRect(padL + i * bw + bw * 0.15, H - padB - h, bw * 0.7, h);
    ctx.fillStyle = "#e6edf3";
    ctx.fillText(String(it.value), padL + i * bw + bw * 0.2, H - padB - h - 8);
    ctx.fillStyle = "#8b949e";
    ctx.fillText(it.name, padL + i * bw + bw * 0.1, H - 25);
  });
}

function fmtNum(v) {
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

/* ---------------- charts data ---------------- */
function smooth(data, win) {
  if (win <= 1 || data.length <= 2) return data;
  const out = [];
  for (let i = 0; i < data.length; i++) {
    const a = Math.max(0, i - win + 1);
    let s = 0;
    for (let j = a; j <= i; j++) s += data[j].y;
    out.push({ x: data[i].x, y: s / (i - a + 1) });
  }
  return out;
}

let chartsRefreshing = false;
async function refreshCharts() {
  if (chartsRefreshing) return;
  chartsRefreshing = true;
  try {
    const win = Math.max(1, parseInt($("smoothWin").value) || 1);
    const [train, steps, evals, evalGames, sp, pre] = await Promise.all([
      api("/api/metrics?type=train&limit=2000"),
      api("/api/metrics?type=train_step&limit=10000"),
      api("/api/metrics?type=eval&limit=2000"),
      api("/api/metrics?type=eval_game&limit=4000"),
      api("/api/metrics?type=selfplay&limit=4000"),
      api("/api/metrics?type=pretrain&limit=500"),
    ]);
    const by = (arr, key) => arr.map((m, i) => ({ x: i + 1, y: m[key] }));
    const byIter = (arr, key) => arr.map((m) => ({ x: m.iteration, y: m[key] }));

    lineChart($("chStepLoss"), [
      { name: "policy", data: smooth(by(steps, "policy_loss"), win) },
      { name: "value", data: smooth(by(steps, "value_loss"), win) },
      { name: "total", data: smooth(by(steps, "total_loss"), win) },
    ]);
    lineChart($("chStepAcc"), [
      { name: "policy_acc", data: smooth(by(steps, "policy_acc"), win) },
      { name: "value_acc", data: smooth(by(steps, "value_acc"), win) },
    ], { ymin: 0, ymax: 1 });
    lineChart($("chEntropy"), [{ name: "entropy", data: smooth(by(steps, "entropy"), win) }]);
    lineChart($("chLoss"), [
      { name: "total", data: byIter(train, "total_loss") },
      { name: "policy", data: byIter(train, "policy_loss") },
      { name: "value", data: byIter(train, "value_loss") },
    ]);
    lineChart($("chElo"), [{ name: "ELO", data: byIter(evals, "elo") }]);
    lineChart($("chWinrate"), [
      { name: "vs 启发式", data: byIter(evals, "vs_heuristic") },
      { name: "vs 随机", data: byIter(evals, "vs_random") },
    ], { ymin: 0, ymax: 1, yline: 0.5 });
    const heurGames = evalGames.filter((g) => g.opponent === "heuristic");
    lineChart($("chEvalGames"), [
      { name: "胜率(MA6)", data: smooth(heurGames.map((g, i) => ({ x: i + 1, y: g.ai_won ? 1 : 0 })), 6) },
      { name: "单局结果", data: heurGames.map((g, i) => ({ x: i + 1, y: g.ai_won ? 1 : 0 })) },
    ], { ymin: -0.05, ymax: 1, yline: 0.5 });
    lineChart($("chGameLen"), [{ name: "手数", data: smooth(by(sp, "moves"), win) }]);
    lineChart($("chBlackWin"), [
      { name: "黑胜率", data: smooth(sp.map((m, i) => ({ x: i + 1, y: m.winner === 1 ? 1 : 0 })), 20) },
    ], { ymin: 0, ymax: 1, yline: 0.5 });
    lineChart($("chDuration"), [{ name: "秒/局", data: smooth(by(sp, "duration"), win) }]);
    const winCount = { "黑胜": 0, "白胜": 0, "和棋": 0 };
    sp.forEach((m) => winCount[m.winner === 1 ? "黑胜" : m.winner === -1 ? "白胜" : "和棋"]++);
    barChart($("chWinner"), Object.entries(winCount).map(([name, value]) => ({ name, value })));
    lineChart($("chBuffer"), [{ name: "样本数", data: by(steps, "buffer_size") }]);
    lineChart($("chLr"), [{ name: "lr", data: by(steps, "lr") }]);
    lineChart($("chPretrain"), [
      { name: "loss", data: pre.map((m) => ({ x: m.epoch, y: m.loss })) },
      { name: "accuracy", data: pre.map((m) => ({ x: m.epoch, y: m.accuracy })) },
    ]);
    updateEventFeed([...train, ...steps.slice(-3), ...evals, ...evalGames.slice(-6), ...sp.slice(-5), ...pre]);
    $("chartsUpdated").textContent = "更新于 " + new Date().toLocaleTimeString();
  } finally {
    chartsRefreshing = false;
  }
}

function updateEventFeed(events) {
  events.sort((a, b) => (b.ts || 0) - (a.ts || 0));
  const fmt = {
    train_step: (m) => `训练步 it${m.iteration}#${m.step} loss=${m.total_loss.toFixed(3)} (p=${m.policy_loss.toFixed(3)} v=${m.value_loss.toFixed(3)}) p_acc=${(m.policy_acc*100).toFixed(0)}% v_acc=${(m.value_acc*100).toFixed(0)}%`,
    train: (m) => `第${m.iteration}轮训练完成 loss=${m.total_loss.toFixed(3)} 经验池=${m.buffer_size}`,
    selfplay: (m) => `自对弈 #${m.game} ${m.winner === 1 ? "黑胜" : m.winner === -1 ? "白胜" : "和"} ${m.moves}手 ${m.duration.toFixed(1)}s`,
    eval: (m) => `第${m.iteration}轮评估 对启发式=${(m.vs_heuristic*100).toFixed(0)}% 对随机=${(m.vs_random*100).toFixed(0)}% ELO=${m.elo.toFixed(0)}`,
    eval_game: (m) => `评估局 vs ${m.opponent === "heuristic" ? "启发式" : "随机"} AI执${m.ai_black ? "黑" : "白"} ${m.ai_won ? "胜" : "负/和"} ${m.moves}手`,
    pretrain: (m) => `预训练 epoch${m.epoch} loss=${m.loss.toFixed(3)} acc=${(m.accuracy*100).toFixed(1)}%`,
  };
  $("eventFeed").innerHTML = events.slice(0, 15).map((m) => {
    const t = m.ts ? new Date(m.ts * 1000).toLocaleTimeString() : "";
    const text = (fmt[m.type] || (() => m.type))(m);
    return `<div>[${t}] ${text}</div>`;
  }).join("") || "等待事件…";
}

$("btnRefreshCharts").onclick = refreshCharts;
$("smoothWin").onchange = refreshCharts;
setInterval(() => {
  if ($("autoRefresh").checked && $("tab-charts").classList.contains("active")) refreshCharts();
}, 2000);

/* ---------------- control tab ---------------- */
function fillConfig() {
  if (!CFG) return;
  $("cfgBoardSize").value = CFG.board_size;
  $("cfgWinLen").value = CFG.win_len;
  $("cfgSims").value = CFG.mcts_simulations;
  $("cfgTempMoves").value = CFG.temp_moves;
  $("cfgBatch").value = CFG.batch_size;
  $("cfgLr").value = CFG.lr;
  $("cfgGames").value = CFG.selfplay_games_per_iter;
  $("cfgSteps").value = CFG.train_steps_per_iter;
  $("cfgDModel").value = CFG.d_model;
  $("cfgLayers").value = CFG.n_layers;
  $("cfgHeads").value = CFG.n_heads;
  $("cfgDff").value = CFG.d_ff;
}

$("btnTrainStart").onclick = async () => {
  const it = parseInt($("trainIters").value) || 0;
  await post(`/api/train/start?iterations=${it}`);
};
$("btnTrainStop").onclick = () => post("/api/train/stop");

$("btnAcquire").onclick = async () => {
  $("taskStatus").textContent = "正在下载棋谱…";
  await post("/api/records/acquire");
  pollTasks();
};
$("btnPretrain").onclick = async () => {
  const ep = parseInt($("pretrainEpochs").value) || 3;
  const mx = parseInt($("pretrainMax").value) || 0;
  $("taskStatus").textContent = "预训练进行中… (可在总览页查看进度)";
  await post(`/api/pretrain?epochs=${ep}&max_games=${mx}`);
  pollTasks();
};

async function pollTasks() {
  const tasks = await api("/api/tasks");
  const parts = Object.entries(tasks).map(
    ([id, t]) => `${t.kind}: ${t.status}${t.result ? " " + JSON.stringify(t.result) : ""}`);
  $("taskStatus").textContent = parts.join(" | ") || "无任务";
  loadRecordsInfo();
}
setInterval(pollTasks, 4000);

$("btnSaveConfig").onclick = async () => {
  const body = {
    board_size: parseInt($("cfgBoardSize").value),
    win_len: parseInt($("cfgWinLen").value),
    mcts_simulations: parseInt($("cfgSims").value),
    temp_moves: parseInt($("cfgTempMoves").value),
    batch_size: parseInt($("cfgBatch").value),
    lr: parseFloat($("cfgLr").value),
    selfplay_games_per_iter: parseInt($("cfgGames").value),
    train_steps_per_iter: parseInt($("cfgSteps").value),
    d_model: parseInt($("cfgDModel").value),
    n_layers: parseInt($("cfgLayers").value),
    n_heads: parseInt($("cfgHeads").value),
    d_ff: parseInt($("cfgDff").value),
  };
  const d = await post("/api/config", body);
  $("configMsg").textContent = d.rebuilt ? "已保存并重建模型" : "已保存";
  setTimeout(() => ($("configMsg").textContent = ""), 3000);
};

$("btnClearMetrics").onclick = async () => {
  if (confirm("确定清空所有训练指标?")) await fetch("/api/metrics", { method: "DELETE" });
};

async function loadCheckpoints() {
  const list = await api("/api/checkpoints");
  $("ckptTable").querySelector("tbody").innerHTML = list.map((c) =>
    `<tr><td>${c.name}</td><td>${(c.size / 1024 / 1024).toFixed(1)} MB</td>` +
    `<td>${new Date(c.mtime * 1000).toLocaleString()}</td>` +
    `<td><button class="btn" onclick="loadCkpt('${c.name}')">加载</button></td></tr>`).join("");
}
window.loadCkpt = (name) => post(`/api/checkpoints/load?name=${name}`);

/* ---------------- self-play replay ---------------- */
let replay = null; // {moves, size, step, timer}
let replayGeom = null;

async function loadGamesList() {
  const list = await api("/api/selfplay?limit=50");
  $("gamesTable").querySelector("tbody").innerHTML = list.map((g, i) =>
    `<tr style="cursor:pointer" onclick="openGame('${g.name}')">` +
    `<td>${new Date(g.time * 1000).toLocaleString()}</td>` +
    `<td><span class="badge ${g.winner === 1 ? "black" : g.winner === -1 ? "white" : "draw"}">` +
    `${g.winner === 1 ? "黑" : g.winner === -1 ? "白" : "和"}</span></td>` +
    `<td>${g.moves}</td><td>${g.board_size}×${g.board_size}</td></tr>`).join("") ||
    `<tr><td colspan="4" class="muted">暂无自我对弈棋谱 — 先开始训练</td></tr>`;
}

window.openGame = async (name) => {
  const g = await api("/api/selfplay/" + name);
  replay = { moves: g.moves, size: g.board_size, step: 0, timer: null };
  const el = $("replayBoard");
  const w = Math.min(640, el.parentElement.clientWidth - 24);
  el.width = w; el.height = w;
  renderReplay();
};

function renderReplay() {
  if (!replay) return;
  const cells = new Array(replay.size * replay.size).fill(0);
  replay.moves.slice(0, replay.step).forEach((m, i) => (cells[m] = i % 2 === 0 ? 1 : -1));
  replayGeom = drawBoard($("replayBoard"), replay.size, cells,
    { lastMove: replay.step ? replay.moves[replay.step - 1] : -1 });
  $("rpStep").textContent = `${replay.step} / ${replay.moves.length}`;
}

$("rpFirst").onclick = () => { replay.step = 0; renderReplay(); };
$("rpPrev").onclick = () => { if (replay.step > 0) replay.step--; renderReplay(); };
$("rpNext").onclick = () => { if (replay.step < replay.moves.length) replay.step++; renderReplay(); };
$("rpLast").onclick = () => { replay.step = replay.moves.length; renderReplay(); };
$("rpAuto").onclick = () => {
  if (!replay) return;
  if (replay.timer) { clearInterval(replay.timer); replay.timer = null; $("rpAuto").textContent = "自动播放"; return; }
  $("rpAuto").textContent = "停止";
  replay.timer = setInterval(() => {
    if (replay.step >= replay.moves.length) { $("rpAuto").click(); return; }
    replay.step++; renderReplay();
  }, 400);
};

window.addEventListener("resize", () => { if (game) resizeBoard(); });
