"""Autopilot watchdog for EpsilonZero — fully autonomous, no AI in the loop.

Runs as a detached background process. Every CYCLE_SECONDS it:
  1. health-checks the training server (restarts it if dead, resumes training
     if idle),
  2. reads metrics via the HTTP API,
  3. applies AT MOST ONE repair action per cycle, by priority:
     a. server dead          -> restart server
     b. training stopped     -> resume training
     c. policy loss diverged (>50% up vs 3h ago, no recovery) or
        value loss > 0.30 sustained 2h -> halve lr (floor 1e-4, 2h cooldown)
     d. black fast-losses >=3/10 and median <100 -> white_sims_boost -0.25
        (floor 1.0, 1h cooldown)
     e. white fast-losses >=3/10 and median <100 -> white_sims_boost +0.25
        (cap 2.0, 1h cooldown)
     f. self-play black winrate >85% or <10% AND value_acc<0.6
        -> value-bias relapse: re-distill pro records (2000 games)
     g. no new 24-game winrate high for 6h -> re-distill pro records
        (3000 games; 12h cooldown either way)
  4. appends a Chinese report line to data/autopilot.log.

State persists in data/autopilot_state.json so restarts keep cooldowns.
Distillation ALWAYS uses the pro backup (base_records_pro28300_backup.json);
synthetic records are never generated and base_records.json is never
overwritten with anything else.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOG = DATA / "autopilot.log"
STATE_FILE = DATA / "autopilot_state.json"
PRO_BACKUP = DATA / "records" / "base_records_pro28300_backup.json"
BASE_RECORDS = DATA / "records" / "base_records.json"
PID_FILE = DATA / "autopilot.pid"

API = "http://127.0.0.1:8000"
CYCLE_SECONDS = 1800  # 30 min

LR_FLOOR = 1e-4
LR_COOLDOWN = 2 * 3600
BOOST_COOLDOWN = 1 * 3600
BOOST_MIN, BOOST_MAX, BOOST_STEP = 1.0, 2.0, 0.25
DISTILL_COOLDOWN = 12 * 3600
VLOSS_LIMIT, VLOSS_SUSTAIN = 0.30, 2 * 3600
PLOSS_JUMP = 1.5        # +50% vs 3h ago counts as divergence
STAGNANT_SECONDS = 6 * 3600


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(st: dict):
    STATE_FILE.write_text(json.dumps(st, indent=2))


def api_get(path: str, timeout=10):
    r = requests.get(API + path, timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload=None, timeout=30):
    r = requests.post(API + path, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def metrics(mtype: str, limit: int) -> list:
    try:
        return api_get(f"/api/metrics?type={mtype}&limit={limit}")
    except Exception:
        return []


def restart_server():
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    subprocess.Popen(
        [str(py), str(ROOT / "main.py"), "--port", "8000"],
        cwd=str(ROOT),
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    log("动作: 服务器无响应, 已后台重启 (断点续训自动接管)")


def wait_api(deadline_s: int = 180) -> bool:
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        try:
            api_get("/api/state", timeout=5)
            return True
        except Exception:
            time.sleep(5)
    return False


def distill_refresh(n_games: int, reason: str):
    """Serial re-distillation from the pro backup, then resume self-play."""
    try:
        st = api_get("/api/state")
        if st.get("status") == "pretraining":
            return  # already distilling
        import shutil
        shutil.copy(PRO_BACKUP, BASE_RECORDS)
        api_post("/api/train/stop")
        for _ in range(120):  # wait up to 10 min for iteration to wind down
            st = api_get("/api/state")
            if not st.get("running"):
                break
            time.sleep(5)
        res = api_post(f"/api/pretrain?epochs=1&max_games={n_games}", timeout=60)
        log(f"动作: 职业谱回炉蒸馏 ({reason}, {res.get('games', '?')} 局)")
        # wait for the background pretrain task to finish (poll state)
        for _ in range(720):  # up to 2h
            st = api_get("/api/state")
            if st.get("status") != "pretraining":
                break
            time.sleep(10)
        api_post("/api/train/start?iterations=0")
        log("动作: 蒸馏完成, 已恢复自对弈训练")
    except Exception as e:
        log(f"蒸馏失败: {e}; 尝试恢复训练")
        try:
            api_post("/api/train/start?iterations=0")
        except Exception:
            pass


def summarize_eval(games: list) -> dict:
    """Split recent heuristic eval games by AI color."""
    out = {}
    for color, key in ((True, "black"), (False, "white")):
        g = [m for m in games if m.get("opponent") == "heuristic"
             and m.get("ai_black") == color]
        wins = sum(1 for m in g if m.get("ai_won"))
        moves = [m["moves"] for m in g]
        fast = sum(1 for m in g if not m.get("ai_won") and m["moves"] < 50)
        out[key] = {
            "n": len(g), "wins": wins, "fast": fast,
            "med": statistics.median(moves) if moves else 0,
        }
    heur = [m for m in games if m.get("opponent") == "heuristic"]
    out["winrate"] = (sum(1 for m in heur if m.get("ai_won")) / len(heur)) if heur else 0.0
    return out


def cycle(st: dict):
    now = time.time()

    # ---- 1. health ----
    try:
        state = api_get("/api/state", timeout=8)
    except Exception:
        restart_server()
        wait_api()
        return
    if not state.get("running"):
        try:
            api_post("/api/train/start?iterations=0")
            log("动作: 训练处于停止状态, 已恢复训练")
        except Exception as e:
            log(f"恢复训练失败: {e}")
        return

    # ---- 2. collect metrics ----
    train = metrics("train", 40)
    steps = metrics("train_step", 100)
    eval_games = metrics("eval_game", 48)
    selfplay = metrics("selfplay", 60)

    p_loss = train[-1]["policy_loss"] if train else None
    v_loss = train[-1]["value_loss"] if train else None
    va = [m["value_acc"] for m in steps if m.get("value_acc")]
    value_acc = sum(va) / len(va) if va else 1.0
    ev = summarize_eval(eval_games)
    sp_b = sum(1 for m in selfplay if m.get("winner") == 1)
    sp_w = sum(1 for m in selfplay if m.get("winner") == -1)
    sp_n = len(selfplay) or 1
    black_pct = sp_b / sp_n

    # rolling history for trend rules
    hist = st.setdefault("loss_hist", [])  # [(ts, p_loss, v_loss)]
    hist.append([now, p_loss, v_loss])
    st["loss_hist"] = [h for h in hist if now - h[0] < 6 * 3600]

    cfg = api_get("/api/state")["config"]
    lr, boost = cfg["lr"], cfg.get("white_sims_boost", 1.0)

    report = (f"巡检: iter={state['iteration']} 局数={state['total_games']} "
              f"p_loss={p_loss:.3f} v_loss={v_loss:.3f} value_acc={value_acc:.2f} "
              f"启发式胜率={ev['winrate']:.0%} "
              f"黑{ev['black']['wins']}/{ev['black']['n']}(中位{ev['black']['med']:.0f}) "
              f"白{ev['white']['wins']}/{ev['white']['n']}(中位{ev['white']['med']:.0f}) "
              f"自对弈黑{black_pct:.0%} lr={lr} boost={boost}")

    # ---- 3. repair rules (one action max per cycle) ----
    def cooldown_ok(key: str, cd: float) -> bool:
        return now - st.get(key, 0) >= cd

    acted = False

    # c. loss divergence -> halve lr
    if not acted and cooldown_ok("lr_cut", LR_COOLDOWN) and lr / 2 >= LR_FLOOR * 0.99:
        old = [h for h in st["loss_hist"] if now - h[0] >= 3 * 3600]
        diverged = old and p_loss and p_loss > old[0][1] * PLOSS_JUMP \
            and p_loss > (hist[-2][1] if len(hist) > 1 else p_loss)
        vloss_bad = v_loss and v_loss > VLOSS_LIMIT and all(
            h[2] and h[2] > VLOSS_LIMIT
            for h in st["loss_hist"] if now - h[0] < VLOSS_SUSTAIN)
        if diverged or vloss_bad:
            new_lr = max(LR_FLOOR, lr / 2)
            api_post("/api/config", {"lr": new_lr})
            st["lr_cut"] = now
            acted = True
            log(f"动作: loss 异常 (p_loss={p_loss:.3f} v_loss={v_loss:.3f}), "
                f"lr {lr} -> {new_lr}")

    # d/e. color imbalance -> tune white_sims_boost
    if not acted and cooldown_ok("boost", BOOST_COOLDOWN):
        if ev["black"]["n"] >= 8 and ev["black"]["fast"] >= 3 \
                and ev["black"]["med"] < 100 and boost - BOOST_STEP >= BOOST_MIN:
            api_post("/api/config",
                     {"white_sims_boost": round(boost - BOOST_STEP, 2)})
            st["boost"] = now
            acted = True
            log(f"动作: 执黑速败 {ev['black']['fast']}/{ev['black']['n']}, "
                f"boost {boost} -> {boost - BOOST_STEP:.2f}")
        elif ev["white"]["n"] >= 8 and ev["white"]["fast"] >= 3 \
                and ev["white"]["med"] < 100 and boost + BOOST_STEP <= BOOST_MAX:
            api_post("/api/config",
                     {"white_sims_boost": round(boost + BOOST_STEP, 2)})
            st["boost"] = now
            acted = True
            log(f"动作: 执白速败 {ev['white']['fast']}/{ev['white']['n']}, "
                f"boost {boost} -> {boost + BOOST_STEP:.2f}")

    # f. value-bias relapse -> distill 2000
    if not acted and cooldown_ok("distill", DISTILL_COOLDOWN) \
            and (black_pct > 0.85 or black_pct < 0.10) and value_acc < 0.6:
        st["distill"] = now
        acted = True
        distill_refresh(2000, f"价值偏置复发: 自对弈黑胜率 {black_pct:.0%}, "
                              f"value_acc {value_acc:.2f}")

    # g. stagnation -> distill 3000
    if not acted and cooldown_ok("distill", DISTILL_COOLDOWN):
        best = st.get("best_winrate", 0.0)
        if ev["winrate"] > best and (ev["black"]["n"] + ev["white"]["n"]) >= 16:
            st["best_winrate"] = ev["winrate"]
            st["best_ts"] = now
            if ev["winrate"] >= 0.5:
                log(f"里程碑: 对启发式胜率 {ev['winrate']:.0%} (>=50%, 战略目标达成)")
            elif ev["winrate"] >= 0.25:
                log(f"里程碑: 对启发式胜率 {ev['winrate']:.0%} (>=25%)")
        elif now - st.get("best_ts", now) >= STAGNANT_SECONDS:
            st["distill"] = now
            acted = True
            distill_refresh(3000, "胜率 6 小时无新高")

    log(report + (" | 已处置" if acted else " | 正常"))
    save_state(st)


def main():
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text())
            import psutil  # optional; fall back to os.kill probe below
            if psutil.pid_exists(pid):
                print(f"autopilot already running (pid {pid})", flush=True)
                return
        except ImportError:
            try:
                import os
                os.kill(int(PID_FILE.read_text()), 0)
                print("autopilot already running", flush=True)
                return
            except Exception:
                pass
        except Exception:
            pass
    PID_FILE.write_text(str(__import__("os").getpid()))
    log("autopilot 启动 (每 30 分钟巡检一次)")
    st = load_state()
    while True:
        try:
            cycle(st)
        except Exception as e:
            log(f"巡检异常 (下周期重试): {e}")
        time.sleep(CYCLE_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
