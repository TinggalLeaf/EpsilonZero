"""FastAPI backend for EpsilonZero.

REST API for control/monitoring + WebSocket for live status push.
Static frontend served from web/static.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import Config, METRICS_FILE, SELFPLAY_DIR
from ..game.board import Board
from ..trainer import Trainer
from ..data import download_data

STATIC_DIR = Path(__file__).parent / "static"


class ConfigUpdate(BaseModel):
    board_size: Optional[int] = None
    win_len: Optional[int] = None
    mcts_simulations: Optional[int] = None
    c_puct: Optional[float] = None
    dirichlet_alpha: Optional[float] = None
    white_sims_boost: Optional[float] = None
    mcts_eval_batch: Optional[int] = None
    temp_moves: Optional[int] = None
    batch_size: Optional[int] = None
    lr: Optional[float] = None
    selfplay_games_per_iter: Optional[int] = None
    train_steps_per_iter: Optional[int] = None
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    n_layers: Optional[int] = None
    d_ff: Optional[int] = None
    device: Optional[str] = None


class NewGame(BaseModel):
    human_color: int = 1          # 1 = black (first), -1 = white
    ai_simulations: int = 100
    board_size: Optional[int] = None
    win_len: Optional[int] = None


class Move(BaseModel):
    session: str
    row: int
    col: int


def create_app(trainer: Trainer) -> FastAPI:
    app = FastAPI(title="EpsilonZero Gomoku AI")
    sessions: Dict[str, dict] = {}
    ws_clients: List[WebSocket] = []
    bg_tasks: Dict[str, dict] = {}

    # ---------------- state / config ----------------
    @app.get("/api/state")
    def state():
        return trainer.state()

    @app.post("/api/config")
    def update_config(upd: ConfigUpdate):
        cfg = trainer.cfg
        changed_model = False
        changed_device = False
        for k, v in upd.model_dump(exclude_none=True).items():
            if k == "device" and v != cfg.device:
                changed_device = True
            if k in ("board_size", "d_model", "n_heads", "n_layers", "d_ff") \
                    and getattr(cfg, k) != v:
                changed_model = True
            setattr(cfg, k, v)
        cfg.save()
        if changed_device and not changed_model:
            import torch
            trainer.stop()
            if trainer.thread and trainer.thread.is_alive():
                trainer.thread.join(timeout=10)
            trainer.device = "cuda" if (cfg.device == "auto" and
                                        torch.cuda.is_available()) else \
                             ("cuda" if cfg.device == "cuda" else "cpu")
            trainer.net.to(trainer.device)
            trainer.optimizer = torch.optim.AdamW(
                trainer.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            trainer._stop.clear()
        if changed_model:
            # rebuild network + optimizer for the new shape (old checkpoint kept on disk)
            was_running = trainer.is_running()
            trainer.stop()
            if was_running and trainer.thread:
                trainer.thread.join(timeout=5)
            from ..model.transformer import GomokuTransformer
            import torch
            trainer.net = GomokuTransformer(cfg.board_size, cfg.d_model, cfg.n_heads,
                                            cfg.n_layers, cfg.d_ff, cfg.dropout)
            trainer.optimizer = torch.optim.AdamW(
                trainer.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            trainer.buffer.data.clear()
            trainer.iteration = 0
            trainer._stop.clear()
        else:
            for g in trainer.optimizer.param_groups:
                g["lr"] = cfg.lr
        return {"ok": True, "rebuilt": changed_model, "config": trainer.state()["config"]}

    # ---------------- training control ----------------
    @app.post("/api/train/start")
    def train_start(iterations: int = 0):
        ok = trainer.start(iterations)
        return {"ok": ok}

    @app.post("/api/train/stop")
    def train_stop():
        trainer.stop()
        return {"ok": True}

    @app.get("/api/metrics")
    def metrics(type: Optional[str] = None, limit: int = 2000):
        data = trainer.metrics
        if type:
            data = [m for m in data if m.get("type") == type]
        return data[-limit:]

    @app.delete("/api/metrics")
    def clear_metrics():
        trainer.metrics.clear()
        METRICS_FILE.write_text("")
        return {"ok": True}

    # ---------------- checkpoints ----------------
    @app.get("/api/checkpoints")
    def checkpoints():
        return trainer.list_checkpoints()

    @app.post("/api/checkpoints/load")
    def load_ckpt(name: str = "latest"):
        return {"ok": trainer.load_checkpoint(name)}

    # ---------------- base records / pretrain ----------------
    @app.get("/api/records/stats")
    def records_stats():
        recs = download_data.load_records()
        by_league: Dict[str, int] = {}
        by_size: Dict[str, int] = {}
        for r in recs:
            by_league[r["source"]] = by_league.get(r["source"], 0) + 1
            by_size[str(r["board_size"])] = by_size.get(str(r["board_size"]), 0) + 1
        return {"games": len(recs), "by_league": by_league, "by_size": by_size}

    def _run_bg(kind: str, fn):
        tid = uuid.uuid4().hex[:8]
        bg_tasks[tid] = {"kind": kind, "status": "running", "result": None,
                         "started": time.time()}

        def wrap():
            try:
                bg_tasks[tid]["result"] = fn()
                bg_tasks[tid]["status"] = "done"
            except Exception as e:  # noqa: BLE001
                bg_tasks[tid]["status"] = f"error: {e}"
        threading.Thread(target=wrap, daemon=True).start()
        return tid

    @app.post("/api/records/acquire")
    def records_acquire():
        tid = _run_bg("acquire", lambda: download_data.acquire_base_records())
        return {"task": tid}

    @app.post("/api/pretrain")
    def pretrain(epochs: int = 3, max_games: int = 0):
        recs = download_data.load_records()
        if max_games:
            recs = recs[:max_games]
        tid = _run_bg("pretrain",
                      lambda: trainer.pretrain_from_records(recs, epochs=epochs))
        return {"task": tid, "games": len(recs)}

    @app.get("/api/tasks")
    def tasks():
        return bg_tasks

    # ---------------- play vs AI ----------------
    @app.post("/api/game/new")
    def game_new(g: NewGame):
        size = g.board_size or trainer.cfg.board_size
        win = g.win_len or trainer.cfg.win_len
        board = Board(size=size, win_len=win)
        sid = uuid.uuid4().hex[:8]
        sessions[sid] = {"board": board, "human": g.human_color,
                         "sims": g.ai_simulations, "created": time.time()}
        resp = {"session": sid, "size": size, "win_len": win,
                "human": g.human_color, "board": board.cells}
        if g.human_color == -1:  # AI (black) moves first
            move, info = trainer.ai_move(board, g.ai_simulations)
            board.play(move)
            resp["ai_first_move"] = {"move": move, "row": move // size,
                                     "col": move % size, **info}
            resp["board"] = board.cells
        return resp

    def _game_payload(sess: dict) -> dict:
        b: Board = sess["board"]
        return {"board": b.cells, "current": b.current, "winner": b.winner,
                "size": b.size, "history": b.history}

    @app.post("/api/game/move")
    def game_move(m: Move):
        sess = sessions.get(m.session)
        if not sess:
            return JSONResponse({"error": "session not found"}, status_code=404)
        b: Board = sess["board"]
        if b.game_over():
            return {"ok": False, "reason": "game over", **_game_payload(sess)}
        idx = m.row * b.size + m.col
        if not (0 <= idx < len(b.cells)) or b.cells[idx] != 0:
            return {"ok": False, "reason": "illegal move", **_game_payload(sess)}
        b.play(idx)
        payload = {"ok": True, **_game_payload(sess)}
        if not b.game_over() and b.current != sess["human"]:
            move, info = trainer.ai_move(b, sess["sims"])
            b.play(move)
            payload["ai_move"] = {"move": move, "row": move // b.size,
                                  "col": move % b.size, **info}
            payload.update(_game_payload(sess))
        return payload

    @app.post("/api/game/undo")
    def game_undo(session: str):
        sess = sessions.get(session)
        if not sess:
            return JSONResponse({"error": "session not found"}, status_code=404)
        b: Board = sess["board"]
        # undo AI + human move so it's the human's turn again
        b.undo()
        if b.history and b.current != sess["human"]:
            b.undo()
        return {"ok": True, **_game_payload(sess)}

    @app.get("/api/game/{session}")
    def game_state(session: str):
        sess = sessions.get(session)
        if not sess:
            return JSONResponse({"error": "session not found"}, status_code=404)
        return _game_payload(sess)

    # ---------------- self-play game viewer ----------------
    @app.get("/api/selfplay")
    def selfplay_list(limit: int = 50):
        files = sorted(SELFPLAY_DIR.glob("game_*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        out = []
        for f in files:
            try:
                d = json.loads(f.read_text())
                out.append({"name": f.name, "winner": d.get("winner"),
                            "moves": d.get("num_moves"),
                            "board_size": d.get("board_size"),
                            "time": d.get("time")})
            except Exception:
                continue
        return out

    @app.get("/api/selfplay/{name}")
    def selfplay_game(name: str):
        p = SELFPLAY_DIR / name
        if not p.exists() or p.parent != SELFPLAY_DIR:
            return JSONResponse({"error": "not found"}, status_code=404)
        return json.loads(p.read_text())

    # ---------------- websocket live status ----------------
    @app.websocket("/ws/status")
    async def ws_status(ws: WebSocket):
        await ws.accept()
        ws_clients.append(ws)
        try:
            while True:
                await ws.send_json({"type": "state", "data": trainer.state()})
                await asyncio.sleep(1.0)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if ws in ws_clients:
                ws_clients.remove(ws)

    # ---------------- static ----------------
    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.on_event("shutdown")
    def _shutdown():
        # best-effort graceful stop: save checkpoint + buffer on exit
        trainer.stop()
        if trainer.thread and trainer.thread.is_alive():
            trainer.thread.join(timeout=30)
        try:
            trainer.save_checkpoint("latest")
        except Exception:
            pass

    return app
