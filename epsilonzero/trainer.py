"""Training orchestrator: self-play -> replay buffer -> SGD -> evaluation.

Runs in a background thread so the web UI can monitor and control it live.
"""
from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config, CHECKPOINT_DIR, METRICS_FILE, SELFPLAY_DIR, DATA_DIR
from .game.board import Board, BLACK
from .mcts import MCTS
from .model.transformer import GomokuTransformer
from .replay_buffer import ReplayBuffer
from .selfplay import play_self_game, save_game
from .heuristic import heuristic_move

BUFFER_FILE = DATA_DIR / "buffer.npz"


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if cfg.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device
        if self.device == "cpu":
            n_threads = cfg.num_threads or max(1, (os.cpu_count() or 4) - 1)
            torch.set_num_threads(n_threads)

        self.net = GomokuTransformer(cfg.board_size, cfg.d_model, cfg.n_heads,
                                     cfg.n_layers, cfg.d_ff, cfg.dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr,
                                           weight_decay=cfg.weight_decay)
        self.buffer = ReplayBuffer(cfg.buffer_size)

        self.lock = threading.Lock()          # guards net for concurrent inference
        self._stop = threading.Event()
        self._pause = threading.Event()
        self.thread: Optional[threading.Thread] = None

        self.iteration = 0
        self.total_games = 0
        self.elo = 1200.0
        self.elo_history: List[dict] = []
        self.status = "idle"   # idle | selfplay | training | evaluating | pretraining
        self.progress = {"done": 0, "total": 0, "label": ""}
        self.metrics: List[dict] = []
        self._load_metrics()
        self.load_checkpoint("latest")
        self._load_buffer()

    # ---------------- replay buffer persistence ----------------
    def _save_buffer(self):
        """Persist replay buffer so training can resume after a restart."""
        try:
            data = list(self.buffer.data)
            if not data:
                return
            states = np.stack([s[0] for s in data]).astype(np.uint8)
            pis = np.stack([s[1] for s in data]).astype(np.float16)
            zs = np.array([s[2] for s in data], dtype=np.float32)
            tmp = BUFFER_FILE.with_name("buffer_tmp.npz")
            np.savez(tmp, states=states, pis=pis, zs=zs)
            tmp.replace(BUFFER_FILE)
        except Exception:
            pass

    def _load_buffer(self):
        if not BUFFER_FILE.exists():
            return
        try:
            d = np.load(BUFFER_FILE)
            samples = [
                (d["states"][i].astype(np.float32),
                 d["pis"][i].astype(np.float32),
                 float(d["zs"][i]))
                for i in range(len(d["zs"]))
            ]
            self.buffer.add(samples)
        except Exception:
            pass

    # ---------------- metrics ----------------
    def _load_metrics(self):
        if METRICS_FILE.exists():
            for line in METRICS_FILE.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        self.metrics.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    def log_metric(self, entry: dict):
        entry.setdefault("ts", time.time())
        self.metrics.append(entry)
        with METRICS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ---------------- checkpoints ----------------
    def save_checkpoint(self, name: str = "latest"):
        path = CHECKPOINT_DIR / f"{name}.pt"
        torch.save({
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": vars(self.cfg) if hasattr(self.cfg, "__dict__") else {},
            "iteration": self.iteration,
            "total_games": self.total_games,
            "elo": self.elo,
        }, path)
        if name == "latest":
            self._save_buffer()
        return path

    def load_checkpoint(self, name: str = "latest") -> bool:
        path = CHECKPOINT_DIR / f"{name}.pt"
        if not path.exists():
            return False
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.net.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                except Exception:
                    pass
            self.iteration = ckpt.get("iteration", 0)
            self.total_games = ckpt.get("total_games", 0)
            self.elo = ckpt.get("elo", 1200.0)
            return True
        except Exception:
            return False

    def list_checkpoints(self) -> List[dict]:
        out = []
        for p in sorted(CHECKPOINT_DIR.glob("*.pt")):
            out.append({"name": p.stem, "size": p.stat().st_size,
                        "mtime": p.stat().st_mtime})
        return out

    # ---------------- inference ----------------
    @torch.no_grad()
    def predict(self, board: Board) -> tuple[np.ndarray, float]:
        with self.lock:
            self.net.eval()
            x = torch.tensor(board.encode(), dtype=torch.float32,
                             device=self.device).view(1, 4, board.size, board.size)
            logits, value = self.net(x)
            mask = torch.zeros(1, board.size * board.size, device=self.device)
            mask[0, board.legal_moves()] = 1.0
            logits = logits.masked_fill(mask <= 0, -1e9)
            policy = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        return policy, float(value.item())

    def make_mcts(self, simulations: Optional[int] = None) -> MCTS:
        net = _LockedNet(self)
        return MCTS(net, self.cfg.board_size, self.cfg.win_len,
                    simulations or self.cfg.mcts_simulations,
                    self.cfg.c_puct, self.cfg.dirichlet_alpha, self.cfg.dirichlet_eps,
                    device=self.device)

    def ai_move(self, board: Board, simulations: Optional[int] = None,
                temperature: float = 0.0) -> tuple[int, dict]:
        """Choose a move for the given board via MCTS. Returns (move, extra)."""
        mcts = self.make_mcts(simulations)
        counts, root = mcts.run(board, add_noise=False)
        if temperature > 1e-3:
            pi = counts / counts.sum() if counts.sum() else counts
            move = int(np.random.choice(len(pi), p=pi))
        else:
            move = int(np.argmax(counts))
        policy, value = self.predict(board)
        top = sorted(((float(counts[i]), i) for i in board.legal_moves()),
                     reverse=True)[:5]
        info = {
            "value": value,
            "top_moves": [{"move": i, "row": i // board.size, "col": i % board.size,
                           "visits": int(c)} for c, i in top],
            "simulations": mcts.sims,
        }
        return move, info

    # ---------------- training loop ----------------
    def start(self, iterations: int = 0):
        """Start background training. iterations=0 means run until stopped."""
        if self.thread and self.thread.is_alive():
            return False
        self._stop.clear()
        self.thread = threading.Thread(target=self._loop, args=(iterations,),
                                       daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self._stop.set()

    def is_running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def _loop(self, iterations: int):
        cfg = self.cfg
        it = 0
        while not self._stop.is_set():
            if iterations and it >= iterations:
                break
            it += 1
            self.iteration += 1

            # ---- self-play ----
            self.status = "selfplay"
            iter_samples: List[tuple] = []
            t0 = time.time()
            for g in range(cfg.selfplay_games_per_iter):
                if self._stop.is_set():
                    break
                self.progress = {"done": g, "total": cfg.selfplay_games_per_iter,
                                 "label": f"自我对弈 第{g+1}/{cfg.selfplay_games_per_iter}局"}
                mcts = self.make_mcts()
                samples, info = play_self_game(mcts, cfg.board_size, cfg.win_len,
                                               cfg.temp_moves, cfg.max_game_moves)
                iter_samples.extend(samples)
                self.total_games += 1
                save_game(info, SELFPLAY_DIR)
                self.log_metric({
                    "type": "selfplay", "iteration": self.iteration,
                    "game": self.total_games, "winner": info["winner"],
                    "moves": info["num_moves"], "duration": time.time() - t0,
                })
            self.buffer.add(iter_samples)

            # ---- SGD ----
            if self._stop.is_set():
                break
            self.status = "training"
            losses = self._train_steps(cfg.train_steps_per_iter)
            self.log_metric({
                "type": "train", "iteration": self.iteration,
                "policy_loss": losses["policy"], "value_loss": losses["value"],
                "total_loss": losses["total"], "entropy": losses["entropy"],
                "lr": self.optimizer.param_groups[0]["lr"],
                "buffer_size": len(self.buffer),
                "games": self.total_games,
            })

            # ---- evaluation + checkpoint ----
            self.status = "evaluating"
            self.progress = {"done": 0, "total": 1, "label": "评估中"}
            ev = self.evaluate(n_games=6, simulations=max(30, cfg.mcts_simulations // 4))
            self._update_elo(ev["vs_heuristic"])
            self.log_metric({
                "type": "eval", "iteration": self.iteration,
                "vs_random": ev["vs_random"], "vs_heuristic": ev["vs_heuristic"],
                "avg_moves": ev["avg_moves"], "elo": self.elo,
            })
            self.save_checkpoint("latest")
            self.save_checkpoint(f"iter_{self.iteration}")

        self.status = "idle"
        self.progress = {"done": 0, "total": 0, "label": ""}
        self.save_checkpoint("latest")

    def _train_steps(self, steps: int) -> Dict[str, float]:
        self.net.train()
        agg = {"policy": 0.0, "value": 0.0, "total": 0.0, "entropy": 0.0}
        n_done = 0
        for s in range(steps):
            if self._stop.is_set():
                break
            self.progress = {"done": s, "total": steps,
                             "label": f"梯度训练 {s+1}/{steps}"}
            states, pis, zs = self.buffer.sample(self.cfg.batch_size)
            with self.lock:
                n = self.cfg.board_size
                dev = self.device
                x = torch.tensor(states, dtype=torch.float32, device=dev).view(-1, 4, n, n)
                pi = torch.tensor(pis, dtype=torch.float32, device=dev)
                z = torch.tensor(zs, dtype=torch.float32, device=dev).unsqueeze(1)
                logits, value = self.net(x)
                policy_loss = -(pi * F.log_softmax(logits, dim=-1)).sum(-1).mean()
                value_loss = F.mse_loss(value, z)
                loss = policy_loss + self.cfg.value_weight * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                self.optimizer.step()
                p = F.softmax(logits.detach(), dim=-1)
                entropy = float(-(p * (p + 1e-12).log()).sum(-1).mean())
                # policy top-1 agreement with MCTS visit distribution
                pol_acc = float((logits.argmax(-1) == pi.argmax(-1)).float().mean())
                # value sign accuracy on decisive samples
                decisive = z.abs() > 0.5
                if decisive.any():
                    vacc = float(((value[decisive] > 0) == (z[decisive] > 0)).float().mean())
                else:
                    vacc = 0.0
            agg["policy"] += policy_loss.item()
            agg["value"] += value_loss.item()
            agg["total"] += loss.item()
            agg["entropy"] += entropy
            n_done += 1
            self.log_metric({
                "type": "train_step", "iteration": self.iteration, "step": s + 1,
                "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                "total_loss": loss.item(), "entropy": entropy,
                "policy_acc": pol_acc, "value_acc": vacc,
                "lr": self.optimizer.param_groups[0]["lr"],
                "buffer_size": len(self.buffer), "games": self.total_games,
            })
        if n_done:
            for k in agg:
                agg[k] /= n_done
        return agg

    # ---------------- evaluation ----------------
    def evaluate(self, n_games: int = 10, simulations: int = 50) -> dict:
        """Play eval games vs heuristic baseline and random policy."""
        res = {"vs_random": 0.0, "vs_heuristic": 0.0, "avg_moves": 0.0}
        for opponent in ("random", "heuristic"):
            wins = 0
            moves_total = 0
            for g in range(n_games):
                if self._stop.is_set():
                    break
                ai_black = g % 2 == 0
                winner, nmoves = self._eval_game(opponent, ai_black, simulations)
                moves_total += nmoves
                ai_won = winner != 0 and (winner == 1) == ai_black
                if ai_won:
                    wins += 1
                self.log_metric({
                    "type": "eval_game", "iteration": self.iteration,
                    "opponent": opponent, "ai_black": ai_black,
                    "ai_won": ai_won, "winner": winner, "moves": nmoves,
                })
            played = max(1, n_games)
            res[f"vs_{opponent}"] = wins / played
            res["avg_moves"] = moves_total / played
        return res

    def _eval_game(self, opponent: str, ai_black: bool, simulations: int):
        board = Board(size=self.cfg.board_size, win_len=self.cfg.win_len)
        mcts = self.make_mcts(simulations)
        while not board.game_over() and len(board.history) < self.cfg.max_game_moves:
            is_ai_turn = (board.current == BLACK) == ai_black
            if is_ai_turn:
                counts, _ = mcts.run(board, add_noise=False)
                move = int(np.argmax(counts))
            elif opponent == "random":
                move = random.choice(board.legal_moves())
            else:
                move = heuristic_move(board, noise=0.05)
                if move is None:
                    move = random.choice(board.legal_moves())
            board.play(move)
        winner = board.winner if board.winner in (1, -1) else 0
        return winner, len(board.history)

    def _update_elo(self, score: float, k: float = 32.0):
        expected = 1.0 / (1.0 + 10 ** ((1400.0 - self.elo) / 400.0))  # heuristic ~1400
        self.elo += k * (score - expected)
        self.elo_history.append({"iteration": self.iteration, "elo": self.elo})

    # ---------------- supervised pretrain from records ----------------
    def pretrain_from_records(self, records: List[dict], epochs: int = 3,
                              progress_cb: Optional[Callable] = None) -> dict:
        """Behavior-clone downloaded/self-generated game records."""
        self._stop.clear()
        self.status = "pretraining"
        samples: List[tuple] = []
        for rec in records:
            samples.extend(record_to_samples(rec, augment=True))
        if not samples:
            self.status = "idle"
            return {"samples": 0, "epochs": 0, "final_loss": None}
        # bucket by board size so 15x15 and 20x20 records can mix
        buckets: Dict[int, List[tuple]] = {}
        for s in samples:
            n = int(round((len(s[0]) // 4) ** 0.5))
            buckets.setdefault(n, []).append(s)
        bs = self.cfg.batch_size
        self.net.train()
        final_loss = 0.0
        total_samples = len(samples)
        for ep in range(epochs):
            for bkt in buckets.values():
                random.shuffle(bkt)
            ep_loss, ep_cnt, correct, total_moves = 0.0, 0, 0, 0
            done = 0
            batches_since_save = 0
            for n, bkt in buckets.items():
                for i in range(0, len(bkt), bs):
                    if self._stop.is_set():
                        break
                    chunk = bkt[i:i + bs]
                    states = np.stack([c[0] for c in chunk])
                    pis = np.stack([c[1] for c in chunk])
                    zs = np.array([c[2] for c in chunk], dtype=np.float32)
                    with self.lock:
                        x = torch.tensor(states, dtype=torch.float32,
                                         device=self.device).view(-1, 4, n, n)
                        pi = torch.tensor(pis, dtype=torch.float32, device=self.device)
                        z = torch.tensor(zs, dtype=torch.float32,
                                         device=self.device).unsqueeze(1)
                        logits, value = self.net(x)
                        policy_loss = -(pi * F.log_softmax(logits, dim=-1)).sum(-1).mean()
                        value_loss = F.mse_loss(value, z)
                        loss = policy_loss + self.cfg.value_weight * value_loss
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                        self.optimizer.step()
                        pred = logits.argmax(-1)
                        target = pi.argmax(-1)
                        correct += int((pred == target).sum())
                        total_moves += len(chunk)
                    ep_loss += loss.item()
                    ep_cnt += 1
                    done += len(chunk)
                    batches_since_save += 1
                    if batches_since_save >= 200:
                        batches_since_save = 0
                        self.save_checkpoint("latest")  # periodic save: stop anytime
                    if progress_cb:
                        progress_cb(ep, done, total_samples, ep_loss / max(1, ep_cnt))
                    self.progress = {"done": done, "total": total_samples,
                                     "label": f"棋谱预训练 epoch {ep+1}/{epochs}"}
            final_loss = ep_loss / max(1, ep_cnt)
            acc = correct / max(1, total_moves)
            self.log_metric({"type": "pretrain", "epoch": ep + 1, "loss": final_loss,
                             "accuracy": acc, "samples": total_samples})
        self.save_checkpoint("latest")
        self.status = "idle"
        self.progress = {"done": 0, "total": 0, "label": ""}
        return {"samples": total_samples, "epochs": epochs, "final_loss": final_loss}

    # ---------------- status ----------------
    def state(self) -> dict:
        return {
            "status": self.status,
            "running": self.is_running(),
            "iteration": self.iteration,
            "total_games": self.total_games,
            "buffer_size": len(self.buffer),
            "elo": round(self.elo, 1),
            "progress": self.progress,
            "params": self.net.param_count(),
            "device": self.device,
            "config": {k: getattr(self.cfg, k) for k in self.cfg.__dataclass_fields__},
        }


class _LockedNet(torch.nn.Module):
    """Wraps trainer.predict so MCTS inference is serialized with training."""

    def __init__(self, trainer: Trainer):
        super().__init__()
        self.trainer = trainer

    def forward(self, x: torch.Tensor):
        with self.trainer.lock:
            self.trainer.net.eval()
            return self.trainer.net(x)


def record_to_samples(rec: dict, augment: bool = True) -> List[tuple]:
    """Convert a game record {moves, winner, board_size, win_len} to (state, pi, z)."""
    size = rec.get("board_size", 15)
    win_len = rec.get("win_len", 5)
    winner = rec.get("winner", 0)
    board = Board(size=size, win_len=win_len)
    out = []
    for m in rec["moves"]:
        state = np.array(board.encode(), dtype=np.float32)
        pi = np.zeros(size * size, dtype=np.float32)
        pi[m] = 1.0
        player = board.current
        z = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
        out.append((state, pi, z))
        if augment:
            n = size
            st = state.reshape(4, n, n)
            p2 = pi.reshape(n, n)
            for k in (1, 2, 3):
                out.append((np.rot90(st, k, axes=(1, 2)).copy().reshape(-1),
                            np.rot90(p2, k).copy().reshape(-1), z))
            fs, fp = st[:, :, ::-1], p2[:, ::-1]
            out.append((fs.copy().reshape(-1), fp.copy().reshape(-1), z))
            for k in (1, 2, 3):
                out.append((np.rot90(fs, k, axes=(1, 2)).copy().reshape(-1),
                            np.rot90(fp, k).copy().reshape(-1), z))
        if not board.play(m):
            break
    return out
