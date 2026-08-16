"""Self-play game generation using MCTS + network."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from .game.board import Board, BLACK, WHITE
from .mcts import MCTS, counts_to_pi


def play_self_game(mcts: MCTS, board_size: int, win_len: int, temp_moves: int = 12,
                   max_moves: int = 300, augment: bool = True,
                   progress_cb: Optional[Callable[[int, int], None]] = None,
                   white_sims_boost: float = 1.0
                   ) -> Tuple[list, dict]:
    """Play one self-play game.

    Returns (samples, game_info). samples: list of (state, pi, z) with D4
    augmentation applied. game_info: dict with moves, winner, etc.
    white_sims_boost multiplies MCTS sims on white's turns (free-rule
    first-move advantage compensation).
    """
    board = Board(size=board_size, win_len=win_len)
    traj: List[Tuple[np.ndarray, np.ndarray, int]] = []  # state, pi, player-to-move
    hopeless = 0
    resigned = False
    base_sims = mcts.sims
    while not board.game_over() and len(board.history) < max_moves:
        mcts.sims = int(base_sims * (white_sims_boost if board.current == WHITE
                                     else 1.0))
        counts, root = mcts.run(board, add_noise=True)
        # early resignation: hopeless position for 2 consecutive moves
        if root is not None and len(board.history) >= 10:
            hopeless = hopeless + 1 if root.q < -0.985 else 0
            if hopeless >= 2:
                resigned = True
                break
        temp = 1.0 if len(board.history) < temp_moves else 1e-3
        pi = counts_to_pi(counts, temp)
        state = np.array(board.encode(), dtype=np.float32)
        traj.append((state, pi, board.current))
        move = int(np.random.choice(len(pi), p=pi)) if temp > 1e-3 else int(np.argmax(pi))
        board.play(move)
        if progress_cb:
            progress_cb(len(board.history), max_moves)

    if resigned:
        winner = -board.current
        outcome = {winner: 1.0, -winner: -1.0}
    elif board.winner == 2 or not board.game_over():
        outcome = {BLACK: 0.0, WHITE: 0.0}
        winner = 0
    else:
        winner = board.winner
        outcome = {winner: 1.0, -winner: -1.0}

    samples = []
    for state, pi, player in traj:
        z = outcome.get(player, 0.0)
        samples.append((state, pi, z))
        if augment:
            # D4 augmentation: 4 rotations x 2 flips (skip identity)
            n = board_size
            st = state.reshape(4, n, n)
            for k in (1, 2, 3):
                rs = np.rot90(st, k, axes=(1, 2))
                rp = np.rot90(pi.reshape(n, n), k)
                samples.append((rs.copy().reshape(-1), rp.copy().reshape(-1), z))
            fs = st[:, :, ::-1]
            fp = pi.reshape(n, n)[:, ::-1]
            samples.append((fs.copy().reshape(-1), fp.copy().reshape(-1), z))
            for k in (1, 2, 3):
                rs = np.rot90(fs, k, axes=(1, 2))
                rp = np.rot90(fp, k)
                samples.append((rs.copy().reshape(-1), rp.copy().reshape(-1), z))

    info = {
        "moves": board.history,
        "winner": winner,           # 1 black, -1 white, 0 draw/unfinished
        "num_moves": len(board.history),
        "board_size": board_size,
        "win_len": win_len,
        "resigned": resigned,
        "time": time.time(),
    }
    return samples, info


def save_game(info: dict, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    name = f"game_{int(info['time']*1000)}.json"
    (directory / name).write_text(json.dumps(info))
    return name
