"""Rule-based heuristic Gomoku player.

Used for:
- evaluation baseline during training
- generating synthetic base game records (基础棋谱) as a fallback
"""
from __future__ import annotations

import random
from typing import List, Optional

from .game.board import Board, EMPTY

# pattern scores
FIVE = 1_000_000
OPEN_FOUR = 100_000
FOUR = 10_000
OPEN_THREE = 5_000
THREE = 500
OPEN_TWO = 200
TWO = 50


def _line_score(count: int, open_ends: int, win_len: int) -> int:
    if count >= win_len:
        return FIVE
    if count == win_len - 1:
        return OPEN_FOUR if open_ends == 2 else FOUR
    if count == win_len - 2:
        return OPEN_THREE if open_ends == 2 else THREE
    if count == win_len - 3:
        return OPEN_TWO if open_ends == 2 else TWO
    return 0


def evaluate_move(board: Board, move: int, player: int) -> int:
    """Score placing `player` stone at `move` (max over 4 directions)."""
    n = board.size
    r, c = board.rc(move)
    best = 0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        open_ends = 0
        for sgn in (1, -1):
            rr, cc = r + sgn * dr, c + sgn * dc
            while 0 <= rr < n and 0 <= cc < n and board.cells[rr * n + cc] == player:
                count += 1
                rr += sgn * dr
                cc += sgn * dc
            if 0 <= rr < n and 0 <= cc < n and board.cells[rr * n + cc] == EMPTY:
                open_ends += 1
        best = max(best, _line_score(count, open_ends, board.win_len))
    return best


def candidate_moves(board: Board, radius: int = 2) -> List[int]:
    """Empty cells within `radius` of an existing stone (or center if empty board)."""
    n = board.size
    if not board.history:
        c = n // 2
        return [c * n + c]
    cands = set()
    for i, v in enumerate(board.cells):
        if v == EMPTY:
            continue
        r, c = board.rc(i)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n and board.cells[rr * n + cc] == EMPTY:
                    cands.add(rr * n + cc)
    return list(cands)


def heuristic_move(board: Board, noise: float = 0.0) -> Optional[int]:
    """Pick a move: win > block opponent win > best combined attack/defense."""
    me = board.current
    opp = -me
    cands = candidate_moves(board)
    if not cands:
        return None

    # 1. immediate win
    for m in cands:
        if evaluate_move(board, m, me) >= FIVE:
            return m
    # 2. block opponent's immediate win
    for m in cands:
        if evaluate_move(board, m, opp) >= FIVE:
            return m

    best_score, best = -1, None
    for m in cands:
        atk = evaluate_move(board, m, me)
        dfn = evaluate_move(board, m, opp)
        score = atk + dfn * 0.9
        if noise > 0:
            score *= 1.0 + random.uniform(-noise, noise)
        if score > best_score:
            best_score, best = score, m
    return best


def play_heuristic_game(board_size: int, win_len: int, noise: float = 0.3,
                        max_moves: int = 300) -> dict:
    """Two heuristic players (with randomness) play a full game -> game record."""
    board = Board(size=board_size, win_len=win_len)
    while not board.game_over() and len(board.history) < max_moves:
        m = heuristic_move(board, noise=noise)
        if m is None:
            break
        board.play(m)
    return {
        "moves": board.history,
        "winner": board.winner if board.winner != 2 else 0,
        "num_moves": len(board.history),
        "board_size": board_size,
        "win_len": win_len,
        "source": "heuristic",
    }
