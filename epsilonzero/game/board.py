"""Gomoku (五子棋) board rules.

Supports arbitrary board sizes and win lengths (e.g. 15x15 连五, 9x9 连五,
19x19 连五). Board is stored as a flat list: 0=empty, 1=current first player
(black), -1=white.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

EMPTY, BLACK, WHITE = 0, 1, -1


@dataclass
class Board:
    size: int = 15
    win_len: int = 5
    cells: List[int] = field(default_factory=list)
    current: int = BLACK
    winner: int = EMPTY  # 0 = ongoing, 1 / -1 = winner, 2 = draw
    history: List[int] = field(default_factory=list)  # move indices

    def __post_init__(self):
        if not self.cells:
            self.cells = [EMPTY] * (self.size * self.size)

    # ---------- helpers ----------
    def idx(self, r: int, c: int) -> int:
        return r * self.size + c

    def rc(self, i: int) -> Tuple[int, int]:
        return divmod(i, self.size)

    def legal_moves(self) -> List[int]:
        return [i for i, v in enumerate(self.cells) if v == EMPTY]

    def is_full(self) -> bool:
        return EMPTY not in self.cells

    def copy(self) -> "Board":
        return Board(self.size, self.win_len, list(self.cells), self.current,
                     self.winner, list(self.history))

    # ---------- core rules ----------
    def play(self, move: int) -> bool:
        """Place a stone for the current player. Returns False if illegal."""
        if self.winner != EMPTY or self.cells[move] != EMPTY:
            return False
        self.cells[move] = self.current
        self.history.append(move)
        if self._check_win(move):
            self.winner = self.current
        elif self.is_full():
            self.winner = 2  # draw
        self.current = -self.current
        return True

    def undo(self) -> Optional[int]:
        if not self.history:
            return None
        move = self.history.pop()
        self.cells[move] = EMPTY
        self.winner = EMPTY
        self.current = -self.current
        return move

    def _check_win(self, move: int) -> bool:
        r, c = self.rc(move)
        p = self.cells[move]
        n, wl = self.size, self.win_len
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sgn in (1, -1):
                rr, cc = r + sgn * dr, c + sgn * dc
                while 0 <= rr < n and 0 <= cc < n and self.cells[rr * n + cc] == p:
                    count += 1
                    rr += sgn * dr
                    cc += sgn * dc
            if count >= wl:
                return True
        return False

    def game_over(self) -> bool:
        return self.winner != EMPTY

    def winning_moves(self, player: int) -> List[int]:
        """Empty cells where placing `player`'s stone completes a line of win_len."""
        out = []
        n = self.size
        for i, v in enumerate(self.cells):
            if v != EMPTY:
                continue
            r, c = self.rc(i)
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                count = 1
                for sgn in (1, -1):
                    rr, cc = r + sgn * dr, c + sgn * dc
                    while 0 <= rr < n and 0 <= cc < n and self.cells[rr * n + cc] == player:
                        count += 1
                        rr += sgn * dr
                        cc += sgn * dc
                if count >= self.win_len:
                    out.append(i)
                    break
        return out

    # ---------- symmetry (D4 group) ----------
    def transform(self, k_rot: int, flip: bool) -> List[int]:
        """Return transformed cell list (for data augmentation)."""
        n = self.size
        out = [EMPTY] * (n * n)
        for r in range(n):
            for c in range(n):
                rr, cc = r, c
                if flip:
                    cc = n - 1 - cc
                for _ in range(k_rot):
                    rr, cc = cc, n - 1 - rr
                out[rr * n + cc] = self.cells[r * n + c]
        return out

    def transform_move(self, move: int, k_rot: int, flip: bool) -> int:
        n = self.size
        r, c = self.rc(move)
        if flip:
            c = n - 1 - c
        for _ in range(k_rot):
            r, c = c, n - 1 - r
        return r * n + c

    # ---------- encoding for the network ----------
    def encode(self) -> List[float]:
        """4 planes: current-player stones, opponent stones, last move,
        side-to-move (1.0 everywhere when the current player is black, else 0.0).

        Planes 1-2 are relative (self/opponent), so without plane 4 the network
        cannot tell whether it is playing black or white. Free-rule Gomoku is
        NOT color-symmetric in practice (first-move advantage: self-play black
        wins ~93%), so a color-blind network learns a single "attacker" style
        that only works for black — observed: AI played well as black but
        hopelessly as white. The side-to-move plane lets it learn
        color-specific strategy. Channel count is unchanged (4), so existing
        checkpoints still load.
        """
        n2 = self.size * self.size
        out = [0.0] * (4 * n2)
        last = self.history[-1] if self.history else -1
        for i, v in enumerate(self.cells):
            if v == self.current:
                out[i] = 1.0
            elif v == -self.current:
                out[n2 + i] = 1.0
        if last >= 0:
            out[2 * n2 + last] = 1.0
        if self.current == BLACK:
            for i in range(n2):
                out[3 * n2 + i] = 1.0
        return out

    def to_text(self) -> str:
        sym = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        lines = []
        for r in range(self.size):
            lines.append(" ".join(sym[self.cells[r * self.size + c]]
                                    for c in range(self.size)))
        return "\n".join(lines)
