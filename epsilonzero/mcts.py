"""MCTS (PUCT) guided by the transformer policy-value network."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .game.board import Board, EMPTY


class Node:
    __slots__ = ("prior", "visit", "value_sum", "children", "to_play")

    def __init__(self, prior: float, to_play: int):
        self.prior = prior
        self.to_play = to_play
        self.visit = 0
        self.value_sum = 0.0
        self.children: Dict[int, "Node"] = {}

    @property
    def q(self) -> float:
        return self.value_sum / self.visit if self.visit else 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    def __init__(self, net, board_size: int, win_len: int, simulations: int = 200,
                 c_puct: float = 1.5, dir_alpha: float = 0.3, dir_eps: float = 0.25,
                 device: str = "cpu"):
        self.net = net
        self.size = board_size
        self.win_len = win_len
        self.sims = simulations
        self.c_puct = c_puct
        self.dir_alpha = dir_alpha
        self.dir_eps = dir_eps
        self.device = device

    # ---------- network ----------
    @torch.no_grad()
    def _eval(self, board: Board) -> Tuple[np.ndarray, float]:
        x = torch.tensor(board.encode(), dtype=torch.float32,
                         device=self.device).view(1, 4, self.size, self.size)
        logits, value = self.net(x)
        mask = torch.zeros(1, self.size * self.size, device=self.device)
        legal = board.legal_moves()
        mask[0, legal] = 1.0
        logits = logits.masked_fill(mask <= 0, -1e9)
        policy = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        return policy, float(value.item())

    # ---------- search ----------
    def run(self, board: Board, add_noise: bool = False) -> np.ndarray:
        """Return visit-count distribution over all cells (size*size).

        Depth-1/2 tactical forcing at the root (standard in gomoku engines):
        take an immediate win if one exists; block the opponent's immediate
        win if there is exactly one threat; block moves that would give the
        opponent an open four (unblockable next move). This is just game-tree
        knowledge, not extra domain bias — MCTS would discover it given
        enough sims.
        """
        from .heuristic import candidate_moves, evaluate_move, OPEN_FOUR
        my_wins = board.winning_moves(board.current)
        if my_wins:
            counts = np.zeros(self.size * self.size, dtype=np.float32)
            for m in my_wins:
                counts[m] = 1.0
            return counts, None
        opp = -board.current
        opp_wins = board.winning_moves(opp)
        forced = set(opp_wins) if len(opp_wins) == 1 else None
        if forced is None:
            # depth-2: cells where the opponent would create an open four
            o4 = [m for m in candidate_moves(board)
                  if evaluate_move(board, m, opp) >= OPEN_FOUR]
            if o4:
                forced = set(o4)

        root = Node(0.0, board.current)
        policy, _ = self._eval(board)
        legal = board.legal_moves()
        if forced is not None:
            legal = [m for m in legal if m in forced]
        if add_noise and len(legal) > 1:
            noise = np.random.dirichlet([self.dir_alpha] * len(legal))
            for i, m in enumerate(legal):
                p = (1 - self.dir_eps) * policy[m] + self.dir_eps * noise[i]
                root.children[m] = Node(float(p), -board.current)
        else:
            for m in legal:
                root.children[m] = Node(float(policy[m]), -board.current)

        for _ in range(self.sims):
            b = board.copy()
            node = root
            path: List[Node] = [node]
            # selection
            while node.is_expanded() and not b.game_over():
                move, node = self._select(node)
                b.play(move)
                path.append(node)
            # expansion + evaluation
            if b.game_over():
                if b.winner == 2:
                    value = 0.0
                else:
                    # player to move at this terminal node is the one who lost
                    value = -1.0
            else:
                policy, v = self._eval(b)
                for m in b.legal_moves():
                    node.children[m] = Node(float(policy[m]), -b.current)
                value = v  # value from perspective of player to move at leaf
            # backprop: value is for path[-1].to_play
            for n in path:
                n.visit += 1
                n.value_sum += value if n.to_play == path[-1].to_play else -value

        counts = np.zeros(self.size * self.size, dtype=np.float32)
        for m, child in root.children.items():
            counts[m] = child.visit
        return counts, root

    def _select(self, node: Node) -> Tuple[int, Node]:
        total = math.sqrt(max(1, node.visit))
        best_score, best_move, best_child = -1e18, -1, None
        for m, child in node.children.items():
            u = self.c_puct * child.prior * total / (1 + child.visit)
            # child.q is stored from the perspective of the player to move at the
            # child (the opponent), so negate it for the deciding player
            score = -child.q + u
            if score > best_score:
                best_score, best_move, best_child = score, m, child
        return best_move, best_child


def counts_to_pi(counts: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 1e-3:
        pi = np.zeros_like(counts)
        pi[int(np.argmax(counts))] = 1.0
        return pi
    c = counts ** (1.0 / temperature)
    s = c.sum()
    return c / s if s > 0 else np.full_like(counts, 1.0 / len(counts))
