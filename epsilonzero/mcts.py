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
                 device: str = "cpu", eval_batch: int = 16):
        self.net = net
        self.size = board_size
        self.win_len = win_len
        self.sims = simulations
        self.c_puct = c_puct
        self.dir_alpha = dir_alpha
        self.dir_eps = dir_eps
        self.device = device
        self.eval_batch = max(1, eval_batch)

    # ---------- network ----------
    @torch.no_grad()
    def _eval(self, board: Board) -> Tuple[np.ndarray, float]:
        return self._eval_batch([board])[0]

    @torch.no_grad()
    def _eval_batch(self, boards: List[Board]) -> List[Tuple[np.ndarray, float]]:
        """One forward pass for many leaf boards (the whole point of wave
        batching: a B=16 forward costs about the same as B=1 on GPU)."""
        n = self.size
        x = torch.tensor(np.stack([b.encode() for b in boards]),
                         dtype=torch.float32, device=self.device).view(-1, 4, n, n)
        logits, values = self.net(x)
        mask = torch.zeros(len(boards), n * n, device=self.device)
        for i, b in enumerate(boards):
            mask[i, b.legal_moves()] = 1.0
        logits = logits.masked_fill(mask <= 0, -1e9)
        policies = torch.softmax(logits, dim=-1).cpu().numpy()
        vals = values.reshape(-1).cpu().numpy()
        return [(policies[i], float(vals[i])) for i in range(len(boards))]

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

        # Wave-batched simulations with virtual loss: each wave selects
        # `eval_batch` leaf positions (virtual loss keeps the wave from
        # collapsing onto one leaf), then evaluates them in a single batched
        # forward pass. Search semantics are unchanged apart from the virtual
        # loss, which is corrected exactly at backprop time.
        VL = 1.0  # virtual loss, applied from each node's own perspective
        sims_done = 0
        while sims_done < self.sims:
            wave = min(self.eval_batch, self.sims - sims_done)
            pending: List[Tuple[Board, Node, List[Node]]] = []
            for _ in range(wave):
                b = board.copy()
                node = root
                path: List[Node] = [node]
                # selection
                while node.is_expanded() and not b.game_over():
                    move, node = self._select(node)
                    b.play(move)
                    path.append(node)
                if b.game_over():
                    # terminal: exact value, backprop immediately (no VL)
                    if b.winner == 2:
                        value = 0.0
                    else:
                        # player to move at this terminal node is the one who lost
                        value = -1.0
                    for n in path:
                        n.visit += 1
                        n.value_sum += value if n.to_play == path[-1].to_play \
                            else -value
                else:
                    # reserve this path with virtual loss, queue for batch eval
                    for n in path:
                        n.visit += 1
                        n.value_sum += VL
                    pending.append((b, node, path))
            if pending:
                results = self._eval_batch([p[0] for p in pending])
                for (b, node, path), (policy, v) in zip(pending, results):
                    for m in b.legal_moves():
                        node.children[m] = Node(float(policy[m]), -b.current)
                    # backprop: value is for path[-1].to_play; subtract the VL
                    # reservation so the net effect is the true value
                    for n in path:
                        n.value_sum += (v - VL) if n.to_play == path[-1].to_play \
                            else (-v - VL)
            sims_done += wave

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
