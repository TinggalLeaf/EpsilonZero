"""Global configuration for EpsilonZero."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
RECORD_DIR = DATA_DIR / "records"      # downloaded base game records
SELFPLAY_DIR = DATA_DIR / "selfplay"   # self-play games (jsonl)
METRICS_FILE = DATA_DIR / "metrics.jsonl"
CONFIG_FILE = DATA_DIR / "config.json"

for d in (CHECKPOINT_DIR, RECORD_DIR, SELFPLAY_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    # board
    board_size: int = 15
    win_len: int = 5
    # model
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 256
    dropout: float = 0.1
    # mcts
    mcts_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.15
    dirichlet_eps: float = 0.25
    white_sims_boost: float = 2.0  # white moves get Nx MCTS sims: compensates
                                   # free-rule first-move advantage so white
                                   # stops being the "dumb" side
    mcts_eval_batch: int = 16      # leaf evaluations per batched forward pass
                                   # (wave batching; 1 = classic serial MCTS)
    temp_moves: int = 12          # first N moves sampled by temperature
    # training
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    buffer_size: int = 100_000
    selfplay_games_per_iter: int = 10
    train_steps_per_iter: int = 50
    value_weight: float = 1.0
    max_game_moves: int = 300
    # runtime
    num_threads: int = 0           # 0 = auto (os.cpu_count)
    device: str = "auto"           # auto | cpu | cuda
    auto_resume: bool = True       # start training automatically on launch if checkpoint exists

    def save(self, path: Path = CONFIG_FILE):
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path = CONFIG_FILE) -> "Config":
        if path.exists():
            data = json.loads(path.read_text())
            known = {f for f in cls.__dataclass_fields__}
            return cls(**{k: v for k, v in data.items() if k in known})
        return cls()
