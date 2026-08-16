"""One-shot: download ALL gomocup archives (2023-2025), parse every 15x15
game, save as base_records.json, then behavior-clone them into the current
checkpoint in memory-safe chunks (no D4 augmentation — dataset is huge).

Run with the server STOPPED (serial distillation, per project decision).
"""
from __future__ import annotations

import json
import random
import time
from collections import Counter

from epsilonzero.config import Config
from epsilonzero.data import download_data
from epsilonzero.trainer import Trainer

CHUNK_GAMES = 3000  # ~600MB of samples per chunk, augment=False


def main():
    t0 = time.time()
    print("[1/3] downloading archives (2023/2024/2025)...", flush=True)
    zips = download_data.download_archives()
    print("  zips:", [z.name for z in zips], flush=True)

    print("[2/3] parsing all psq games...", flush=True)
    records = download_data.load_archives(zips)
    recs15 = [r for r in records if r["board_size"] == 15]
    random.shuffle(recs15)
    by_league = Counter(r["source"] for r in recs15)
    by_winner = Counter(r["winner"] for r in recs15)
    print(f"  total parsed: {len(records)}, 15x15 kept: {len(recs15)}")
    print(f"  by league: {dict(by_league)}")
    print(f"  by winner (1=black,-1=white,0=draw): {dict(by_winner)}")
    download_data.save_records(recs15)  # -> data/records/base_records.json

    print("[3/3] distilling into current checkpoint...", flush=True)
    cfg = Config.load()
    trainer = Trainer(cfg)
    for i in range(0, len(recs15), CHUNK_GAMES):
        chunk = recs15[i:i + CHUNK_GAMES]
        res = trainer.pretrain_from_records(chunk, epochs=1, augment=False)
        print(f"  games {i}-{i + len(chunk)}: samples={res['samples']} "
              f"loss={res['final_loss']:.4f} elapsed={time.time()-t0:.0f}s",
              flush=True)
    trainer.save_checkpoint("latest")
    print(f"DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
