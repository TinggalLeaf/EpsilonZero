"""Entry point: run the EpsilonZero web server."""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="EpsilonZero Gomoku AI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    from .config import Config, CHECKPOINT_DIR
    from .trainer import Trainer
    from .web.server import create_app

    cfg = Config.load()
    trainer = Trainer(cfg)
    app = create_app(trainer)
    if cfg.auto_resume and (CHECKPOINT_DIR / "latest.pt").exists():
        trainer.start(0)  # resume training from checkpoint
        print(f"已从检查点恢复训练 (迭代 {trainer.iteration}, 经验池 {len(trainer.buffer)})")
    print(f"EpsilonZero 已启动: http://{args.host}:{args.port}  设备: {trainer.device}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
