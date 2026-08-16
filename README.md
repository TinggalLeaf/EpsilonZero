# EpsilonZero · Transformer 五子棋 AI

基于 **Transformer + AlphaZero 式自我对弈强化学习** 的五子棋 AI，支持 **GPU/CPU 训练**，带完整 Web UI。

## 功能

- **Transformer 策略-价值网络**：棋盘每个交叉点作为一个 token，2D 可学习位置编码（双线性插值，天然支持任意棋盘大小）
- **自我对弈训练**（训练模式）：MCTS(PUCT) + Dirichlet 噪声探索 → 经验回放池 → 策略/价值联合训练 → 自动评估（对随机 / 对启发式引擎）→ ELO 追踪 → 自动保存检查点
- **断点续训**：权重、优化器状态、经验回放池（10 万样本）全部持久化，随时中止/重启自动恢复并继续训练
- **职业棋谱蒸馏**：内置下载器抓取 Gomocup（世界顶级五子棋 AI 比赛）2023-2025 全部对局（psq 格式，11.8 万+ 局），行为克隆蒸馏；`scripts/distill_full.py` 支持内存安全的分块全量蒸馏
- **战术强制**：MCTS 根节点深度-1（必胜必下 / 单威胁必挡）与深度-2（防活四）强制，杜绝低级漏杀
- **白方补偿**：自由规则五子棋先手优势巨大，`white_sims_boost` 让白方着子获得多倍 MCTS 模拟数，平衡黑白训练信号
- **认输机制**：价值评估连续极度绝望时提前认输，节省算力
- **人机对局**（对局模式）：网页棋盘点击落子，可选执黑/执白、AI 思考深度（MCTS 模拟数），显示 AI 局面评估与候选着法，支持悔棋
- **Web UI**：
  - 总览：状态、迭代、对局数、经验池、ELO、参数量、实时进度条（WebSocket 推送）
  - 训练图表：总/策略/价值损失、策略熵、策略/价值准确率、ELO 曲线、评估胜率、对局长度、胜负分布、经验池大小、预训练损失/准确率，自动刷新
  - 训练控制：开始/停止训练（可设迭代轮数）、下载棋谱、预训练、全部超参数配置、检查点管理、清空指标
  - 棋谱回放：浏览自我对弈棋谱，单步/自动播放回放
- **任意棋盘规格**：棋盘大小（9/13/15/19/20…）与连子数可配置，15×15 与 20×20 棋谱可混合训练
- **D4 对称增强**：自我对弈与棋谱样本自动做 8 重旋转/镜像数据增强

## 快速开始

```bash
# 安装依赖（已提供 .venv 时可跳过）
# GPU（CUDA 12.8）:
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
# 或纯 CPU:
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install fastapi "uvicorn[standard]" numpy requests

# 启动（有检查点时自动恢复并继续训练）
python main.py --port 8000
```

打开 http://127.0.0.1:8000

推荐流程：
1. 「训练控制」→ **从网上下载棋谱**（或运行 `python scripts/distill_full.py` 做全量蒸馏）
2. **用棋谱预训练** 1-3 轮，让 AI 学会基本棋形
3. **开始自我对弈训练**，持续提升；在「训练图表」观察损失/ELO/胜率曲线
4. 「人机对局」与 AI 对弈，「棋谱回放」研究 AI 的对局

## 项目结构

```
epsilonzero/
├── config.py              # 全局配置 (data/config.json 持久化)
├── game/board.py          # 五子棋规则：落子、胜负判定、对称变换、网络编码
├── model/transformer.py   # Transformer 策略-价值网络
├── mcts.py                # PUCT 蒙特卡洛树搜索 (+ 根节点战术强制)
├── selfplay.py            # 自我对弈生成 (+ D4 增强 / 白方补偿 / 认输)
├── replay_buffer.py       # 经验回放池
├── trainer.py             # 训练编排：自对弈→训练→评估→ELO→检查点（后台线程，断点续训）
├── heuristic.py           # 启发式引擎（评估基线 + 合成棋谱）
├── data/download_data.py  # Gomocup 棋谱下载与 psq 解析
└── web/
    ├── server.py          # FastAPI：REST + WebSocket
    └── static/            # 前端（原生 JS + Canvas 自绘图表，零外部依赖）
scripts/
└── distill_full.py        # 全量棋谱蒸馏（分块、内存安全）
```

## 技术说明

- **网络输入**：4 个平面（己方子、对方子、最后一手、行棋方指示——执黑全 1 执白全 0，让网络能学习黑白不同的策略），每点一个 token
- **MCTS**：PUCT 选择 + 叶节点网络评估 + 根节点 Dirichlet 噪声；根节点战术强制（必胜/必挡/防活四）；前 `temp_moves` 手按温度采样；白方模拟数 ×`white_sims_boost`
- **训练目标**：策略损失（与 MCTS 访问分布的交叉熵）+ 价值损失（对局结果 MSE）
- **评估**：每轮结束与随机策略、启发式引擎各下若干局（黑白轮换），更新 ELO
- **设备**：`device=auto` 自动选择 CUDA/CPU；GPU（如 RTX 5060）训练速度数倍于 CPU
- CPU 训练提示：速度与 `棋盘大小² × 模型规模 × MCTS 模拟数` 成正比；CPU 较弱时可先在「训练控制」里调小模型（如 d_model=64, n_layers=4）或棋盘（9×9/13×13）快速迭代
