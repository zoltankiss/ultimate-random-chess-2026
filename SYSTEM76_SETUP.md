# System76 Linux Training Setup

Parallelized AlphaZero training for Ultimate Random Chess on Linux with NVIDIA GPU.

## Quick Start

```bash
# Clone
git clone https://github.com/zoltankiss/ultimate-random-chess-2026.git
cd ultimate-random-chess-2026

# Check GPU
nvidia-smi

# Setup Python
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA (adjust cu121 to match your CUDA version from nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy tqdm pyyaml anthropic

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Train
cd src
python train.py --iterations 1000 --games 50 --sims 100 --checkpoint-interval 25
```

## Copy Checkpoint from Mac (to continue training)

```bash
# On Mac, find IP:
ipconfig getifaddr en0

# On System76:
mkdir -p src/checkpoints
scp zoltankiss@<MAC_IP>:~/Projects/ai-experiments/ultimate-random-chess-2026/src/checkpoints/final.pt ./src/checkpoints/

# Continue training from checkpoint
cd src
python train.py --iterations 1000 --games 50 --sims 100 --checkpoint ../checkpoints/final.pt
```

## Training Parameters

| Param | Mac M2 Max | System76 (GPU) | Notes |
|-------|------------|----------------|-------|
| `--games` | 20 | 50-100 | Games per iteration |
| `--sims` | 50 | 100-200 | MCTS simulations (higher = better quality) |
| `--iterations` | 100 | 1000+ | Total training iterations |
| `--checkpoint-interval` | 25 | 25-50 | Save every N iterations |

## Current Training Stats (Mac M2 Max baseline)

- 20 games/iteration: ~8 minutes
- Network: 12.8M parameters
- Buffer fills at 50k positions
- Iteration 130+: Loss ~2.7 (policy ~2.7, value ~0.0004)

## Parallelization (TODO)

The main bottleneck is sequential self-play in `train.py`. To parallelize:

1. Modify `Trainer.run_self_play()` to use `multiprocessing.Pool`
2. Run 4-8 games in parallel (depending on GPU memory)
3. Expected speedup: 4-8x

```python
# Example parallel self-play (add to train.py)
from multiprocessing import Pool

def play_game_worker(args):
    network_state, mcts_config, device = args
    # Load network from state_dict, play game, return data
    ...

# In run_self_play:
with Pool(processes=6) as pool:
    games_data = pool.map(play_game_worker, [args] * num_games)
```

## Project Structure

```
src/
├── game.py          # Game logic, 14 piece types, move generation
├── network.py       # ChessNetwork (12.8M params), board encoding
├── mcts.py          # Monte Carlo Tree Search
├── train.py         # AlphaZero training loop
├── pgn.py           # PGN game logging
├── claude_qa_player.py  # QA evaluation using Claude API
└── audit.py         # Move validation auditing

pieces/
├── standard.yaml    # King, Queen, Rook, Bishop, Knight, Pawn
└── custom.yaml      # Archbishop, Chancellor, Amazon, Camel, Zebra, Grasshopper, Cannon
```

## Piece Types

Standard (6): King, Queen, Rook, Bishop, Knight, Pawn
Fairy (8): Archbishop (B+N), Chancellor (R+N), Amazon (Q+N), Camel (3,1 leaper), Zebra (3,2 leaper), Grasshopper (hops over pieces), Cannon (captures by hopping), Nightrider (repeated knight moves)

## Goal

Beat Magnus Carlsen at Ultimate Random Chess. Estimated: 1-3 years of training.
