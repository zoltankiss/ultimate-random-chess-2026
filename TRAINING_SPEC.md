# Ultimate Random Chess - AlphaZero Training Specification

## Overview

This document specifies everything needed to perform an AlphaGo-style self-play reinforcement learning training run for "Ultimate Random Chess."

---

## Game Rules: Ultimate Random Chess

### What Is It?

Ultimate Random Chess is a variant of Fischer Random Chess (Chess960) with even more randomization:
- **Fischer Random**: Back rank randomized from 960 preset positions
- **Ultimate Random**: Back rank randomized with ANY combination of standard chess pieces

### Starting Position Rules

#### Fixed Constraints
1. **Board**: Standard 8x8 chessboard
2. **Pawns**: 8 pawns on rank 2 (white) / rank 7 (black) - standard positions
3. **Back Rank Requirements**:
   - Exactly **1 King** (mandatory)
   - Exactly **2 Rooks** (mandatory for castling)
   - **5 other pieces**: randomly selected from the piece pool
4. **King Position**: Can be anywhere on the back rank (files a-h)
5. **Rook Positions**: One rook must be on each side of the king (for castling to work)

#### Piece Pool for Random Slots (5 pieces)
Standard pieces (can repeat):
- Queen (Q)
- Rook (R) - additional rooks beyond the mandatory 2
- Bishop (B)
- Knight (N)

**Note**: The mandatory 2 rooks + King fill 3 slots. The remaining 5 slots are filled randomly.

#### Example Valid Positions
```
R N B Q K B N R   (standard chess)
R Q B K N B Q R   (two queens, one less knight)
N N N N K R N R   (6 knights, no bishops/queen)
R R R K B R R R   (6 rooks, one bishop)
```

#### Invalid Positions
- No king
- Less than 2 rooks
- King on edge with no rook on one side (can't castle both ways)

### Setup Generation Algorithm

```python
def generate_starting_position():
    # 1. Place king randomly on files b-g (not corners, to allow castling both ways)
    king_file = random.randint(1, 6)  # b=1, g=6 (0-indexed: 1-6)

    # 2. Place rooks: one on each side of king
    left_rook_file = random.randint(0, king_file - 1)
    right_rook_file = random.randint(king_file + 1, 7)

    # 3. Fill remaining 5 squares with random pieces
    piece_pool = ['Q', 'R', 'B', 'N']  # can add custom pieces
    remaining_files = [f for f in range(8) if f not in [king_file, left_rook_file, right_rook_file]]

    position = [''] * 8
    position[king_file] = 'K'
    position[left_rook_file] = 'R'
    position[right_rook_file] = 'R'

    for file in remaining_files:
        position[file] = random.choice(piece_pool)

    return position
```

### Castling Rules (Fischer Random Style)

Castling works identically to Fischer Random Chess (Chess960):

#### Key Principle: Final Positions Are Always The Same

No matter where the King and Rooks start, after castling they end up in the **standard chess positions**:

| Castling Type | King Final | Rook Final |
|--------------|------------|------------|
| **Kingside (O-O)** | g1 (white) / g8 (black) | f1 (white) / f8 (black) |
| **Queenside (O-O-O)** | c1 (white) / c8 (black) | d1 (white) / d8 (black) |

#### Castling Requirements

1. **Neither piece has moved**: King and the castling rook haven't moved
2. **Not in check**: King is not currently in check
3. **Path safety**: King doesn't pass through or land on an attacked square
4. **Path clear**: All squares between king's start and end, and rook's start and end, are empty (except for the pieces themselves)

#### Special Cases (Valid in Fischer Random)

- King and rook may swap positions
- Only the king may move (if already on g1/c1 but rook isn't on f1/d1)
- Only the rook may move (if king is already on g1/c1)
- In extreme cases, the castling move might look like no movement at all

### Other Rules (Standard Chess)

- **Check**: King attacked = must escape
- **Checkmate**: No legal moves while in check = loss
- **Stalemate**: No legal moves, not in check = draw
- **50-move rule**: 50 moves without pawn move or capture = draw claimable
- **Threefold repetition**: Same position 3 times = draw claimable
- **Insufficient material**: K vs K, K+B vs K, K+N vs K = draw

### Pawn Rules

- **Movement**: Forward 1, or forward 2 from starting rank
- **Capture**: Diagonal forward only
- **En Passant**: After opponent's double-move, can capture as if single move
- **Promotion**: On reaching rank 8, must promote to Q, R, B, or N

---

## Neural Network Architecture

### Input Representation

Based on AlphaZero's approach, adapted for Ultimate Random Chess.

#### Board State Planes (8x8 each)

**Per-Position Planes (P = 14 for standard, more for custom pieces):**

| Planes | Description |
|--------|-------------|
| 6 | White pieces: P, N, B, R, Q, K |
| 6 | Black pieces: P, N, B, R, Q, K |
| 2 | Repetition counters (0, 1, 2+ times) |

**For Ultimate Random with custom pieces**, add 2 planes per custom piece type (white + black).

**Historical Planes:**
- Include T=8 time steps (current + 7 previous positions)
- Total: P × T planes

**Meta-Information Planes (scalar values broadcast to 8x8):**

| Planes | Description |
|--------|-------------|
| 1 | Side to move (all 1s = white, all 0s = black) |
| 1 | Total move count (normalized) |
| 1 | White kingside castling rights |
| 1 | White queenside castling rights |
| 1 | Black kingside castling rights |
| 1 | Black queenside castling rights |
| 1 | No-progress count (for 50-move rule, normalized) |

**Total Input Planes (standard pieces):**
- (14 planes × 8 timesteps) + 7 meta = **119 planes** (matches AlphaZero)

### Output Representation

#### Policy Head: Move Encoding

AlphaZero uses 73 planes of 8x8 = 4,672 possible moves:

**Move Types (73 planes):**

| Planes | Description |
|--------|-------------|
| 56 | Queen-type moves: 7 distances × 8 directions |
| 8 | Knight moves: 8 possible jumps |
| 9 | Pawn underpromotions: 3 directions × 3 piece types |

Each move is encoded as (from_square, move_type) where:
- from_square: 64 squares (8×8)
- move_type: 73 possible move types

**For Ultimate Random Chess:**
- Same encoding works for standard pieces
- Custom pieces (leapers, hoppers) need additional planes
- Nightrider: Use queen-direction encoding
- Grasshopper/Cannon: May need hopper-specific planes

#### Value Head

Single scalar output: tanh activation → [-1, 1]
- +1 = white wins
- -1 = black wins
- 0 = draw

---

## Action Space

### Legal Move Generation

For each piece type, generate all legal moves:

1. **Slides** (Q, R, B, custom sliders):
   - For each direction, extend until blocked or edge

2. **Leaps** (N, custom leapers):
   - Check each vector offset, if on board and not blocked by own piece

3. **Hoppers** (Grasshopper, Cannon):
   - Find hurdle, then landing squares

4. **Pawns**: Forward moves, captures, en passant, promotions

5. **King**: Normal moves + castling

6. **Filter for legality**: Remove moves that leave king in check

### Move Notation

Use algebraic notation with disambiguation:
- Standard: `e4`, `Nf3`, `O-O`
- From-to for training: `e2e4`, `g1f3`, `e1g1` (castling)

---

## Training Procedure

### Self-Play Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Games per iteration | 25,000 | Adjust for GPU budget |
| MCTS simulations per move | 800 | AlphaZero used 800 |
| Temperature | τ=1 (first 30 moves), τ→0 (later) | Exploration vs exploitation |
| Dirichlet noise | α=0.3, ε=0.25 | Root exploration |
| c_puct | 1.25-2.5 | Exploration constant |
| Virtual loss | 3 | For parallel MCTS |

### Neural Network Training

| Parameter | Value |
|-----------|-------|
| Batch size | 4,096 |
| Learning rate | 0.2 → 0.02 → 0.002 (stepped) |
| Momentum | 0.9 |
| Weight decay | 1e-4 |
| Training steps | 700,000 |

### Network Architecture

| Component | Specification |
|-----------|---------------|
| Input | 119 × 8 × 8 (or more for custom pieces) |
| Body | 19 residual blocks |
| Filters | 256 per layer |
| Kernel size | 3×3, stride 1 |
| Policy head | Conv → 73 × 8 × 8 → softmax |
| Value head | Conv(1) → Dense(256) → Dense(1) → tanh |

### Training Loop

```
for iteration in range(max_iterations):
    # 1. Self-play phase
    games = []
    for _ in range(games_per_iteration):
        game = play_game(network, mcts_config)
        games.append(game)

    # 2. Sample training data
    # From recent games, sample (state, policy, value) tuples
    training_data = sample_from_replay_buffer(games, sample_size=1_000_000)

    # 3. Train network
    for batch in batches(training_data):
        loss = policy_loss(batch) + value_loss(batch) + l2_reg
        optimizer.step(loss)

    # 4. Evaluate (optional)
    if iteration % eval_interval == 0:
        win_rate = evaluate_vs_previous(network, previous_network)
        if win_rate > 0.55:
            previous_network = network.copy()
```

---

## GPU Rental Considerations

### Current GPU Prices (December 2025)

| GPU | Provider | Price/hr | Notes |
|-----|----------|----------|-------|
| **H100 80GB** | Vast.ai | $1.87 | Cheapest H100 |
| **H100 80GB** | RunPod | $1.99-2.39 | Community/Secure cloud |
| **H100 80GB** | Lambda Labs | $2.49-2.99 | Reliable, easy setup |
| **A100 80GB** | Thunder Compute | $0.78 | Best value for training |
| **A100 40GB** | Thunder Compute | $0.66 | Budget option |
| **A100 80GB** | RunPod | ~$1.00 | Good middle ground |

### Compute Requirements & Cost Estimates

| Level | GPU Hours | A100 Cost | H100 Cost | Expected Result |
|-------|-----------|-----------|-----------|-----------------|
| **Quick test** | 50-100 | $40-80 | $100-200 | Proof of concept, barely plays |
| **Decent** | 500-1000 | $400-800 | $1,000-2,000 | "Strong beginner" level |
| **Strong** | 2000-5000 | $1,600-4,000 | $4,000-10,000 | Actually competitive |
| **AlphaZero-scale** | 10,000+ | $8,000+ | $20,000+ | Near-optimal play |

### Recommended Budget Tiers

#### Tier 1: Proof of Concept ($50-100)
- 50 GPU hours on A100 @ $0.78/hr = ~$40
- Small network (10 blocks, 128 filters)
- 400 MCTS sims, ~10k games
- **Goal**: Verify training loop works

#### Tier 2: Playable AI ($500-800)
- 500-1000 GPU hours on A100
- Medium network (15 blocks, 192 filters)
- 600 MCTS sims, ~100k games
- **Goal**: Beats random, shows real strategy

#### Tier 3: Strong Beginner ($2,000-4,000)
- 2000-5000 GPU hours on A100
- Full network (19 blocks, 256 filters)
- 800 MCTS sims, ~500k games
- **Goal**: Plays at strong amateur level

### Recommended Setup

1. **Self-play workers**: Multiple cheap GPUs (or CPUs with small networks)
2. **Training**: Single powerful GPU (A100, H100)
3. **Parallelization**: Ray, Dask, or custom distributed setup

### Cost Optimization Tips

- Start with smaller network (10 residual blocks, 128 filters)
- Use fewer MCTS simulations (400 instead of 800)
- Train on fewer positions per iteration
- Use mixed precision (FP16) training
- Use A100 instead of H100 (similar perf, ~60% cheaper)
- Spot/preemptible instances can save 50-70%

### Provider Recommendations

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Quick experiments | Vast.ai / RunPod | Cheapest, instant spin-up |
| Serious training | Lambda Labs | Reliable, good tooling |
| Maximum budget | Thunder Compute | A100 @ $0.66-0.78/hr |
| Enterprise/team | AWS/GCP | SLAs, but 2-3x more expensive |

---

## Implementation Checklist

### Core Game Engine
- [ ] Board representation (bitboards or array)
- [ ] Move generation for all piece types
- [ ] Legal move filtering (check detection)
- [ ] Game state (castling rights, en passant, 50-move counter)
- [ ] Position hashing (Zobrist hashing for repetition)
- [ ] FEN/position serialization

### Neural Network
- [ ] Input encoder (position → tensor)
- [ ] ResNet architecture
- [ ] Policy head with move masking
- [ ] Value head
- [ ] Model serialization/loading

### MCTS
- [ ] Node structure (visit count, value sum, prior)
- [ ] UCB/PUCT selection
- [ ] Virtual loss for parallel search
- [ ] Dirichlet noise at root
- [ ] Temperature-based move selection

### Training Infrastructure
- [ ] Self-play worker
- [ ] Replay buffer
- [ ] Training loop
- [ ] Checkpointing
- [ ] Tensorboard/logging
- [ ] Distributed coordination (if multi-GPU)

### Evaluation
- [ ] Self-play evaluation
- [ ] ELO estimation
- [ ] vs. random baseline
- [ ] Opening diversity metrics

---

## Appendix: Piece Definitions Reference

See `pieces/standard.yaml` and `pieces/custom.yaml` for complete piece definitions.

### Standard Pieces for Training

| Piece | Symbol | Value | Movement |
|-------|--------|-------|----------|
| King | K | ∞ | 1 square any direction |
| Queen | Q | 9 | Slide any direction |
| Rook | R | 5 | Slide orthogonally |
| Bishop | B | 3 | Slide diagonally |
| Knight | N | 3 | Leap (2,1) |
| Pawn | P | 1 | Forward + special rules |

---

## Sources

- [AlphaZero - Chessprogramming Wiki](https://www.chessprogramming.org/AlphaZero)
- [AlphaZero Paper (Science)](https://www.science.org/doi/10.1126/science.aar6404)
- [Chess960 Castling Rules](https://www.chess.com/article/view/how-to-castle-in-fischer-random-chess)
- [Chess.com Chess960 Reference](https://www.chess.com/terms/chess960)
- [Leela Chess Zero Neural Network Topology](https://lczero.org/dev/backend/nn/)
