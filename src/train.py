"""
Ultimate Random Chess - Self-Play Training

AlphaZero-style training loop. Supports MPS (Mac), CUDA (NVIDIA), and CPU.
"""

import os
import time
import random
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from game import GameState, create_starting_position, generate_legal_moves, get_game_result, make_move, print_board, Color
from network import ChessNetwork, NetworkConfig, encode_board, move_to_policy_index, get_policy_mask, get_device
from mcts import MCTS, MCTSConfig, play_game
from parallel_mcts import ParallelSelfPlay
from pgn import PGNWriter


class TrainingConfig:
    """Training hyperparameters."""

    # Self-play
    games_per_iteration: int = 25       # Games per training iteration
    mcts_simulations: int = 50          # MCTS sims per move (reduced for M2)

    # Training
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    training_steps_per_iteration: int = 100

    # Replay buffer
    replay_buffer_size: int = 50000     # Max positions to store
    min_buffer_size: int = 1000         # Min positions before training

    # Evaluation
    eval_games: int = 10                # Games for evaluation
    eval_interval: int = 5              # Evaluate every N iterations

    # Checkpointing
    checkpoint_interval: int = 10       # Save every N iterations
    checkpoint_dir: str = "checkpoints"

    # PGN logging
    pgn_dir: str = "training-runs"      # Relative path (works on cloud and local)
    save_pgn: bool = True               # Save all games as PGN files

    # General
    num_iterations: int = 100           # Total training iterations
    device: str = "auto"                # "auto", "mps", "cuda", "cpu"

    # Parallelization
    num_workers: int = 8                # Parallel self-play games (batched on GPU)


class ReplayBuffer:
    """Store game positions for training."""

    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add_game(self, game_data: List[Tuple[GameState, np.ndarray, float]]):
        """Add positions from a game to the buffer."""
        for state, policy, value in game_data:
            encoded = encode_board(state)
            self.buffer.append((encoded, policy, value))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of positions."""
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        samples = [self.buffer[i] for i in indices]

        boards = np.array([s[0] for s in samples])
        policies = np.array([s[1] for s in samples])
        values = np.array([s[2] for s in samples]).reshape(-1, 1)

        return boards, policies, values

    def __len__(self):
        return len(self.buffer)


class ChessDataset(Dataset):
    """PyTorch dataset for training."""

    def __init__(self, boards: np.ndarray, policies: np.ndarray, values: np.ndarray):
        self.boards = torch.tensor(boards, dtype=torch.float32)
        self.policies = torch.tensor(policies, dtype=torch.float32)
        self.values = torch.tensor(values, dtype=torch.float32)

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return self.boards[idx], self.policies[idx], self.values[idx]


class Trainer:
    """AlphaZero-style trainer."""

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()

        # Device
        if self.config.device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(self.config.device)

        print(f"Using device: {self.device}")

        # Network
        self.network = ChessNetwork().to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)

        # Stats
        self.iteration = 0
        self.total_games = 0
        self.total_positions = 0

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(exist_ok=True)

        # PGN writer
        self.pgn_writer = None
        if self.config.save_pgn:
            self.pgn_writer = PGNWriter(self.config.pgn_dir)

    def self_play_game(self, return_moves: bool = False):
        """Play a single game and return training data."""
        mcts_config = MCTSConfig(num_simulations=self.config.mcts_simulations)
        return play_game(self.network, mcts_config, self.device, verbose=False, return_moves=return_moves)

    def run_self_play(self) -> int:
        """Run self-play games and add to replay buffer using parallel execution."""
        self.network.eval()

        mcts_config = MCTSConfig(num_simulations=self.config.mcts_simulations)
        parallel_player = ParallelSelfPlay(
            self.network,
            mcts_config,
            self.device,
            num_parallel=self.config.num_workers
        )

        print(f"Playing {self.config.games_per_iteration} self-play games ({self.config.num_workers} parallel)...")

        # Play all games with parallel batched inference
        games_results = []
        games_played = 0
        total_positions = 0

        with tqdm(total=self.config.games_per_iteration, desc="Self-play") as pbar:
            while games_played < self.config.games_per_iteration:
                batch_size = min(
                    self.config.num_workers,
                    self.config.games_per_iteration - games_played
                )

                batch_results = parallel_player.play_games(
                    batch_size,
                    verbose=False,
                    return_moves=bool(self.pgn_writer)
                )

                games_results.extend(batch_results)
                games_played += batch_size
                pbar.update(batch_size)

        # Process results
        for result in games_results:
            if self.pgn_writer:
                training_data, move_history, game_result = result
                self.replay_buffer.add_game(training_data)
                total_positions += len(training_data)

                # Save PGN
                self.pgn_writer.save_game(
                    move_history, game_result,
                    iteration=self.iteration,
                    game_num=self.total_games + 1,
                    metadata={'MCTSSimulations': str(self.config.mcts_simulations)}
                )
            else:
                training_data = result
                self.replay_buffer.add_game(training_data)
                total_positions += len(training_data)

            self.total_games += 1

        self.total_positions += total_positions

        return total_positions

    def train_step(self, batch_size: int) -> Tuple[float, float, float]:
        """Run a single training step."""
        self.network.train()

        # Sample from replay buffer
        boards, policies, values = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        boards = torch.tensor(boards, dtype=torch.float32).to(self.device)
        target_policies = torch.tensor(policies, dtype=torch.float32).to(self.device)
        target_values = torch.tensor(values, dtype=torch.float32).to(self.device)

        # Forward pass
        pred_policies, pred_values = self.network(boards)

        # Policy loss (cross-entropy)
        # Only compute loss for legal moves (target policies are already masked)
        policy_loss = -torch.sum(target_policies * torch.log_softmax(pred_policies, dim=1)) / boards.size(0)

        # Value loss (MSE)
        value_loss = torch.mean((pred_values - target_values) ** 2)

        # Total loss
        total_loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), policy_loss.item(), value_loss.item()

    def train_iteration(self) -> Tuple[float, float, float]:
        """Run training steps for one iteration."""
        if len(self.replay_buffer) < self.config.min_buffer_size:
            print(f"Buffer too small ({len(self.replay_buffer)}/{self.config.min_buffer_size}), skipping training")
            return 0.0, 0.0, 0.0

        total_loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0

        for _ in tqdm(range(self.config.training_steps_per_iteration), desc="Training"):
            tl, pl, vl = self.train_step(self.config.batch_size)
            total_loss += tl
            policy_loss += pl
            value_loss += vl

        n = self.config.training_steps_per_iteration
        return total_loss / n, policy_loss / n, value_loss / n

    def evaluate(self) -> Tuple[float, float, float]:
        """Evaluate the network by playing games against itself."""
        self.network.eval()

        results = []
        game_lengths = []

        mcts_config = MCTSConfig(num_simulations=self.config.mcts_simulations // 2)

        print(f"Running {self.config.eval_games} evaluation games...")

        for _ in tqdm(range(self.config.eval_games), desc="Evaluation"):
            game_data = play_game(self.network, mcts_config, self.device, verbose=False)
            game_lengths.append(len(game_data))

            if game_data:
                final_value = game_data[-1][2]
                results.append(final_value)

        # Calculate stats
        white_wins = sum(1 for r in results if r > 0.5)
        black_wins = sum(1 for r in results if r < -0.5)
        draws = len(results) - white_wins - black_wins

        avg_length = np.mean(game_lengths) if game_lengths else 0

        return white_wins / len(results), black_wins / len(results), avg_length

    def save_checkpoint(self, path: str = None):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_{self.iteration:04d}.pt"
            )

        torch.save({
            'iteration': self.iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_games': self.total_games,
            'total_positions': self.total_positions,
        }, path)

        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint['iteration']
        self.total_games = checkpoint.get('total_games', 0)
        self.total_positions = checkpoint.get('total_positions', 0)

        print(f"Loaded checkpoint from {path}")
        print(f"Iteration: {self.iteration}, Games: {self.total_games}, Positions: {self.total_positions}")

    def train(self, num_iterations: int = None):
        """Main training loop."""
        if num_iterations is None:
            num_iterations = self.config.num_iterations

        print("="*60)
        print("Ultimate Random Chess - AlphaZero Training")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
        print(f"Iterations: {num_iterations}")
        print(f"Games per iteration: {self.config.games_per_iteration}")
        print(f"MCTS simulations: {self.config.mcts_simulations}")
        print(f"Parallel workers: {self.config.num_workers}")

        # Initialize PGN writer for this run
        if self.pgn_writer:
            run_dir = self.pgn_writer.start_run()
            print(f"PGN games will be saved to: {run_dir}")

        print("="*60)

        start_time = time.time()

        for iteration in range(self.iteration, self.iteration + num_iterations):
            self.iteration = iteration
            iter_start = time.time()

            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{self.iteration + num_iterations}")
            print(f"{'='*60}")

            # Self-play
            positions = self.run_self_play()
            print(f"Generated {positions} positions ({len(self.replay_buffer)} in buffer)")

            # Training
            total_loss, policy_loss, value_loss = self.train_iteration()
            if total_loss > 0:
                print(f"Loss: {total_loss:.4f} (policy: {policy_loss:.4f}, value: {value_loss:.4f})")

            # Evaluation
            if (iteration + 1) % self.config.eval_interval == 0:
                white_wr, black_wr, avg_len = self.evaluate()
                print(f"Eval: White wins {white_wr:.1%}, Black wins {black_wr:.1%}, Avg length {avg_len:.1f}")

            # Checkpoint
            if (iteration + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint()

            iter_time = time.time() - iter_start
            print(f"Iteration time: {iter_time:.1f}s")

        # Final save
        self.save_checkpoint(os.path.join(self.config.checkpoint_dir, "final.pt"))

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Total games: {self.total_games}")
        print(f"Total positions: {self.total_positions}")
        print(f"{'='*60}")


def play_vs_human(checkpoint_path: str = None):
    """Play a game against the trained model."""
    from game import create_starting_position, print_board, generate_legal_moves, make_move, get_game_result

    device = get_device()
    network = ChessNetwork().to(device)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        network.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")

    mcts_config = MCTSConfig(num_simulations=100)
    mcts = MCTS(network, mcts_config, device)

    state = create_starting_position()
    history = []

    print("\n=== Ultimate Random Chess ===")
    print("Enter moves in format like 'e2e4' or 'g1f3'")
    print("Type 'quit' to exit")
    print()

    while True:
        print_board(state)

        result = get_game_result(state)
        if result is not None:
            if result > 0:
                print("White wins!")
            elif result < 0:
                print("Black wins!")
            else:
                print("Draw!")
            break

        if state.turn == Color.WHITE:
            # Human plays white
            legal_moves = generate_legal_moves(state)
            print(f"Legal moves: {', '.join(str(m) for m in legal_moves[:10])}...")

            while True:
                move_str = input("Your move: ").strip().lower()

                if move_str == 'quit':
                    return

                # Find matching move
                found = None
                for move in legal_moves:
                    if str(move) == move_str:
                        found = move
                        break

                if found:
                    history = [state] + history[:7]
                    state = make_move(state, found)
                    break
                else:
                    print("Invalid move, try again")

        else:
            # AI plays black
            print("AI thinking...")
            move, _ = mcts.select_move(state, history, temperature=0.1)
            print(f"AI plays: {move}")

            history = [state] + history[:7]
            state = make_move(state, move)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ultimate Random Chess Training")
    parser.add_argument("--play", action="store_true", help="Play against trained model")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to load")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--games", type=int, default=10, help="Games per iteration")
    parser.add_argument("--sims", type=int, default=25, help="MCTS simulations per move")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N iterations")
    parser.add_argument("--workers", type=int, default=8, help="Parallel self-play games (batched on GPU)")

    args = parser.parse_args()

    if args.play:
        checkpoint = args.checkpoint or "checkpoints/final.pt"
        if os.path.exists(checkpoint):
            play_vs_human(checkpoint)
        else:
            print(f"Checkpoint not found: {checkpoint}")
            print("Training a new model first...")
            args.play = False

    if not args.play:
        config = TrainingConfig()
        config.num_iterations = args.iterations
        config.games_per_iteration = args.games
        config.mcts_simulations = args.sims
        config.checkpoint_interval = args.checkpoint_interval
        config.num_workers = args.workers

        trainer = Trainer(config)

        if args.checkpoint and os.path.exists(args.checkpoint):
            trainer.load_checkpoint(args.checkpoint)

        trainer.train()
