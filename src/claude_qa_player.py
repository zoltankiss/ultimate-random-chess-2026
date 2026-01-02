"""
Ultimate Random Chess - Claude QA Player

This script uses Claude API to play against a trained checkpoint,
evaluating the checkpoint's strength and providing QA analysis.
"""

import os
import sys
import json
import time
import torch
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import anthropic

from game import (
    GameState, Move, Color, Piece,
    create_starting_position, print_board, generate_legal_moves,
    make_move, get_game_result, get_piece_type, get_piece_color,
    file_of, rank_of, sq, PIECE_SYMBOLS, find_king, is_in_check
)
from network import ChessNetwork, get_device
from mcts import MCTS, MCTSConfig


# Piece movement descriptions for Claude's context
PIECE_MOVEMENT_RULES = """
## Ultimate Random Chess - Piece Movement Rules

### Standard Pieces:
- **P (Pawn)**: Moves forward 1 square (2 from start), captures diagonally. Promotes on last rank.
- **N (Knight)**: Jumps in L-shape: 2 squares in one direction + 1 perpendicular. Can jump over pieces.
- **B (Bishop)**: Slides diagonally any distance. Cannot jump.
- **R (Rook)**: Slides horizontally/vertically any distance. Cannot jump.
- **Q (Queen)**: Slides in any direction (diagonal + orthogonal). Cannot jump.
- **K (King)**: Moves 1 square in any direction. Can castle with Rooks.

### Fairy Pieces (Ultimate Random Chess):
- **A (Archbishop)**: Combines Knight + Bishop. Can move like either piece.
- **C (Chancellor)**: Combines Knight + Rook. Can move like either piece.
- **M (Amazon)**: Combines Knight + Queen. Most powerful piece! Can move like either.
- **L (Camel)**: Leaper that jumps (3,1) - like a Knight but 3 squares + 1 square instead of 2+1.
- **Z (Zebra)**: Leaper that jumps (3,2) - 3 squares + 2 squares perpendicular.
- **G (Grasshopper)**: Moves on Queen lines but MUST hop over exactly one piece, landing immediately after.
- **X (Cannon)**: Xiangqi cannon. Slides like Rook for non-captures. To capture, must hop over exactly one "screen" piece.

### Important Notes:
- Uppercase = White pieces, lowercase = Black pieces
- The King (K/k) can capture adjacent pieces by moving onto their square
- Be careful not to leave pieces where the enemy King can capture them!
- Archbishop attacks DIAGONALLY and via KNIGHT JUMPS, not orthogonally adjacent squares
"""


def format_board_for_claude(state: GameState) -> str:
    """Format the board state in a clear way for Claude."""
    lines = []
    lines.append("```")
    lines.append("  a b c d e f g h")
    for rank in range(7, -1, -1):
        row = f"{rank + 1} "
        for file in range(8):
            piece = state.board[sq(file, rank)]
            if piece == 0:
                row += ". "
            else:
                piece_type = get_piece_type(piece)
                color = get_piece_color(piece)
                symbol = PIECE_SYMBOLS[piece_type]
                if color == Color.BLACK:
                    symbol = symbol.lower()
                row += f"{symbol} "
        lines.append(row)
    lines.append("  a b c d e f g h")
    lines.append("```")
    return "\n".join(lines)


def get_material_count(state: GameState) -> Dict[str, List[str]]:
    """Count material for both sides."""
    white_pieces = []
    black_pieces = []

    for sq_idx in range(64):
        piece = state.board[sq_idx]
        if piece == 0:
            continue
        piece_type = get_piece_type(piece)
        color = get_piece_color(piece)
        symbol = PIECE_SYMBOLS[piece_type]

        if color == Color.WHITE:
            white_pieces.append(symbol)
        else:
            black_pieces.append(symbol)

    return {"white": white_pieces, "black": black_pieces}


def describe_position(state: GameState) -> str:
    """Generate a detailed position description."""
    material = get_material_count(state)

    in_check = is_in_check(state, state.turn)
    check_status = " **IN CHECK!**" if in_check else ""

    description = f"""
### Position Analysis

**Turn**: {'White' if state.turn == Color.WHITE else 'Black'}{check_status}
**Move Number**: {state.fullmove_number}
**Halfmove Clock**: {state.halfmove_clock} (50-move rule at 100)

**Material**:
- White: {' '.join(sorted(material['white'], key=lambda x: 'QRACMGXNBZLPK'.find(x)))} ({len(material['white'])} pieces)
- Black: {' '.join(sorted(material['black'], key=lambda x: 'QRACMGXNBZLPK'.find(x)))} ({len(material['black'])} pieces)
"""
    return description


def format_legal_moves(state: GameState, legal_moves: List[Move]) -> str:
    """Format legal moves in a clear way for Claude."""
    if not legal_moves:
        return "No legal moves available."

    # Group by piece type
    moves_by_piece = {}

    for move in legal_moves:
        piece = state.board[move.from_sq]
        piece_type = get_piece_type(piece)
        symbol = PIECE_SYMBOLS[piece_type]

        if symbol not in moves_by_piece:
            moves_by_piece[symbol] = []

        move_str = str(move)
        # Add capture indicator
        if state.board[move.to_sq] != 0:
            target = state.board[move.to_sq]
            target_symbol = PIECE_SYMBOLS[get_piece_type(target)]
            if get_piece_color(target) == Color.BLACK:
                target_symbol = target_symbol.lower()
            move_str += f" (captures {target_symbol})"

        moves_by_piece[symbol].append(move_str)

    lines = ["### Legal Moves (by piece type):"]
    for symbol in sorted(moves_by_piece.keys()):
        moves = moves_by_piece[symbol]
        lines.append(f"**{symbol}**: {', '.join(moves)}")

    lines.append(f"\n**Total: {len(legal_moves)} legal moves**")
    return "\n".join(lines)


def create_claude_prompt(
    state: GameState,
    legal_moves: List[Move],
    move_history: List[str],
    game_context: str
) -> str:
    """Create a comprehensive prompt for Claude to select a move."""

    board_str = format_board_for_claude(state)
    position_desc = describe_position(state)
    moves_str = format_legal_moves(state, legal_moves)

    history_str = ""
    if move_history:
        # Show last 20 moves with move numbers
        recent = move_history[-40:]  # Last 20 full moves
        history_lines = []
        for i, move in enumerate(recent):
            if i % 2 == 0:
                move_num = (len(move_history) - len(recent) + i) // 2 + 1
                history_lines.append(f"{move_num}. {move}")
            else:
                history_lines[-1] += f" {move}"
        history_str = f"\n### Recent Moves:\n{' | '.join(history_lines[-10:])}"

    prompt = f"""You are playing Ultimate Random Chess as {'White' if state.turn == Color.WHITE else 'Black'}.

{PIECE_MOVEMENT_RULES}

{game_context}

## Current Board Position:
{board_str}

{position_desc}
{history_str}

{moves_str}

## Your Task:
Analyze the position and select the BEST legal move. Consider:
1. Material safety - don't leave pieces where they can be captured (especially by the King!)
2. King safety - don't expose your King to attacks
3. Tactics - look for captures, forks, pins, and threats
4. Development - control the center and activate pieces
5. Piece coordination - make pieces work together

**CRITICAL**: The enemy King can capture adjacent pieces! Check if your move puts a piece next to the enemy King without protection.

**RESPOND WITH ONLY**:
1. Your chosen move in coordinate notation (e.g., "e2e4")
2. A brief explanation (1-2 sentences)

Format your response as:
MOVE: <move>
REASON: <brief explanation>
"""
    return prompt


class ClaudePlayer:
    """Claude-based chess player using the Anthropic API."""

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514"):
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            # Will use ANTHROPIC_API_KEY environment variable
            self.client = anthropic.Anthropic()
        self.model = model
        self.move_history = []
        self.reasoning_history = []
        self.illegal_move_attempts = 0
        self.total_moves = 0

    def select_move(
        self,
        state: GameState,
        legal_moves: List[Move],
        game_context: str = ""
    ) -> Tuple[Move, str]:
        """Select a move using Claude API."""

        prompt = create_claude_prompt(
            state, legal_moves, self.move_history, game_context
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )

                response_text = response.content[0].text.strip()

                # Parse response
                move_str = None
                reason = ""

                for line in response_text.split("\n"):
                    line = line.strip()
                    if line.upper().startswith("MOVE:"):
                        move_str = line.split(":", 1)[1].strip().lower()
                    elif line.upper().startswith("REASON:"):
                        reason = line.split(":", 1)[1].strip()

                if not move_str:
                    # Try to find a move pattern in the response
                    import re
                    match = re.search(r'\b([a-h][1-8][a-h][1-8][qrbnQRBN]?)\b', response_text)
                    if match:
                        move_str = match.group(1).lower()

                if not move_str:
                    print(f"  [Retry {attempt+1}] Could not parse move from response")
                    continue

                # Find matching legal move
                for move in legal_moves:
                    if str(move).lower() == move_str:
                        self.total_moves += 1
                        self.move_history.append(str(move))
                        self.reasoning_history.append(reason)
                        return move, reason

                # Move not found - illegal move attempt
                self.illegal_move_attempts += 1
                print(f"  [Retry {attempt+1}] Illegal move: {move_str}")

                # Add hint about legal moves to retry
                if attempt < max_retries - 1:
                    prompt += f"\n\n**Your previous response '{move_str}' was illegal. Choose from the legal moves listed above.**"

            except Exception as e:
                print(f"  [Retry {attempt+1}] API error: {e}")
                time.sleep(1)

        # Fallback: random legal move
        import random
        fallback = random.choice(legal_moves)
        self.total_moves += 1
        self.move_history.append(str(fallback))
        self.reasoning_history.append("(fallback - could not parse Claude response)")
        print(f"  [Warning] Using fallback move: {fallback}")
        return fallback, "(fallback)"


class AIPlayer:
    """MCTS-based player using the trained checkpoint."""

    def __init__(self, checkpoint_path: str, mcts_sims: int = 100):
        self.device = get_device()
        self.network = ChessNetwork().to(self.device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.iteration = checkpoint.get('iteration', 0)

        self.mcts_config = MCTSConfig(num_simulations=mcts_sims)
        self.mcts = MCTS(self.network, self.mcts_config, self.device)

        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Training iteration: {self.iteration}")

    def select_move(self, state: GameState, history: List[GameState]) -> Move:
        """Select a move using MCTS."""
        move, _ = self.mcts.select_move(state, history, temperature=0.1)
        return move


def play_game(
    claude_player: ClaudePlayer,
    ai_player: AIPlayer,
    claude_color: Color = Color.WHITE,
    verbose: bool = True
) -> Dict:
    """Play a single game between Claude and the AI."""

    state = create_starting_position()
    history = []
    move_log = []

    game_context = f"""
## Game Context:
- You are playing against a neural network trained with AlphaZero-style self-play
- This is game evaluation to test if the training is working
- The AI opponent is at training iteration {ai_player.iteration}
- Play your best chess - we want to honestly evaluate the AI's strength
"""

    if verbose:
        print("\n" + "="*60)
        print("GAME START")
        print("="*60)
        print(f"Claude plays: {'White' if claude_color == Color.WHITE else 'Black'}")
        print(f"AI at iteration: {ai_player.iteration}")
        print_board(state)

    move_number = 0

    while True:
        result = get_game_result(state)
        if result is not None:
            break

        legal_moves = generate_legal_moves(state)
        if not legal_moves:
            break

        is_claude_turn = state.turn == claude_color

        if is_claude_turn:
            # Claude's turn
            if verbose:
                print(f"\n--- Move {state.fullmove_number}: Claude ({'White' if state.turn == Color.WHITE else 'Black'}) thinking...")

            move, reason = claude_player.select_move(state, legal_moves, game_context)

            if verbose:
                print(f"  Claude plays: {move}")
                print(f"  Reason: {reason}")
        else:
            # AI's turn
            if verbose:
                print(f"\n--- Move {state.fullmove_number}: AI ({'White' if state.turn == Color.WHITE else 'Black'}) thinking...")

            move = ai_player.select_move(state, history)
            reason = "(AI/MCTS)"

            if verbose:
                print(f"  AI plays: {move}")

        # Record move
        move_log.append({
            "move_number": state.fullmove_number,
            "player": "claude" if is_claude_turn else "ai",
            "color": "white" if state.turn == Color.WHITE else "black",
            "move": str(move),
            "reason": reason
        })

        # Make move
        history = [state] + history[:7]
        state = make_move(state, move)
        move_number += 1

        if verbose:
            print_board(state)

        # Safety limit
        if move_number > 200:
            result = 0.0
            break

    # Determine winner
    if result is None:
        result = 0.0

    if result > 0:
        winner = "white"
    elif result < 0:
        winner = "black"
    else:
        winner = "draw"

    claude_result = "win" if (winner == "white" and claude_color == Color.WHITE) or \
                            (winner == "black" and claude_color == Color.BLACK) else \
                   "loss" if winner != "draw" else "draw"

    if verbose:
        print("\n" + "="*60)
        print("GAME OVER")
        print(f"Result: {winner.upper()}")
        print(f"Claude result: {claude_result}")
        print(f"Total moves: {move_number}")
        print("="*60)

    return {
        "result": result,
        "winner": winner,
        "claude_result": claude_result,
        "claude_color": "white" if claude_color == Color.WHITE else "black",
        "total_moves": move_number,
        "move_log": move_log,
        "illegal_attempts": claude_player.illegal_move_attempts,
        "ai_iteration": ai_player.iteration
    }


def run_qa_evaluation(
    checkpoint_path: str,
    num_games: int = 4,
    mcts_sims: int = 100,
    verbose: bool = True,
    api_key: str = None
) -> Dict:
    """Run full QA evaluation."""

    print("="*60)
    print("ULTIMATE RANDOM CHESS - QA EVALUATION")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Number of games: {num_games}")
    print(f"MCTS simulations: {mcts_sims}")
    print("="*60)

    # Initialize players
    claude_player = ClaudePlayer(api_key=api_key)
    ai_player = AIPlayer(checkpoint_path, mcts_sims)

    games = []

    for i in range(num_games):
        print(f"\n{'='*60}")
        print(f"GAME {i+1}/{num_games}")
        print("="*60)

        # Alternate colors
        claude_color = Color.WHITE if i % 2 == 0 else Color.BLACK

        # Reset Claude's history for new game
        claude_player.move_history = []
        claude_player.reasoning_history = []

        game_result = play_game(
            claude_player, ai_player, claude_color, verbose
        )
        games.append(game_result)

    # Compile statistics
    claude_wins = sum(1 for g in games if g["claude_result"] == "win")
    claude_losses = sum(1 for g in games if g["claude_result"] == "loss")
    draws = sum(1 for g in games if g["claude_result"] == "draw")

    avg_game_length = sum(g["total_moves"] for g in games) / len(games)
    total_illegal_attempts = sum(g["illegal_attempts"] for g in games)

    # Generate report
    report = f"""
{'='*60}
QA EVALUATION REPORT
{'='*60}

## Summary
- **Checkpoint**: {checkpoint_path}
- **AI Training Iteration**: {ai_player.iteration}
- **Games Played**: {num_games}
- **MCTS Simulations**: {mcts_sims}

## Results
- **Claude Wins**: {claude_wins} ({claude_wins/num_games*100:.1f}%)
- **AI Wins**: {claude_losses} ({claude_losses/num_games*100:.1f}%)
- **Draws**: {draws} ({draws/num_games*100:.1f}%)

## Game Statistics
- **Average Game Length**: {avg_game_length:.1f} moves
- **Total Illegal Move Attempts by Claude**: {total_illegal_attempts}

## QA Assessment

### Is the training working?
"""

    if claude_losses > claude_wins:
        report += """
**YES - Training appears to be working!**
The AI is winning more games than Claude, suggesting it has learned meaningful
chess patterns and tactics. At iteration {}, the model is already competitive.
""".format(ai_player.iteration)
    elif claude_wins > claude_losses:
        report += """
**INCONCLUSIVE - AI may need more training**
Claude is winning more games. This could mean:
1. The AI needs more training iterations
2. The AI is making systematic tactical errors
3. Claude's full context helps it play more accurately
"""
    else:
        report += """
**PROMISING - AI is competitive**
The games are roughly even, suggesting the AI has learned reasonable play
but may benefit from additional training.
"""

    # Analyze game patterns
    report += """
### Game-by-Game Analysis
"""
    for i, g in enumerate(games):
        report += f"""
**Game {i+1}**: Claude ({g['claude_color']}) vs AI
- Result: {g['claude_result'].upper()}
- Moves: {g['total_moves']}
- Illegal attempts: {g['illegal_attempts']}
"""

    report += """
### Move Legality QA
"""
    if total_illegal_attempts == 0:
        report += "All Claude moves were legal on first attempt.\n"
    else:
        report += f"""
Claude attempted {total_illegal_attempts} illegal moves across all games.
This is typically due to parsing issues or misunderstanding piece movement.
All games completed successfully using retry logic.
"""

    print(report)

    return {
        "report": report,
        "summary": {
            "claude_wins": claude_wins,
            "ai_wins": claude_losses,
            "draws": draws,
            "avg_game_length": avg_game_length,
            "illegal_attempts": total_illegal_attempts,
            "ai_iteration": ai_player.iteration
        },
        "games": games
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Claude QA Player for Ultimate Random Chess")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_0099.pt",
                       help="Path to checkpoint file")
    parser.add_argument("--games", type=int, default=4,
                       help="Number of games to play")
    parser.add_argument("--sims", type=int, default=100,
                       help="MCTS simulations per move for AI")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    parser.add_argument("--api-key", type=str, default=None,
                       help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        # Find latest checkpoint
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
            if checkpoints:
                args.checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f"Using latest checkpoint: {args.checkpoint}")
            else:
                print("No checkpoints found!")
                sys.exit(1)
        else:
            print(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

    results = run_qa_evaluation(
        checkpoint_path=args.checkpoint,
        num_games=args.games,
        mcts_sims=args.sims,
        verbose=not args.quiet,
        api_key=args.api_key
    )

    # Save results
    output_file = f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "checkpoint": args.checkpoint,
            "summary": results["summary"],
            "games": results["games"]
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")
