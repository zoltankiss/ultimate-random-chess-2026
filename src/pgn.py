"""
PGN (Portable Game Notation) export for Ultimate Random Chess.

Saves games in standard chess notation format for review/analysis.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

from game import (
    GameState, Move, Color, Piece,
    get_piece_type, get_piece_color,
    file_of, rank_of, sq, PIECE_SYMBOLS,
    generate_legal_moves, is_in_check, get_game_result
)


def move_to_san(state: GameState, move: Move) -> str:
    """
    Convert a move to Standard Algebraic Notation (SAN).

    Examples: e4, Nf3, Bxc6, O-O, e8=Q
    """
    piece = state.board[move.from_sq]
    piece_type = get_piece_type(piece)

    # Castling
    if move.is_castling:
        if file_of(move.to_sq) == 6:
            return "O-O"
        else:
            return "O-O-O"

    from_file = chr(ord('a') + file_of(move.from_sq))
    from_rank = str(rank_of(move.from_sq) + 1)
    to_file = chr(ord('a') + file_of(move.to_sq))
    to_rank = str(rank_of(move.to_sq) + 1)

    # Capture?
    captured = state.board[move.to_sq]
    is_capture = captured != 0

    # En passant capture
    if piece_type == Piece.PAWN and move.to_sq == state.en_passant_square:
        is_capture = True

    # Build SAN
    if piece_type == Piece.PAWN:
        if is_capture:
            san = f"{from_file}x{to_file}{to_rank}"
        else:
            san = f"{to_file}{to_rank}"

        # Promotion
        if move.promotion:
            san += f"={PIECE_SYMBOLS[move.promotion]}"
    else:
        piece_letter = PIECE_SYMBOLS[piece_type]

        # Disambiguation: check if multiple pieces of same type can reach target
        legal_moves = generate_legal_moves(state)
        same_piece_moves = [
            m for m in legal_moves
            if m.to_sq == move.to_sq
            and get_piece_type(state.board[m.from_sq]) == piece_type
            and m.from_sq != move.from_sq
        ]

        disambig = ""
        if same_piece_moves:
            same_file = any(file_of(m.from_sq) == file_of(move.from_sq) for m in same_piece_moves)
            same_rank = any(rank_of(m.from_sq) == rank_of(move.from_sq) for m in same_piece_moves)

            if not same_file:
                disambig = from_file
            elif not same_rank:
                disambig = from_rank
            else:
                disambig = from_file + from_rank

        capture_str = "x" if is_capture else ""
        san = f"{piece_letter}{disambig}{capture_str}{to_file}{to_rank}"

    return san


def add_check_annotation(san: str, state_after: GameState) -> str:
    """Add + for check or # for checkmate."""
    if is_in_check(state_after, state_after.turn):
        result = get_game_result(state_after)
        if result is not None and result != 0:
            return san + "#"
        else:
            return san + "+"
    return san


def format_starting_position(state: GameState) -> str:
    """Format the starting position for the PGN header."""
    # Build FEN-like string for back rank
    pieces = []
    for file in range(8):
        piece = state.board[sq(file, 0)]
        if piece != 0:
            pieces.append(PIECE_SYMBOLS[get_piece_type(piece)])
        else:
            pieces.append(".")
    return "".join(pieces)


class PGNWriter:
    """Write games to PGN files."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.run_dir = None
        self.game_count = 0

    def start_run(self, run_name: str = None):
        """Start a new training run directory."""
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_dir = self.base_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.game_count = 0

        # Write run info
        info_path = self.run_dir / "run_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Training Run: {run_name}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")

        return str(self.run_dir)

    def save_game(
        self,
        moves: List[Tuple[GameState, Move]],
        result: float,
        iteration: int = 0,
        game_num: int = 0,
        metadata: dict = None
    ) -> str:
        """
        Save a game to PGN format.

        Args:
            moves: List of (state_before_move, move) tuples
            result: Game result (1.0 = white wins, -1.0 = black wins, 0.0 = draw)
            iteration: Training iteration number
            game_num: Game number within iteration
            metadata: Additional metadata for headers

        Returns:
            Path to saved PGN file
        """
        if self.run_dir is None:
            self.start_run()

        self.game_count += 1

        # Determine result string
        if result > 0.5:
            result_str = "1-0"
        elif result < -0.5:
            result_str = "0-1"
        else:
            result_str = "1/2-1/2"

        # Build PGN content
        lines = []

        # Headers
        lines.append(f'[Event "Ultimate Random Chess Training"]')
        lines.append(f'[Site "Local"]')
        lines.append(f'[Date "{datetime.now().strftime("%Y.%m.%d")}"]')
        lines.append(f'[Round "{iteration}.{game_num}"]')
        lines.append(f'[White "AlphaZero-URC"]')
        lines.append(f'[Black "AlphaZero-URC"]')
        lines.append(f'[Result "{result_str}"]')

        # Starting position (non-standard, so note it)
        if moves:
            starting_pos = format_starting_position(moves[0][0])
            lines.append(f'[SetUp "1"]')
            lines.append(f'[Variant "Ultimate Random Chess"]')
            lines.append(f'[StartingPosition "{starting_pos}"]')

        if metadata:
            for key, value in metadata.items():
                lines.append(f'[{key} "{value}"]')

        lines.append('')  # Blank line before moves

        # Moves
        move_text = []
        for i, (state, move) in enumerate(moves):
            san = move_to_san(state, move)

            # Get state after move to check for check/checkmate
            from game import make_move
            state_after = make_move(state, move)
            san = add_check_annotation(san, state_after)

            if i % 2 == 0:
                move_num = i // 2 + 1
                move_text.append(f"{move_num}. {san}")
            else:
                move_text.append(san)

        move_text.append(result_str)

        # Wrap moves to 80 chars per line
        current_line = ""
        for token in move_text:
            if len(current_line) + len(token) + 1 > 80:
                lines.append(current_line)
                current_line = token
            else:
                if current_line:
                    current_line += " " + token
                else:
                    current_line = token
        if current_line:
            lines.append(current_line)

        lines.append('')  # Trailing newline

        # Write file
        filename = f"game_{iteration:04d}_{game_num:04d}.pgn"
        filepath = self.run_dir / filename

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        return str(filepath)

    def save_batch(
        self,
        games: List[Tuple[List[Tuple[GameState, Move]], float]],
        iteration: int = 0
    ) -> List[str]:
        """Save multiple games from one iteration."""
        paths = []
        for i, (moves, result) in enumerate(games):
            path = self.save_game(moves, result, iteration, i + 1)
            paths.append(path)
        return paths

    def get_stats(self) -> dict:
        """Get statistics about saved games."""
        if self.run_dir is None:
            return {'games': 0}

        pgn_files = list(self.run_dir.glob("*.pgn"))
        return {
            'games': len(pgn_files),
            'directory': str(self.run_dir),
        }


# Test
if __name__ == "__main__":
    from game import create_starting_position, make_move, generate_legal_moves
    import random

    print("Testing PGN export...")

    # Play a short random game
    state = create_starting_position()
    moves = []

    for _ in range(20):
        result = get_game_result(state)
        if result is not None:
            break

        legal = generate_legal_moves(state)
        if not legal:
            break

        move = random.choice(legal)
        moves.append((state.copy(), move))
        state = make_move(state, move)

    result = get_game_result(state) or 0.0

    # Save it
    writer = PGNWriter("/tmp/test_pgn")
    path = writer.save_game(moves, result, iteration=1, game_num=1)

    print(f"Saved to: {path}")
    print("\nPGN content:")
    with open(path) as f:
        print(f.read())
