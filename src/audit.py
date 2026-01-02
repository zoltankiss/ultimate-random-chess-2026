"""
Audit self-play games to verify rules are correct.

This lets you watch games move-by-move and verify:
1. Starting positions are valid (King between two Rooks)
2. Piece movements are correct
3. Castling works properly (Fischer Random style)
4. Check/checkmate detection works
5. En passant, promotion, etc.
"""

import random
from game import (
    GameState, Move, Color, Piece,
    create_starting_position, create_standard_position,
    generate_legal_moves, make_move, get_game_result,
    print_board, get_piece_type, get_piece_color,
    file_of, rank_of, sq, PIECE_SYMBOLS
)


def format_move_verbose(state: GameState, move: Move) -> str:
    """Format a move with piece info for clarity."""
    piece = state.board[move.from_sq]
    piece_type = get_piece_type(piece)
    piece_name = PIECE_SYMBOLS[piece_type]

    from_file = chr(ord('a') + file_of(move.from_sq))
    from_rank = rank_of(move.from_sq) + 1
    to_file = chr(ord('a') + file_of(move.to_sq))
    to_rank = rank_of(move.to_sq) + 1

    captured = state.board[move.to_sq]
    capture_str = ""
    if captured != 0:
        cap_type = get_piece_type(captured)
        capture_str = f" captures {PIECE_SYMBOLS[cap_type]}"

    promo_str = ""
    if move.promotion:
        promo_str = f" promotes to {PIECE_SYMBOLS[move.promotion]}"

    castle_str = ""
    if move.is_castling:
        if file_of(move.to_sq) == 6:
            castle_str = " (O-O kingside castle)"
        else:
            castle_str = " (O-O-O queenside castle)"

    return f"{piece_name}{from_file}{from_rank}-{to_file}{to_rank}{capture_str}{promo_str}{castle_str}"


def verify_starting_position(state: GameState) -> list:
    """Verify starting position is valid. Returns list of issues."""
    issues = []

    # Check white back rank
    white_pieces = []
    king_file = -1
    rook_files = []

    for file in range(8):
        piece = state.board[sq(file, 0)]
        if piece != 0:
            ptype = get_piece_type(piece)
            pcolor = get_piece_color(piece)
            if pcolor != Color.WHITE:
                issues.append(f"Non-white piece on white back rank at file {file}")
            white_pieces.append((file, ptype))
            if ptype == Piece.KING:
                king_file = file
            elif ptype == Piece.ROOK:
                rook_files.append(file)

    # Check king exists
    if king_file == -1:
        issues.append("No white king on back rank!")

    # Check at least two rooks (for castling)
    if len(rook_files) < 2:
        issues.append(f"Only {len(rook_files)} rooks on white back rank (need 2 for castling)")

    # Check king is between rooks
    if king_file != -1 and len(rook_files) >= 2:
        rook_files.sort()
        if not (rook_files[0] < king_file < rook_files[-1]):
            issues.append(f"King (file {king_file}) not between rooks (files {rook_files})")

    # Check pawns
    for file in range(8):
        piece = state.board[sq(file, 1)]
        if piece == 0:
            issues.append(f"Missing white pawn at file {file}")
        elif get_piece_type(piece) != Piece.PAWN:
            issues.append(f"Non-pawn on white pawn rank at file {file}")

    # Check black mirrors white
    for file in range(8):
        white_piece = state.board[sq(file, 0)]
        black_piece = state.board[sq(file, 7)]
        if white_piece != 0 and black_piece != 0:
            w_type = get_piece_type(white_piece)
            b_type = get_piece_type(black_piece)
            if w_type != b_type:
                issues.append(f"Black back rank doesn't mirror white at file {file}")

    return issues


def describe_backrank(state: GameState) -> str:
    """Describe the pieces on the back rank."""
    pieces = []
    for file in range(8):
        piece = state.board[sq(file, 0)]
        if piece != 0:
            ptype = get_piece_type(piece)
            pieces.append(PIECE_SYMBOLS[ptype])
        else:
            pieces.append(".")
    return " ".join(pieces)


def play_random_game(verbose: bool = True, step_through: bool = False) -> dict:
    """
    Play a game with random moves (no neural network).

    Args:
        verbose: Print the game as it progresses
        step_through: Wait for Enter after each move

    Returns:
        Dict with game stats and any issues found
    """
    state = create_starting_position()

    stats = {
        'issues': [],
        'moves': [],
        'castles': [],
        'promotions': [],
        'en_passants': [],
        'checks': 0,
        'result': None,
        'length': 0,
    }

    # Verify starting position
    position_issues = verify_starting_position(state)
    if position_issues:
        stats['issues'].extend(position_issues)

    if verbose:
        print("\n" + "="*60)
        print("STARTING POSITION")
        print("="*60)
        print_board(state)

        # Show back rank composition
        print(f"Back rank: {describe_backrank(state)}")
        print(f"White King file: {chr(ord('a') + state.white_king_file)}")
        print(f"White Rooks: {chr(ord('a') + state.white_rook_queenside_file)} (queenside), {chr(ord('a') + state.white_rook_kingside_file)} (kingside)")

        if position_issues:
            print("\n⚠️  POSITION ISSUES:")
            for issue in position_issues:
                print(f"  - {issue}")
        else:
            print("\n✓ Starting position valid")

        if step_through:
            input("\nPress Enter to start game...")

    move_num = 0
    history = []

    while True:
        move_num += 1

        # Check game over
        result = get_game_result(state)
        if result is not None:
            stats['result'] = result
            stats['length'] = move_num - 1

            if verbose:
                print("\n" + "="*60)
                if result > 0:
                    print("GAME OVER: White wins by checkmate!")
                elif result < 0:
                    print("GAME OVER: Black wins by checkmate!")
                else:
                    print("GAME OVER: Draw!")
                print("="*60)
            break

        # Safety limit
        if move_num > 200:
            stats['result'] = 0.0
            stats['length'] = move_num
            if verbose:
                print("\n(Game stopped at 200 moves)")
            break

        # Get legal moves
        legal_moves = generate_legal_moves(state)

        if not legal_moves:
            # Shouldn't happen if get_game_result is correct
            stats['issues'].append(f"No legal moves at move {move_num} but game not over!")
            break

        # Pick random move
        move = random.choice(legal_moves)
        move_str = format_move_verbose(state, move)

        stats['moves'].append(move_str)

        # Track special moves
        if move.is_castling:
            side = "kingside" if file_of(move.to_sq) == 6 else "queenside"
            color = "White" if state.turn == Color.WHITE else "Black"
            stats['castles'].append(f"Move {move_num}: {color} {side}")

        if move.promotion:
            stats['promotions'].append(f"Move {move_num}: {move_str}")

        # Check for en passant
        piece = state.board[move.from_sq]
        if get_piece_type(piece) == Piece.PAWN:
            if move.to_sq == state.en_passant_square:
                stats['en_passants'].append(f"Move {move_num}: {move_str}")

        if verbose:
            turn = "White" if state.turn == Color.WHITE else "Black"
            print(f"\nMove {move_num}. {turn}: {move_str}")

        # Make move
        new_state = make_move(state, move)

        # Check if opponent is in check
        from game import is_in_check
        if is_in_check(new_state, new_state.turn):
            stats['checks'] += 1
            if verbose:
                print("  → Check!")

        if verbose:
            print_board(new_state)

            if step_through:
                cmd = input("Press Enter for next move (or 'q' to quit): ")
                if cmd.lower() == 'q':
                    break

        state = new_state

    return stats


def audit_multiple_games(num_games: int = 5, verbose: bool = False):
    """Run multiple games and report any issues."""
    print(f"\n{'='*60}")
    print(f"AUDITING {num_games} RANDOM GAMES")
    print(f"{'='*60}")

    all_issues = []
    total_castles = 0
    total_promotions = 0
    total_en_passants = 0
    total_checks = 0
    total_moves = 0
    results = {'white': 0, 'black': 0, 'draw': 0}

    for i in range(num_games):
        print(f"\nGame {i+1}/{num_games}...", end=" ")
        stats = play_random_game(verbose=verbose, step_through=False)

        if stats['issues']:
            all_issues.extend([(i+1, issue) for issue in stats['issues']])
            print(f"⚠️  {len(stats['issues'])} issues!")
        else:
            print(f"✓ {stats['length']} moves", end="")
            if stats['castles']:
                print(f", {len(stats['castles'])} castles", end="")
            if stats['promotions']:
                print(f", {len(stats['promotions'])} promotions", end="")
            print()

        total_castles += len(stats['castles'])
        total_promotions += len(stats['promotions'])
        total_en_passants += len(stats['en_passants'])
        total_checks += stats['checks']
        total_moves += stats['length']

        if stats['result'] and stats['result'] > 0:
            results['white'] += 1
        elif stats['result'] and stats['result'] < 0:
            results['black'] += 1
        else:
            results['draw'] += 1

    print(f"\n{'='*60}")
    print("AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"Games played: {num_games}")
    print(f"Total moves: {total_moves} (avg {total_moves/num_games:.1f} per game)")
    print(f"Results: White {results['white']}, Black {results['black']}, Draw {results['draw']}")
    print(f"Castles: {total_castles}")
    print(f"Promotions: {total_promotions}")
    print(f"En passants: {total_en_passants}")
    print(f"Checks delivered: {total_checks}")

    if all_issues:
        print(f"\n⚠️  ISSUES FOUND ({len(all_issues)}):")
        for game_num, issue in all_issues:
            print(f"  Game {game_num}: {issue}")
    else:
        print(f"\n✓ No issues found!")

    return all_issues


def test_castling():
    """Specifically test castling in various positions."""
    print("\n" + "="*60)
    print("TESTING CASTLING")
    print("="*60)

    # Create a position where castling should be possible
    for trial in range(5):
        state = create_starting_position()
        print(f"\nTrial {trial + 1}:")
        print_board(state)
        print(f"King: {chr(ord('a') + state.white_king_file)}1")
        print(f"Rooks: {chr(ord('a') + state.white_rook_queenside_file)}1 (Q-side), {chr(ord('a') + state.white_rook_kingside_file)}1 (K-side)")

        # Check if castling moves are generated
        legal = generate_legal_moves(state)
        castling_moves = [m for m in legal if m.is_castling]

        if castling_moves:
            print(f"Castling available: {[str(m) for m in castling_moves]}")
        else:
            print("No immediate castling (pieces in the way)")

        # Try to clear path and castle
        # (This would need specific setup - just showing concept)


def interactive_game():
    """Play through a game interactively to verify rules."""
    print("\n" + "="*60)
    print("INTERACTIVE GAME AUDIT")
    print("="*60)
    print("Watch a random game move-by-move.")
    print("Press Enter to advance, 'q' to quit.")
    print("="*60)

    play_random_game(verbose=True, step_through=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audit Ultimate Random Chess rules")
    parser.add_argument("--games", type=int, default=10, help="Number of games to audit")
    parser.add_argument("--interactive", action="store_true", help="Step through one game interactively")
    parser.add_argument("--verbose", action="store_true", help="Show all moves in batch audit")
    parser.add_argument("--castling", action="store_true", help="Test castling specifically")

    args = parser.parse_args()

    if args.interactive:
        interactive_game()
    elif args.castling:
        test_castling()
    else:
        audit_multiple_games(args.games, verbose=args.verbose)
