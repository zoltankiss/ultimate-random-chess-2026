"""
Ultimate Random Chess - Game Engine

Core game logic: board representation, move generation, and game rules.
Follows Fischer Random (Chess960) castling rules.
"""

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import copy


class Piece(IntEnum):
    EMPTY = 0
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6
    # Special pieces for Ultimate Random Chess
    ARCHBISHOP = 7    # Knight + Bishop
    CHANCELLOR = 8    # Knight + Rook
    AMAZON = 9        # Knight + Queen (most powerful!)
    CAMEL = 10        # Leaps (3,1) instead of (2,1)
    ZEBRA = 11        # Leaps (3,2)
    GRASSHOPPER = 12  # Hops over pieces on queen lines
    CANNON = 13       # Xiangqi cannon - slides to move, hops to capture


class Color(IntEnum):
    WHITE = 0
    BLACK = 1


# Number of piece types (excluding EMPTY)
NUM_PIECE_TYPES = 13

# Piece encoding: color * NUM_PIECE_TYPES + piece_type (1-13), 0 = empty
# White pieces: 1-13, Black pieces: 14-26
def make_piece(piece: Piece, color: Color) -> int:
    if piece == Piece.EMPTY:
        return 0
    return color * NUM_PIECE_TYPES + piece


def get_piece_type(encoded: int) -> Piece:
    if encoded == 0:
        return Piece.EMPTY
    return Piece((encoded - 1) % NUM_PIECE_TYPES + 1)


def get_piece_color(encoded: int) -> Optional[Color]:
    if encoded == 0:
        return None
    return Color((encoded - 1) // NUM_PIECE_TYPES)


PIECE_SYMBOLS = {
    Piece.EMPTY: '.',
    Piece.PAWN: 'P',
    Piece.KNIGHT: 'N',
    Piece.BISHOP: 'B',
    Piece.ROOK: 'R',
    Piece.QUEEN: 'Q',
    Piece.KING: 'K',
    Piece.ARCHBISHOP: 'A',
    Piece.CHANCELLOR: 'C',
    Piece.AMAZON: 'M',
    Piece.CAMEL: 'L',
    Piece.ZEBRA: 'Z',
    Piece.GRASSHOPPER: 'G',
    Piece.CANNON: 'X',
}


@dataclass
class Move:
    from_sq: int  # 0-63
    to_sq: int    # 0-63
    promotion: Optional[Piece] = None  # For pawn promotion
    is_castling: bool = False

    def __repr__(self):
        from_file = chr(ord('a') + self.from_sq % 8)
        from_rank = self.from_sq // 8 + 1
        to_file = chr(ord('a') + self.to_sq % 8)
        to_rank = self.to_sq // 8 + 1
        promo = PIECE_SYMBOLS.get(self.promotion, '').lower() if self.promotion else ''
        return f"{from_file}{from_rank}{to_file}{to_rank}{promo}"

    def uci(self) -> str:
        return repr(self)


@dataclass
class GameState:
    """Complete game state for Ultimate Random Chess."""

    board: list = field(default_factory=lambda: [0] * 64)
    turn: Color = Color.WHITE

    # Castling rights: store the file (0-7) of each rook, or -1 if lost
    white_king_file: int = -1
    white_rook_kingside_file: int = -1  # Right rook (higher file than king)
    white_rook_queenside_file: int = -1  # Left rook (lower file than king)
    black_king_file: int = -1
    black_rook_kingside_file: int = -1
    black_rook_queenside_file: int = -1

    # Has the king/rook moved? (loses castling rights)
    white_can_castle_kingside: bool = True
    white_can_castle_queenside: bool = True
    black_can_castle_kingside: bool = True
    black_can_castle_queenside: bool = True

    en_passant_square: int = -1  # Square where en passant capture is possible
    halfmove_clock: int = 0  # For 50-move rule
    fullmove_number: int = 1

    # Position history for repetition detection (list of board hashes)
    position_history: list = field(default_factory=list)

    def copy(self) -> 'GameState':
        new_state = GameState(
            board=self.board.copy(),
            turn=self.turn,
            white_king_file=self.white_king_file,
            white_rook_kingside_file=self.white_rook_kingside_file,
            white_rook_queenside_file=self.white_rook_queenside_file,
            black_king_file=self.black_king_file,
            black_rook_kingside_file=self.black_rook_kingside_file,
            black_rook_queenside_file=self.black_rook_queenside_file,
            white_can_castle_kingside=self.white_can_castle_kingside,
            white_can_castle_queenside=self.white_can_castle_queenside,
            black_can_castle_kingside=self.black_can_castle_kingside,
            black_can_castle_queenside=self.black_can_castle_queenside,
            en_passant_square=self.en_passant_square,
            halfmove_clock=self.halfmove_clock,
            fullmove_number=self.fullmove_number,
            position_history=self.position_history.copy(),
        )
        return new_state


def sq(file: int, rank: int) -> int:
    """Convert file (0-7) and rank (0-7) to square index (0-63)."""
    return rank * 8 + file


def file_of(square: int) -> int:
    return square % 8


def rank_of(square: int) -> int:
    return square // 8


def generate_random_backrank() -> list:
    """
    Generate a random back rank for Ultimate Random Chess.
    Rules:
    - Exactly 1 King
    - Exactly 2 Rooks (one on each side of King for castling)
    - 5 other pieces randomly from the piece pool (including special pieces!)
    """
    # Place king on files b-g (1-6) to ensure room for rooks on both sides
    king_file = random.randint(1, 6)

    # Place rooks: one on each side of king
    left_rook_file = random.randint(0, king_file - 1)
    right_rook_file = random.randint(king_file + 1, 7)

    # Fill remaining 5 squares randomly - includes ALL pieces!
    piece_pool = [
        # Standard pieces
        Piece.QUEEN,
        Piece.ROOK,
        Piece.BISHOP,
        Piece.KNIGHT,
        # Special pieces - Ultimate Random Chess!
        Piece.ARCHBISHOP,   # Knight + Bishop
        Piece.CHANCELLOR,   # Knight + Rook
        Piece.AMAZON,       # Knight + Queen (super powerful)
        Piece.CAMEL,        # Leaps (3,1)
        Piece.ZEBRA,        # Leaps (3,2)
        Piece.GRASSHOPPER,  # Hops over pieces
        Piece.CANNON,       # Xiangqi cannon
    ]

    backrank = [Piece.EMPTY] * 8
    backrank[king_file] = Piece.KING
    backrank[left_rook_file] = Piece.ROOK
    backrank[right_rook_file] = Piece.ROOK

    for file in range(8):
        if backrank[file] == Piece.EMPTY:
            backrank[file] = random.choice(piece_pool)

    return backrank, king_file, left_rook_file, right_rook_file


def create_starting_position() -> GameState:
    """Create a new game with random starting position."""
    state = GameState()

    # Generate random backrank
    backrank, king_file, left_rook_file, right_rook_file = generate_random_backrank()

    # Set up white pieces (rank 0 and 1)
    for file in range(8):
        # Back rank
        state.board[sq(file, 0)] = make_piece(backrank[file], Color.WHITE)
        # Pawns
        state.board[sq(file, 1)] = make_piece(Piece.PAWN, Color.WHITE)

    # Set up black pieces (rank 6 and 7) - mirror of white
    for file in range(8):
        # Pawns
        state.board[sq(file, 6)] = make_piece(Piece.PAWN, Color.BLACK)
        # Back rank
        state.board[sq(file, 7)] = make_piece(backrank[file], Color.BLACK)

    # Store castling info
    state.white_king_file = king_file
    state.white_rook_queenside_file = left_rook_file
    state.white_rook_kingside_file = right_rook_file
    state.black_king_file = king_file
    state.black_rook_queenside_file = left_rook_file
    state.black_rook_kingside_file = right_rook_file

    return state


def create_standard_position() -> GameState:
    """Create a standard chess starting position (for testing)."""
    state = GameState()

    # Standard back rank
    backrank = [Piece.ROOK, Piece.KNIGHT, Piece.BISHOP, Piece.QUEEN,
                Piece.KING, Piece.BISHOP, Piece.KNIGHT, Piece.ROOK]

    for file in range(8):
        state.board[sq(file, 0)] = make_piece(backrank[file], Color.WHITE)
        state.board[sq(file, 1)] = make_piece(Piece.PAWN, Color.WHITE)
        state.board[sq(file, 6)] = make_piece(Piece.PAWN, Color.BLACK)
        state.board[sq(file, 7)] = make_piece(backrank[file], Color.BLACK)

    state.white_king_file = 4
    state.white_rook_queenside_file = 0
    state.white_rook_kingside_file = 7
    state.black_king_file = 4
    state.black_rook_queenside_file = 0
    state.black_rook_kingside_file = 7

    return state


def print_board(state: GameState):
    """Print the board in a human-readable format."""
    print()
    for rank in range(7, -1, -1):
        print(f"{rank + 1} ", end="")
        for file in range(8):
            piece = state.board[sq(file, rank)]
            if piece == 0:
                print(". ", end="")
            else:
                piece_type = get_piece_type(piece)
                color = get_piece_color(piece)
                symbol = PIECE_SYMBOLS[piece_type]
                if color == Color.BLACK:
                    symbol = symbol.lower()
                print(f"{symbol} ", end="")
        print()
    print("  a b c d e f g h")
    print(f"Turn: {'White' if state.turn == Color.WHITE else 'Black'}")
    print()


# Direction vectors for sliding pieces
DIRECTIONS = {
    'orthogonal': [(1, 0), (-1, 0), (0, 1), (0, -1)],
    'diagonal': [(1, 1), (1, -1), (-1, 1), (-1, -1)],
    'all': [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)],
}

KNIGHT_MOVES = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]

# Special piece leap patterns
CAMEL_MOVES = [(3, 1), (3, -1), (-3, 1), (-3, -1), (1, 3), (1, -3), (-1, 3), (-1, -3)]
ZEBRA_MOVES = [(3, 2), (3, -2), (-3, 2), (-3, -2), (2, 3), (2, -3), (-2, 3), (-2, -3)]


def is_valid_square(file: int, rank: int) -> bool:
    return 0 <= file < 8 and 0 <= rank < 8


def generate_sliding_moves(state: GameState, from_sq: int, directions: list) -> list:
    """Generate moves for sliding pieces (rook, bishop, queen)."""
    moves = []
    from_file, from_rank = file_of(from_sq), rank_of(from_sq)
    my_color = get_piece_color(state.board[from_sq])

    for df, dr in directions:
        for dist in range(1, 8):
            to_file = from_file + df * dist
            to_rank = from_rank + dr * dist

            if not is_valid_square(to_file, to_rank):
                break

            to_sq = sq(to_file, to_rank)
            target = state.board[to_sq]

            if target == 0:
                moves.append(Move(from_sq, to_sq))
            elif get_piece_color(target) != my_color:
                moves.append(Move(from_sq, to_sq))  # Capture
                break
            else:
                break  # Blocked by own piece

    return moves


def generate_knight_moves(state: GameState, from_sq: int) -> list:
    """Generate knight moves."""
    moves = []
    from_file, from_rank = file_of(from_sq), rank_of(from_sq)
    my_color = get_piece_color(state.board[from_sq])

    for df, dr in KNIGHT_MOVES:
        to_file = from_file + df
        to_rank = from_rank + dr

        if not is_valid_square(to_file, to_rank):
            continue

        to_sq = sq(to_file, to_rank)
        target = state.board[to_sq]

        if target == 0 or get_piece_color(target) != my_color:
            moves.append(Move(from_sq, to_sq))

    return moves


def generate_pawn_moves(state: GameState, from_sq: int) -> list:
    """Generate pawn moves including en passant and promotion."""
    moves = []
    from_file, from_rank = file_of(from_sq), rank_of(from_sq)
    my_color = get_piece_color(state.board[from_sq])

    direction = 1 if my_color == Color.WHITE else -1
    start_rank = 1 if my_color == Color.WHITE else 6
    promo_rank = 7 if my_color == Color.WHITE else 0

    # Single push
    to_rank = from_rank + direction
    if is_valid_square(from_file, to_rank):
        to_sq = sq(from_file, to_rank)
        if state.board[to_sq] == 0:
            if to_rank == promo_rank:
                for promo_piece in [Piece.QUEEN, Piece.ROOK, Piece.BISHOP, Piece.KNIGHT]:
                    moves.append(Move(from_sq, to_sq, promotion=promo_piece))
            else:
                moves.append(Move(from_sq, to_sq))

            # Double push from starting rank
            if from_rank == start_rank:
                to_rank2 = from_rank + 2 * direction
                to_sq2 = sq(from_file, to_rank2)
                if state.board[to_sq2] == 0:
                    moves.append(Move(from_sq, to_sq2))

    # Captures
    for df in [-1, 1]:
        to_file = from_file + df
        to_rank = from_rank + direction

        if not is_valid_square(to_file, to_rank):
            continue

        to_sq = sq(to_file, to_rank)
        target = state.board[to_sq]

        # Regular capture
        if target != 0 and get_piece_color(target) != my_color:
            if to_rank == promo_rank:
                for promo_piece in [Piece.QUEEN, Piece.ROOK, Piece.BISHOP, Piece.KNIGHT]:
                    moves.append(Move(from_sq, to_sq, promotion=promo_piece))
            else:
                moves.append(Move(from_sq, to_sq))

        # En passant
        if to_sq == state.en_passant_square:
            moves.append(Move(from_sq, to_sq))

    return moves


def generate_king_moves(state: GameState, from_sq: int) -> list:
    """Generate king moves (not including castling - added separately)."""
    moves = []
    from_file, from_rank = file_of(from_sq), rank_of(from_sq)
    my_color = get_piece_color(state.board[from_sq])

    for df, dr in DIRECTIONS['all']:
        to_file = from_file + df
        to_rank = from_rank + dr

        if not is_valid_square(to_file, to_rank):
            continue

        to_sq = sq(to_file, to_rank)
        target = state.board[to_sq]

        if target == 0 or get_piece_color(target) != my_color:
            moves.append(Move(from_sq, to_sq))

    return moves


def generate_leaper_moves(state: GameState, from_sq: int, leap_pattern: list) -> list:
    """Generate moves for leaping pieces (camel, zebra)."""
    moves = []
    from_file, from_rank = file_of(from_sq), rank_of(from_sq)
    my_color = get_piece_color(state.board[from_sq])

    for df, dr in leap_pattern:
        to_file = from_file + df
        to_rank = from_rank + dr

        if not is_valid_square(to_file, to_rank):
            continue

        to_sq = sq(to_file, to_rank)
        target = state.board[to_sq]

        if target == 0 or get_piece_color(target) != my_color:
            moves.append(Move(from_sq, to_sq))

    return moves


def generate_grasshopper_moves(state: GameState, from_sq: int) -> list:
    """
    Generate Grasshopper moves.
    The grasshopper moves along queen lines (orthogonal + diagonal),
    MUST hop over exactly one piece, and lands on the square immediately beyond it.
    """
    moves = []
    from_file, from_rank = file_of(from_sq), rank_of(from_sq)
    my_color = get_piece_color(state.board[from_sq])

    for df, dr in DIRECTIONS['all']:
        found_hurdle = False
        for dist in range(1, 8):
            to_file = from_file + df * dist
            to_rank = from_rank + dr * dist

            if not is_valid_square(to_file, to_rank):
                break

            to_sq = sq(to_file, to_rank)
            target = state.board[to_sq]

            if not found_hurdle:
                # Looking for a piece to hop over
                if target != 0:
                    found_hurdle = True
            else:
                # Land immediately after the hurdle
                if target == 0 or get_piece_color(target) != my_color:
                    moves.append(Move(from_sq, to_sq))
                break  # Can only land on the first square after hurdle

    return moves


def generate_cannon_moves(state: GameState, from_sq: int) -> list:
    """
    Generate Cannon moves (Xiangqi-style).
    For non-capture moves: slides like a rook (orthogonally only).
    For captures: must hop over exactly one piece (screen) to capture.
    """
    moves = []
    from_file, from_rank = file_of(from_sq), rank_of(from_sq)
    my_color = get_piece_color(state.board[from_sq])

    for df, dr in DIRECTIONS['orthogonal']:
        found_screen = False
        for dist in range(1, 8):
            to_file = from_file + df * dist
            to_rank = from_rank + dr * dist

            if not is_valid_square(to_file, to_rank):
                break

            to_sq = sq(to_file, to_rank)
            target = state.board[to_sq]

            if not found_screen:
                # No screen yet
                if target == 0:
                    # Can move to empty squares (non-capture)
                    moves.append(Move(from_sq, to_sq))
                else:
                    # Found a screen (piece to hop over)
                    found_screen = True
            else:
                # After screen, can only capture (not move to empty)
                if target != 0:
                    if get_piece_color(target) != my_color:
                        moves.append(Move(from_sq, to_sq))  # Capture!
                    break  # Can't go further regardless

    return moves


def find_king(state: GameState, color: Color) -> int:
    """Find the king's square for the given color."""
    king_piece = make_piece(Piece.KING, color)
    for sq_idx in range(64):
        if state.board[sq_idx] == king_piece:
            return sq_idx
    return -1


def is_square_attacked(state: GameState, square: int, by_color: Color) -> bool:
    """Check if a square is attacked by the given color."""
    sq_file, sq_rank = file_of(square), rank_of(square)

    # Check knight attacks (also Archbishop, Chancellor, Amazon)
    for df, dr in KNIGHT_MOVES:
        atk_file = sq_file + df
        atk_rank = sq_rank + dr
        if is_valid_square(atk_file, atk_rank):
            piece = state.board[sq(atk_file, atk_rank)]
            if piece != 0 and get_piece_color(piece) == by_color:
                ptype = get_piece_type(piece)
                if ptype in [Piece.KNIGHT, Piece.ARCHBISHOP, Piece.CHANCELLOR, Piece.AMAZON]:
                    return True

    # Check camel attacks (3,1 leaper)
    for df, dr in CAMEL_MOVES:
        atk_file = sq_file + df
        atk_rank = sq_rank + dr
        if is_valid_square(atk_file, atk_rank):
            piece = state.board[sq(atk_file, atk_rank)]
            if piece != 0 and get_piece_color(piece) == by_color and get_piece_type(piece) == Piece.CAMEL:
                return True

    # Check zebra attacks (3,2 leaper)
    for df, dr in ZEBRA_MOVES:
        atk_file = sq_file + df
        atk_rank = sq_rank + dr
        if is_valid_square(atk_file, atk_rank):
            piece = state.board[sq(atk_file, atk_rank)]
            if piece != 0 and get_piece_color(piece) == by_color and get_piece_type(piece) == Piece.ZEBRA:
                return True

    # Check sliding attacks (rook, bishop, queen, and compound pieces)
    for df, dr in DIRECTIONS['orthogonal']:
        for dist in range(1, 8):
            atk_file = sq_file + df * dist
            atk_rank = sq_rank + dr * dist
            if not is_valid_square(atk_file, atk_rank):
                break
            piece = state.board[sq(atk_file, atk_rank)]
            if piece != 0:
                if get_piece_color(piece) == by_color:
                    ptype = get_piece_type(piece)
                    # Rook, Queen, Chancellor (N+R), Amazon (N+Q), Cannon (can slide-attack on orthogonal)
                    if ptype in [Piece.ROOK, Piece.QUEEN, Piece.CHANCELLOR, Piece.AMAZON]:
                        return True
                break

    for df, dr in DIRECTIONS['diagonal']:
        for dist in range(1, 8):
            atk_file = sq_file + df * dist
            atk_rank = sq_rank + dr * dist
            if not is_valid_square(atk_file, atk_rank):
                break
            piece = state.board[sq(atk_file, atk_rank)]
            if piece != 0:
                if get_piece_color(piece) == by_color:
                    ptype = get_piece_type(piece)
                    # Bishop, Queen, Archbishop (N+B), Amazon (N+Q)
                    if ptype in [Piece.BISHOP, Piece.QUEEN, Piece.ARCHBISHOP, Piece.AMAZON]:
                        return True
                break

    # Check grasshopper attacks (hops over one piece on queen lines)
    for df, dr in DIRECTIONS['all']:
        found_hurdle = False
        for dist in range(1, 8):
            atk_file = sq_file + df * dist
            atk_rank = sq_rank + dr * dist
            if not is_valid_square(atk_file, atk_rank):
                break
            piece = state.board[sq(atk_file, atk_rank)]
            if not found_hurdle:
                if piece != 0:
                    found_hurdle = True
            else:
                # The grasshopper would land here after hopping
                if piece != 0 and get_piece_color(piece) == by_color and get_piece_type(piece) == Piece.GRASSHOPPER:
                    return True
                break

    # Check cannon attacks (hops over screen to capture on orthogonal)
    for df, dr in DIRECTIONS['orthogonal']:
        found_screen = False
        for dist in range(1, 8):
            atk_file = sq_file + df * dist
            atk_rank = sq_rank + dr * dist
            if not is_valid_square(atk_file, atk_rank):
                break
            piece = state.board[sq(atk_file, atk_rank)]
            if not found_screen:
                if piece != 0:
                    found_screen = True
            else:
                # After screen, cannon can attack here
                if piece != 0:
                    if get_piece_color(piece) == by_color and get_piece_type(piece) == Piece.CANNON:
                        return True
                    break

    # Check king attacks (for adjacent squares)
    for df, dr in DIRECTIONS['all']:
        atk_file = sq_file + df
        atk_rank = sq_rank + dr
        if is_valid_square(atk_file, atk_rank):
            piece = state.board[sq(atk_file, atk_rank)]
            if piece != 0 and get_piece_color(piece) == by_color and get_piece_type(piece) == Piece.KING:
                return True

    # Check pawn attacks
    pawn_direction = -1 if by_color == Color.WHITE else 1  # Pawns attack "backward" from their perspective
    for df in [-1, 1]:
        atk_file = sq_file + df
        atk_rank = sq_rank + pawn_direction
        if is_valid_square(atk_file, atk_rank):
            piece = state.board[sq(atk_file, atk_rank)]
            if piece != 0 and get_piece_color(piece) == by_color and get_piece_type(piece) == Piece.PAWN:
                return True

    return False


def is_in_check(state: GameState, color: Color) -> bool:
    """Check if the given color's king is in check."""
    king_sq = find_king(state, color)
    if king_sq == -1:
        return True  # No king = bad state
    enemy_color = Color.BLACK if color == Color.WHITE else Color.WHITE
    return is_square_attacked(state, king_sq, enemy_color)


def generate_castling_moves(state: GameState) -> list:
    """
    Generate castling moves following Fischer Random (Chess960) rules.

    Key principle: After castling, king ends on g1/c1 and rook ends on f1/d1,
    regardless of where they started.
    """
    moves = []

    if state.turn == Color.WHITE:
        king_sq = find_king(state, Color.WHITE)
        if king_sq == -1:
            return moves

        king_file = file_of(king_sq)
        rank = 0

        # Check if in check
        if is_in_check(state, Color.WHITE):
            return moves

        # Kingside castling (king to g1, rook to f1)
        if state.white_can_castle_kingside and state.white_rook_kingside_file >= 0:
            rook_file = state.white_rook_kingside_file
            king_dest = 6  # g1
            rook_dest = 5  # f1

            can_castle = True

            # Check path is clear (all squares between king start, king end, rook start, rook end)
            min_file = min(king_file, king_dest, rook_file, rook_dest)
            max_file = max(king_file, king_dest, rook_file, rook_dest)

            for f in range(min_file, max_file + 1):
                if f == king_file or f == rook_file:
                    continue  # King and rook can be in the path
                if state.board[sq(f, rank)] != 0:
                    can_castle = False
                    break

            # Check king doesn't pass through or end on attacked square
            if can_castle:
                step = 1 if king_dest > king_file else -1
                for f in range(king_file, king_dest + step, step):
                    if is_square_attacked(state, sq(f, rank), Color.BLACK):
                        can_castle = False
                        break

            if can_castle:
                # Castling move: king to g1
                moves.append(Move(king_sq, sq(king_dest, rank), is_castling=True))

        # Queenside castling (king to c1, rook to d1)
        if state.white_can_castle_queenside and state.white_rook_queenside_file >= 0:
            rook_file = state.white_rook_queenside_file
            king_dest = 2  # c1
            rook_dest = 3  # d1

            can_castle = True

            min_file = min(king_file, king_dest, rook_file, rook_dest)
            max_file = max(king_file, king_dest, rook_file, rook_dest)

            for f in range(min_file, max_file + 1):
                if f == king_file or f == rook_file:
                    continue
                if state.board[sq(f, rank)] != 0:
                    can_castle = False
                    break

            if can_castle:
                step = 1 if king_dest > king_file else -1
                for f in range(king_file, king_dest + step, step):
                    if is_square_attacked(state, sq(f, rank), Color.BLACK):
                        can_castle = False
                        break

            if can_castle:
                moves.append(Move(king_sq, sq(king_dest, rank), is_castling=True))

    else:  # Black's turn
        king_sq = find_king(state, Color.BLACK)
        if king_sq == -1:
            return moves

        king_file = file_of(king_sq)
        rank = 7

        if is_in_check(state, Color.BLACK):
            return moves

        # Kingside castling (king to g8, rook to f8)
        if state.black_can_castle_kingside and state.black_rook_kingside_file >= 0:
            rook_file = state.black_rook_kingside_file
            king_dest = 6  # g8
            rook_dest = 5  # f8

            can_castle = True

            min_file = min(king_file, king_dest, rook_file, rook_dest)
            max_file = max(king_file, king_dest, rook_file, rook_dest)

            for f in range(min_file, max_file + 1):
                if f == king_file or f == rook_file:
                    continue
                if state.board[sq(f, rank)] != 0:
                    can_castle = False
                    break

            if can_castle:
                step = 1 if king_dest > king_file else -1
                for f in range(king_file, king_dest + step, step):
                    if is_square_attacked(state, sq(f, rank), Color.WHITE):
                        can_castle = False
                        break

            if can_castle:
                moves.append(Move(king_sq, sq(king_dest, rank), is_castling=True))

        # Queenside castling (king to c8, rook to d8)
        if state.black_can_castle_queenside and state.black_rook_queenside_file >= 0:
            rook_file = state.black_rook_queenside_file
            king_dest = 2  # c8
            rook_dest = 3  # d8

            can_castle = True

            min_file = min(king_file, king_dest, rook_file, rook_dest)
            max_file = max(king_file, king_dest, rook_file, rook_dest)

            for f in range(min_file, max_file + 1):
                if f == king_file or f == rook_file:
                    continue
                if state.board[sq(f, rank)] != 0:
                    can_castle = False
                    break

            if can_castle:
                step = 1 if king_dest > king_file else -1
                for f in range(king_file, king_dest + step, step):
                    if is_square_attacked(state, sq(f, rank), Color.WHITE):
                        can_castle = False
                        break

            if can_castle:
                moves.append(Move(king_sq, sq(king_dest, rank), is_castling=True))

    return moves


def generate_pseudo_legal_moves(state: GameState) -> list:
    """Generate all pseudo-legal moves (may leave king in check)."""
    moves = []

    for from_sq in range(64):
        piece = state.board[from_sq]
        if piece == 0:
            continue

        color = get_piece_color(piece)
        if color != state.turn:
            continue

        piece_type = get_piece_type(piece)

        if piece_type == Piece.PAWN:
            moves.extend(generate_pawn_moves(state, from_sq))
        elif piece_type == Piece.KNIGHT:
            moves.extend(generate_knight_moves(state, from_sq))
        elif piece_type == Piece.BISHOP:
            moves.extend(generate_sliding_moves(state, from_sq, DIRECTIONS['diagonal']))
        elif piece_type == Piece.ROOK:
            moves.extend(generate_sliding_moves(state, from_sq, DIRECTIONS['orthogonal']))
        elif piece_type == Piece.QUEEN:
            moves.extend(generate_sliding_moves(state, from_sq, DIRECTIONS['all']))
        elif piece_type == Piece.KING:
            moves.extend(generate_king_moves(state, from_sq))
        # === Special pieces for Ultimate Random Chess ===
        elif piece_type == Piece.ARCHBISHOP:
            # Knight + Bishop compound
            moves.extend(generate_knight_moves(state, from_sq))
            moves.extend(generate_sliding_moves(state, from_sq, DIRECTIONS['diagonal']))
        elif piece_type == Piece.CHANCELLOR:
            # Knight + Rook compound
            moves.extend(generate_knight_moves(state, from_sq))
            moves.extend(generate_sliding_moves(state, from_sq, DIRECTIONS['orthogonal']))
        elif piece_type == Piece.AMAZON:
            # Knight + Queen compound (most powerful!)
            moves.extend(generate_knight_moves(state, from_sq))
            moves.extend(generate_sliding_moves(state, from_sq, DIRECTIONS['all']))
        elif piece_type == Piece.CAMEL:
            # Leaps (3,1) instead of knight's (2,1)
            moves.extend(generate_leaper_moves(state, from_sq, CAMEL_MOVES))
        elif piece_type == Piece.ZEBRA:
            # Leaps (3,2) instead of knight's (2,1)
            moves.extend(generate_leaper_moves(state, from_sq, ZEBRA_MOVES))
        elif piece_type == Piece.GRASSHOPPER:
            # Hops over pieces on queen lines
            moves.extend(generate_grasshopper_moves(state, from_sq))
        elif piece_type == Piece.CANNON:
            # Xiangqi cannon - slides to move, hops to capture
            moves.extend(generate_cannon_moves(state, from_sq))

    # Add castling moves
    moves.extend(generate_castling_moves(state))

    return moves


def make_move(state: GameState, move: Move) -> GameState:
    """Make a move and return the new game state."""
    new_state = state.copy()

    from_sq = move.from_sq
    to_sq = move.to_sq
    piece = new_state.board[from_sq]
    piece_type = get_piece_type(piece)
    color = get_piece_color(piece)
    captured = new_state.board[to_sq]

    # Handle castling
    if move.is_castling:
        rank = 0 if color == Color.WHITE else 7
        king_file = file_of(from_sq)
        king_dest_file = file_of(to_sq)

        # Determine which rook to move
        if king_dest_file == 6:  # Kingside
            if color == Color.WHITE:
                rook_file = new_state.white_rook_kingside_file
            else:
                rook_file = new_state.black_rook_kingside_file
            rook_dest = 5
        else:  # Queenside
            if color == Color.WHITE:
                rook_file = new_state.white_rook_queenside_file
            else:
                rook_file = new_state.black_rook_queenside_file
            rook_dest = 3

        # Move king and rook
        rook_piece = new_state.board[sq(rook_file, rank)]
        new_state.board[from_sq] = 0
        new_state.board[sq(rook_file, rank)] = 0
        new_state.board[to_sq] = piece
        new_state.board[sq(rook_dest, rank)] = rook_piece

        # Remove castling rights
        if color == Color.WHITE:
            new_state.white_can_castle_kingside = False
            new_state.white_can_castle_queenside = False
        else:
            new_state.black_can_castle_kingside = False
            new_state.black_can_castle_queenside = False

    else:
        # Regular move
        new_state.board[from_sq] = 0
        new_state.board[to_sq] = piece

        # Handle en passant capture
        if piece_type == Piece.PAWN and to_sq == state.en_passant_square:
            ep_capture_rank = rank_of(from_sq)
            ep_capture_sq = sq(file_of(to_sq), ep_capture_rank)
            new_state.board[ep_capture_sq] = 0

        # Handle promotion
        if move.promotion:
            new_state.board[to_sq] = make_piece(move.promotion, color)

    # Update en passant square
    new_state.en_passant_square = -1
    if piece_type == Piece.PAWN:
        if abs(rank_of(to_sq) - rank_of(from_sq)) == 2:
            # Double pawn push - set en passant square
            ep_rank = (rank_of(from_sq) + rank_of(to_sq)) // 2
            new_state.en_passant_square = sq(file_of(from_sq), ep_rank)

    # Update castling rights based on piece movement
    if piece_type == Piece.KING:
        if color == Color.WHITE:
            new_state.white_can_castle_kingside = False
            new_state.white_can_castle_queenside = False
        else:
            new_state.black_can_castle_kingside = False
            new_state.black_can_castle_queenside = False

    if piece_type == Piece.ROOK:
        from_file = file_of(from_sq)
        if color == Color.WHITE:
            if from_file == new_state.white_rook_kingside_file:
                new_state.white_can_castle_kingside = False
            elif from_file == new_state.white_rook_queenside_file:
                new_state.white_can_castle_queenside = False
        else:
            if from_file == new_state.black_rook_kingside_file:
                new_state.black_can_castle_kingside = False
            elif from_file == new_state.black_rook_queenside_file:
                new_state.black_can_castle_queenside = False

    # Update halfmove clock
    if piece_type == Piece.PAWN or captured != 0:
        new_state.halfmove_clock = 0
    else:
        new_state.halfmove_clock += 1

    # Update fullmove number
    if color == Color.BLACK:
        new_state.fullmove_number += 1

    # Switch turn
    new_state.turn = Color.BLACK if color == Color.WHITE else Color.WHITE

    # Add position to history (simple hash based on board + turn)
    pos_hash = hash((tuple(new_state.board), new_state.turn))
    new_state.position_history.append(pos_hash)

    return new_state


def generate_legal_moves(state: GameState) -> list:
    """Generate all legal moves (filters out moves that leave king in check)."""
    pseudo_legal = generate_pseudo_legal_moves(state)
    legal_moves = []

    for move in pseudo_legal:
        new_state = make_move(state, move)
        # Check if our king is in check after the move
        if not is_in_check(new_state, state.turn):
            legal_moves.append(move)

    return legal_moves


def get_game_result(state: GameState) -> Optional[float]:
    """
    Check if the game is over.
    Returns:
        1.0 if white wins
        -1.0 if black wins
        0.0 if draw
        None if game is ongoing
    """
    legal_moves = generate_legal_moves(state)

    if len(legal_moves) == 0:
        if is_in_check(state, state.turn):
            # Checkmate
            return -1.0 if state.turn == Color.WHITE else 1.0
        else:
            # Stalemate
            return 0.0

    # 50-move rule
    if state.halfmove_clock >= 100:  # 50 full moves = 100 half moves
        return 0.0

    # Threefold repetition
    if len(state.position_history) >= 3:
        current_hash = state.position_history[-1]
        if state.position_history.count(current_hash) >= 3:
            return 0.0

    # Insufficient material (simplified)
    pieces = []
    for sq_idx in range(64):
        piece = state.board[sq_idx]
        if piece != 0:
            pieces.append((get_piece_type(piece), get_piece_color(piece)))

    if len(pieces) == 2:  # K vs K
        return 0.0

    if len(pieces) == 3:  # K+minor vs K
        for ptype, _ in pieces:
            if ptype in [Piece.BISHOP, Piece.KNIGHT]:
                return 0.0

    return None


# Test
if __name__ == "__main__":
    print("=== Standard Position ===")
    state = create_standard_position()
    print_board(state)

    moves = generate_legal_moves(state)
    print(f"Legal moves: {len(moves)}")
    print(f"First 10 moves: {moves[:10]}")

    print("\n=== Random Position ===")
    state = create_starting_position()
    print_board(state)

    moves = generate_legal_moves(state)
    print(f"Legal moves: {len(moves)}")
    print(f"First 10 moves: {moves[:10]}")

    print("\n=== Play a few moves ===")
    for _ in range(4):
        if moves:
            move = random.choice(moves)
            print(f"Playing: {move}")
            state = make_move(state, move)
            print_board(state)
            moves = generate_legal_moves(state)
