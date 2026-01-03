"""
Ultimate Random Chess - Neural Network

AlphaZero-style network with policy and value heads.
Optimized for M2 Max with MPS backend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from game import (
    GameState, Color, Piece, Move,
    get_piece_type, get_piece_color, generate_legal_moves,
    file_of, rank_of, sq, PIECE_SYMBOLS
)


# Network configuration (smaller for M2 Max proof-of-concept)
class NetworkConfig:
    # Input planes
    # Ultimate Random Chess: P, N, B, R, Q, K + Archbishop, Chancellor, Amazon, Camel, Zebra, Grasshopper, Cannon = 13
    NUM_PIECE_TYPES = 13
    HISTORY_LENGTH = 8   # Current + 7 previous positions

    # Per timestep: 13 white pieces + 13 black pieces + 2 repetition = 28
    PLANES_PER_TIMESTEP = NUM_PIECE_TYPES * 2 + 2

    # Meta planes: side to move, move count, 4 castling rights, no-progress count = 7
    META_PLANES = 7

    # Total input planes
    INPUT_PLANES = PLANES_PER_TIMESTEP * HISTORY_LENGTH + META_PLANES  # 28*8 + 7 = 231

    # Network architecture (sized for GPU training)
    NUM_RESIDUAL_BLOCKS = 15  # AlphaZero uses 19
    NUM_FILTERS = 256         # AlphaZero uses 256

    # Output: 73 move planes (queen moves + knight moves + underpromotions)
    # - 56 queen moves: 8 directions * 7 distances
    # - 8 knight moves
    # - 9 underpromotions: 3 directions * 3 pieces (N, B, R)
    POLICY_PLANES = 73
    POLICY_SIZE = 64 * POLICY_PLANES  # 4672 possible moves


class ResidualBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ChessNetwork(nn.Module):
    """
    AlphaZero-style neural network for chess.

    Takes board state as input, outputs:
    - Policy: probability distribution over moves
    - Value: expected game outcome from current position
    """

    def __init__(self, config: NetworkConfig = None):
        super().__init__()
        self.config = config or NetworkConfig()

        # Input convolution
        self.input_conv = nn.Conv2d(
            self.config.INPUT_PLANES,
            self.config.NUM_FILTERS,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.input_bn = nn.BatchNorm2d(self.config.NUM_FILTERS)

        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(self.config.NUM_FILTERS)
            for _ in range(self.config.NUM_RESIDUAL_BLOCKS)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(self.config.NUM_FILTERS, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 64, self.config.POLICY_SIZE)

        # Value head
        self.value_conv = nn.Conv2d(self.config.NUM_FILTERS, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, INPUT_PLANES, 8, 8)

        Returns:
            policy: Raw logits of shape (batch, POLICY_SIZE)
            value: Value estimate of shape (batch, 1) in range [-1, 1]
        """
        # Input convolution
        out = F.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        for block in self.residual_blocks:
            out = block(out)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


def encode_board(state: GameState, history: list = None) -> np.ndarray:
    """
    Encode a board position into neural network input planes.

    Args:
        state: Current game state
        history: List of previous GameStates (most recent first)

    Returns:
        numpy array of shape (INPUT_PLANES, 8, 8)
    """
    config = NetworkConfig()
    planes = np.zeros((config.INPUT_PLANES, 8, 8), dtype=np.float32)

    # Determine perspective (always from current player's view)
    flip = state.turn == Color.BLACK

    def encode_position(board: list, plane_offset: int, flip_board: bool):
        """Encode a single position into planes."""
        for sq_idx in range(64):
            piece = board[sq_idx]
            if piece == 0:
                continue

            piece_type = get_piece_type(piece)
            piece_color = get_piece_color(piece)

            # Get file and rank
            file_idx = file_of(sq_idx)
            rank_idx = rank_of(sq_idx)

            # Flip board if playing as black
            if flip_board:
                rank_idx = 7 - rank_idx
                # Also flip piece colors from our perspective
                piece_color = Color.BLACK if piece_color == Color.WHITE else Color.WHITE

            # Plane index: 0-12 for white pieces, 13-25 for black pieces (13 piece types)
            piece_plane = (piece_type.value - 1) + (piece_color * config.NUM_PIECE_TYPES)
            planes[plane_offset + piece_plane, rank_idx, file_idx] = 1.0

    # Encode current position
    encode_position(state.board, 0, flip)

    # Encode history (if available)
    if history:
        for t, hist_state in enumerate(history[:config.HISTORY_LENGTH - 1]):
            plane_offset = (t + 1) * config.PLANES_PER_TIMESTEP
            encode_position(hist_state.board, plane_offset, flip)

    # Repetition counters (planes 26, 27 for each timestep with 13 piece types)
    # Simplified: just mark if position has been seen before
    for t in range(config.HISTORY_LENGTH):
        rep_plane_base = t * config.PLANES_PER_TIMESTEP + (config.NUM_PIECE_TYPES * 2)
        # Would need proper repetition tracking here
        # For now, leave as zeros

    # Meta planes (after all timestep planes)
    meta_offset = config.PLANES_PER_TIMESTEP * config.HISTORY_LENGTH

    # Side to move (always 1 from current player's perspective after flip)
    planes[meta_offset + 0, :, :] = 1.0

    # Total move count (normalized)
    planes[meta_offset + 1, :, :] = min(state.fullmove_number / 100.0, 1.0)

    # Castling rights (from current player's perspective)
    if flip:
        # Black's turn - swap white/black castling
        planes[meta_offset + 2, :, :] = float(state.black_can_castle_kingside)
        planes[meta_offset + 3, :, :] = float(state.black_can_castle_queenside)
        planes[meta_offset + 4, :, :] = float(state.white_can_castle_kingside)
        planes[meta_offset + 5, :, :] = float(state.white_can_castle_queenside)
    else:
        planes[meta_offset + 2, :, :] = float(state.white_can_castle_kingside)
        planes[meta_offset + 3, :, :] = float(state.white_can_castle_queenside)
        planes[meta_offset + 4, :, :] = float(state.black_can_castle_kingside)
        planes[meta_offset + 5, :, :] = float(state.black_can_castle_queenside)

    # No-progress count (for 50-move rule, normalized)
    planes[meta_offset + 6, :, :] = min(state.halfmove_clock / 100.0, 1.0)

    return planes


# Move encoding/decoding
# Move planes (73 total):
# - Planes 0-55: Queen-like moves (8 directions * 7 distances)
# - Planes 56-63: Knight moves (8 directions)
# - Planes 64-72: Underpromotions (3 directions * 3 pieces)

QUEEN_DIRECTIONS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1)
]

KNIGHT_DIRECTIONS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2)
]

# Underpromotion directions (from white's perspective)
PROMO_DIRECTIONS = [(-1, 1), (0, 1), (1, 1)]  # Left capture, forward, right capture
PROMO_PIECES = [Piece.KNIGHT, Piece.BISHOP, Piece.ROOK]


def move_to_policy_index(move: Move, turn: Color) -> int:
    """
    Convert a move to a policy index (0-4671).

    The index is: from_square * 73 + move_type
    where move_type encodes the direction and distance.
    """
    from_sq = move.from_sq
    to_sq = move.to_sq

    # Flip squares if black's turn (so policy is always from current player's perspective)
    if turn == Color.BLACK:
        from_sq = (7 - rank_of(from_sq)) * 8 + file_of(from_sq)
        to_sq = (7 - rank_of(to_sq)) * 8 + file_of(to_sq)

    from_file, from_rank = file_of(from_sq), rank_of(from_sq)
    to_file, to_rank = file_of(to_sq), rank_of(to_sq)

    df = to_file - from_file
    dr = to_rank - from_rank

    # Check for underpromotion
    if move.promotion and move.promotion != Piece.QUEEN:
        # Underpromotion
        if df == -1:
            dir_idx = 0
        elif df == 0:
            dir_idx = 1
        else:
            dir_idx = 2

        piece_idx = PROMO_PIECES.index(move.promotion)
        move_type = 64 + dir_idx * 3 + piece_idx

    # Check for knight move
    elif (df, dr) in KNIGHT_DIRECTIONS:
        dir_idx = KNIGHT_DIRECTIONS.index((df, dr))
        move_type = 56 + dir_idx

    # Queen-like move (or regular promotion to queen)
    else:
        # Normalize direction
        dist = max(abs(df), abs(dr))
        if dist == 0:
            # Shouldn't happen for valid moves
            return 0

        dir_df = df // dist if df != 0 else 0
        dir_dr = dr // dist if dr != 0 else 0

        dir_idx = QUEEN_DIRECTIONS.index((dir_df, dir_dr))
        move_type = dir_idx * 7 + (dist - 1)

    return from_sq * 73 + move_type


def policy_index_to_move(index: int, state: GameState) -> Move:
    """
    Convert a policy index back to a move.
    Only returns valid move if it's legal in the current position.
    """
    from_sq = index // 73
    move_type = index % 73

    turn = state.turn

    # Flip from_sq if black's turn
    if turn == Color.BLACK:
        from_sq = (7 - rank_of(from_sq)) * 8 + file_of(from_sq)

    from_file, from_rank = file_of(from_sq), rank_of(from_sq)

    if move_type < 56:
        # Queen-like move
        dir_idx = move_type // 7
        dist = (move_type % 7) + 1
        df, dr = QUEEN_DIRECTIONS[dir_idx]
        to_file = from_file + df * dist
        to_rank = from_rank + dr * dist

        # Flip back if black's turn
        if turn == Color.BLACK:
            to_rank = 7 - to_rank
            from_rank_orig = 7 - from_rank
            from_sq = from_rank_orig * 8 + from_file

        to_sq = to_rank * 8 + to_file

        # Check for promotion (pawn reaching last rank)
        promotion = None
        piece = state.board[from_sq]
        if piece != 0 and get_piece_type(piece) == Piece.PAWN:
            promo_rank = 7 if turn == Color.WHITE else 0
            if rank_of(to_sq) == promo_rank:
                promotion = Piece.QUEEN

        return Move(from_sq, to_sq, promotion=promotion)

    elif move_type < 64:
        # Knight move
        dir_idx = move_type - 56
        df, dr = KNIGHT_DIRECTIONS[dir_idx]
        to_file = from_file + df
        to_rank = from_rank + dr

        if turn == Color.BLACK:
            to_rank = 7 - to_rank
            from_rank_orig = 7 - from_rank
            from_sq = from_rank_orig * 8 + from_file

        to_sq = to_rank * 8 + to_file
        return Move(from_sq, to_sq)

    else:
        # Underpromotion
        under_idx = move_type - 64
        dir_idx = under_idx // 3
        piece_idx = under_idx % 3

        df = PROMO_DIRECTIONS[dir_idx][0]
        dr = PROMO_DIRECTIONS[dir_idx][1]

        to_file = from_file + df
        to_rank = from_rank + dr

        if turn == Color.BLACK:
            to_rank = 7 - to_rank
            from_rank_orig = 7 - from_rank
            from_sq = from_rank_orig * 8 + from_file

        to_sq = to_rank * 8 + to_file
        return Move(from_sq, to_sq, promotion=PROMO_PIECES[piece_idx])


def get_policy_mask(state: GameState) -> np.ndarray:
    """
    Get a mask of legal moves for the current position.
    Returns a boolean array of shape (POLICY_SIZE,).
    """
    mask = np.zeros(NetworkConfig.POLICY_SIZE, dtype=np.float32)
    legal_moves = generate_legal_moves(state)

    for move in legal_moves:
        idx = move_to_policy_index(move, state.turn)
        if 0 <= idx < NetworkConfig.POLICY_SIZE:
            mask[idx] = 1.0

    return mask


def get_device():
    """Get the best available device (MPS for M2 Max, otherwise CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# Test
if __name__ == "__main__":
    from game import create_standard_position, create_starting_position

    device = get_device()
    print(f"Using device: {device}")

    # Create network
    config = NetworkConfig()
    print(f"Input planes: {config.INPUT_PLANES}")
    print(f"Residual blocks: {config.NUM_RESIDUAL_BLOCKS}")
    print(f"Filters: {config.NUM_FILTERS}")

    network = ChessNetwork(config).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in network.parameters())
    print(f"Total parameters: {num_params:,}")

    # Test forward pass
    state = create_standard_position()
    board_tensor = torch.tensor(encode_board(state)).unsqueeze(0).to(device)

    print(f"\nInput shape: {board_tensor.shape}")

    with torch.no_grad():
        policy, value = network(board_tensor)

    print(f"Policy shape: {policy.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Value: {value.item():.4f}")

    # Test policy masking
    mask = get_policy_mask(state)
    legal_count = int(mask.sum())
    print(f"\nLegal moves: {legal_count}")

    # Apply mask and get probabilities
    policy_masked = policy.cpu().numpy()[0] * mask
    policy_masked[mask == 0] = -1e9  # Mask illegal moves
    probs = np.exp(policy_masked - policy_masked.max())
    probs = probs / probs.sum()

    # Show top moves
    legal_moves = generate_legal_moves(state)
    move_probs = []
    for move in legal_moves:
        idx = move_to_policy_index(move, state.turn)
        move_probs.append((move, probs[idx]))

    move_probs.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 5 moves:")
    for move, prob in move_probs[:5]:
        print(f"  {move}: {prob:.4f}")
