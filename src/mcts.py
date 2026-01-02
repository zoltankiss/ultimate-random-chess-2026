"""
Ultimate Random Chess - Monte Carlo Tree Search

AlphaZero-style MCTS with neural network guidance.
"""

import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from game import (
    GameState, Move, Color,
    generate_legal_moves, make_move, get_game_result
)
from network import (
    ChessNetwork, NetworkConfig,
    encode_board, move_to_policy_index, get_policy_mask, get_device
)


@dataclass
class MCTSConfig:
    """MCTS hyperparameters."""
    num_simulations: int = 100      # Reduced for M2 (AlphaZero uses 800)
    c_puct: float = 1.5              # Exploration constant
    dirichlet_alpha: float = 0.3    # Noise for exploration at root
    dirichlet_epsilon: float = 0.25  # Weight of noise at root
    temperature: float = 1.0         # Temperature for move selection
    temperature_threshold: int = 30  # Move number after which temp drops to 0


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""
    state: GameState
    parent: Optional['MCTSNode'] = None
    move: Optional[Move] = None      # Move that led to this node

    # Statistics
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0

    # Children
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)  # policy_idx -> node
    is_expanded: bool = False

    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visit_count: int, c_puct: float) -> float:
        """
        Calculate UCB score for node selection.

        UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N)
        """
        exploration = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        return self.value + exploration


class MCTS:
    """Monte Carlo Tree Search with neural network guidance."""

    def __init__(
        self,
        network: ChessNetwork,
        config: MCTSConfig = None,
        device: torch.device = None
    ):
        self.network = network
        self.config = config or MCTSConfig()
        self.device = device or get_device()
        self.network.to(self.device)
        self.network.eval()

    def search(
        self,
        state: GameState,
        history: List[GameState] = None,
        add_noise: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Run MCTS from the given state.

        Args:
            state: Current game state
            history: List of previous states (for network input)
            add_noise: Whether to add Dirichlet noise at root

        Returns:
            policy: Probability distribution over moves (visits-based)
            value: Estimated value of the position
        """
        root = MCTSNode(state=state)

        # Expand root
        self._expand(root, history, add_noise=add_noise)

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]

            # Selection: traverse tree using UCB
            while node.is_expanded and node.children:
                node = self._select_child(node)
                search_path.append(node)

            # Check for terminal state
            result = get_game_result(node.state)

            if result is not None:
                # Terminal node
                value = result
                # Adjust for perspective
                if node.state.turn == Color.BLACK:
                    value = -value
            else:
                # Expand and evaluate
                value = self._expand(node, history)

            # Backprop
            self._backpropagate(search_path, value)

        # Create policy from visit counts
        policy = np.zeros(NetworkConfig.POLICY_SIZE, dtype=np.float32)
        for idx, child in root.children.items():
            policy[idx] = child.visit_count

        # Normalize
        if policy.sum() > 0:
            policy /= policy.sum()

        return policy, root.value

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select the child with highest UCB score."""
        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            score = child.ucb_score(node.visit_count, self.config.c_puct)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand(
        self,
        node: MCTSNode,
        history: List[GameState] = None,
        add_noise: bool = False
    ) -> float:
        """
        Expand a node using the neural network.

        Returns the value estimate for this position.
        """
        state = node.state

        # Get legal moves
        legal_moves = generate_legal_moves(state)
        if not legal_moves:
            node.is_expanded = True
            return 0.0  # No moves = stalemate, handled elsewhere

        # Get network prediction
        board_tensor = torch.tensor(encode_board(state, history)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.network(board_tensor)

        policy_logits = policy_logits.cpu().numpy()[0]
        value = value.cpu().item()

        # Get policy mask and apply
        mask = get_policy_mask(state)
        policy_logits[mask == 0] = -float('inf')

        # Softmax
        policy_logits = policy_logits - policy_logits.max()
        policy = np.exp(policy_logits)
        policy = policy / (policy.sum() + 1e-8)

        # Add Dirichlet noise at root
        if add_noise:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(legal_moves))
            noise_full = np.zeros_like(policy)
            for i, move in enumerate(legal_moves):
                idx = move_to_policy_index(move, state.turn)
                noise_full[idx] = noise[i]

            policy = (1 - self.config.dirichlet_epsilon) * policy + \
                     self.config.dirichlet_epsilon * noise_full

        # Create child nodes
        for move in legal_moves:
            idx = move_to_policy_index(move, state.turn)
            child_state = make_move(state, move)
            child = MCTSNode(
                state=child_state,
                parent=node,
                move=move,
                prior=policy[idx]
            )
            node.children[idx] = child

        node.is_expanded = True

        # Return value from current player's perspective
        return value

    def _backpropagate(self, search_path: List[MCTSNode], value: float):
        """Backpropagate value through the search path."""
        for node in reversed(search_path):
            node.visit_count += 1
            # Value alternates sign as we go up the tree
            node.value_sum += value
            value = -value  # Flip for opponent's perspective

    def select_move(
        self,
        state: GameState,
        history: List[GameState] = None,
        temperature: float = None
    ) -> Tuple[Move, np.ndarray]:
        """
        Select a move using MCTS.

        Args:
            state: Current game state
            history: Previous states
            temperature: Temperature for move selection (0 = greedy)

        Returns:
            move: Selected move
            policy: Full policy distribution
        """
        policy, _ = self.search(state, history)

        temp = temperature if temperature is not None else self.config.temperature

        # Get legal moves
        legal_moves = generate_legal_moves(state)

        if temp == 0:
            # Greedy selection
            best_idx = policy.argmax()
            for move in legal_moves:
                if move_to_policy_index(move, state.turn) == best_idx:
                    return move, policy
        else:
            # Sample according to policy^(1/temp)
            move_probs = []
            for move in legal_moves:
                idx = move_to_policy_index(move, state.turn)
                move_probs.append((move, policy[idx]))

            probs = np.array([p ** (1 / temp) for _, p in move_probs])
            probs = probs / probs.sum()

            idx = np.random.choice(len(legal_moves), p=probs)
            return legal_moves[idx], policy

        # Fallback: random legal move
        return np.random.choice(legal_moves), policy


def play_game(
    network: ChessNetwork,
    mcts_config: MCTSConfig = None,
    device: torch.device = None,
    verbose: bool = False,
    return_moves: bool = False
) -> List[Tuple[GameState, np.ndarray, float]]:
    """
    Play a full game using MCTS, collecting training data.

    Args:
        network: Neural network for evaluation
        mcts_config: MCTS configuration
        device: Torch device
        verbose: Print game as it progresses
        return_moves: If True, also return move history for PGN export

    Returns:
        If return_moves=False:
            List of (state, policy, value) tuples for training.
        If return_moves=True:
            Tuple of (training_data, move_history, result) where
            move_history is List of (state, move) tuples for PGN export
    """
    from game import create_starting_position, print_board

    mcts = MCTS(network, mcts_config, device)
    state = create_starting_position()
    history = []
    game_data = []
    move_history = []  # For PGN export

    move_count = 0

    while True:
        # Check game over
        result = get_game_result(state)
        if result is not None:
            break

        # Determine temperature
        if move_count < mcts.config.temperature_threshold:
            temp = mcts.config.temperature
        else:
            temp = 0.0

        # Get move
        move, policy = mcts.select_move(state, history, temperature=temp)

        if verbose:
            print(f"\nMove {move_count + 1}: {move}")
            print_board(state)

        # Store training data (value filled in later)
        game_data.append((state.copy(), policy, None))

        # Store move for PGN
        move_history.append((state.copy(), move))

        # Make move
        history = [state] + history[:7]  # Keep last 8 positions
        state = make_move(state, move)
        move_count += 1

        # Safety limit
        if move_count > 300:
            result = 0.0  # Draw by move limit
            break

    if verbose:
        print(f"\nGame over! Result: {result}")
        print_board(state)

    # Fill in values from game result
    training_data = []
    for i, (s, p, _) in enumerate(game_data):
        # Result from white's perspective
        # Adjust for whose turn it was
        if s.turn == Color.WHITE:
            value = result
        else:
            value = -result
        training_data.append((s, p, value))

    if return_moves:
        return training_data, move_history, result
    return training_data


# Test
if __name__ == "__main__":
    from game import create_standard_position, print_board

    print("Testing MCTS...")

    device = get_device()
    print(f"Device: {device}")

    # Create network
    network = ChessNetwork().to(device)

    # Create MCTS with reduced simulations for testing
    config = MCTSConfig(num_simulations=50)
    mcts = MCTS(network, config, device)

    # Test on starting position
    state = create_standard_position()
    print_board(state)

    print("\nRunning MCTS search (50 simulations)...")
    policy, value = mcts.search(state)

    print(f"Value estimate: {value:.4f}")

    # Show top moves
    legal_moves = generate_legal_moves(state)
    move_probs = []
    for move in legal_moves:
        idx = move_to_policy_index(move, state.turn)
        move_probs.append((move, policy[idx]))

    move_probs.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 5 moves by visit count:")
    for move, prob in move_probs[:5]:
        print(f"  {move}: {prob:.4f}")

    # Select a move
    move, _ = mcts.select_move(state, temperature=1.0)
    print(f"\nSelected move: {move}")

    # Play a quick game
    print("\n" + "="*50)
    print("Playing a test game (may take a minute)...")
    print("="*50)

    config = MCTSConfig(num_simulations=25)  # Even fewer for speed
    game_data = play_game(network, config, device, verbose=True)

    print(f"\nGame length: {len(game_data)} moves")
    print(f"Training samples: {len(game_data)}")
