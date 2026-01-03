"""
Ultimate Random Chess - Parallel MCTS with Batched Inference

Runs multiple MCTS searches in parallel, batching neural network calls
across all searches for efficient GPU utilization.
"""

import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from game import (
    GameState, Move, Color,
    generate_legal_moves, make_move, get_game_result, create_starting_position
)
from network import (
    ChessNetwork, NetworkConfig,
    encode_board, move_to_policy_index, get_policy_mask, get_device
)
from mcts import MCTSConfig, MCTSNode


class BatchedMCTS:
    """
    MCTS that batches neural network evaluations across multiple parallel searches.

    Instead of running one MCTS search at a time (sequential), this runs N searches
    in lockstep, collecting leaf nodes from all searches and evaluating them in
    a single batched GPU call.
    """

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

    def search_batch(
        self,
        states: List[GameState],
        histories: List[List[GameState]] = None,
        add_noise: bool = True
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Run MCTS search on multiple states simultaneously, batching NN calls.

        Args:
            states: List of game states to search from
            histories: List of history lists for each state
            add_noise: Whether to add Dirichlet noise at roots

        Returns:
            List of (policy, value) tuples for each state
        """
        n = len(states)
        if n == 0:
            return []

        if histories is None:
            histories = [[] for _ in range(n)]

        # Initialize root nodes
        roots = [MCTSNode(state=s) for s in states]

        # Get legal moves for each root
        legal_moves_list = [generate_legal_moves(s) for s in states]

        # Check for terminal states
        active_indices = []
        results = [None] * n

        for i, (root, legal_moves) in enumerate(zip(roots, legal_moves_list)):
            result = get_game_result(root.state)
            if result is not None or not legal_moves:
                # Terminal state - return uniform policy and result
                policy = np.zeros(NetworkConfig.POLICY_SIZE, dtype=np.float32)
                results[i] = (policy, result if result is not None else 0.0)
            else:
                active_indices.append(i)

        if not active_indices:
            return results

        # Batch expand all active roots
        active_roots = [roots[i] for i in active_indices]
        active_histories = [histories[i] for i in active_indices]
        active_legal_moves = [legal_moves_list[i] for i in active_indices]

        self._expand_batch(active_roots, active_histories, active_legal_moves, add_noise=add_noise)

        # Run simulations
        for sim in range(self.config.num_simulations):
            # Selection phase - find leaf for each active tree
            leaves = []
            paths = []
            leaf_histories = []
            needs_expansion = []  # Track which leaves need NN evaluation

            for idx, (root, history) in enumerate(zip(active_roots, active_histories)):
                node = root
                path = [node]

                # Traverse tree using UCB
                while node.is_expanded and node.children:
                    node = self._select_child(node)
                    path.append(node)

                leaves.append(node)
                paths.append(path)
                leaf_histories.append(history)

                # Check if this leaf needs expansion
                result = get_game_result(node.state)
                needs_expansion.append(result is None and not node.is_expanded)

            # Batch evaluate leaves that need expansion
            expand_indices = [i for i, need in enumerate(needs_expansion) if need]

            if expand_indices:
                expand_leaves = [leaves[i] for i in expand_indices]
                expand_histories = [leaf_histories[i] for i in expand_indices]
                expand_legal_moves = [generate_legal_moves(leaves[i].state) for i in expand_indices]

                values = self._expand_batch(expand_leaves, expand_histories, expand_legal_moves)

                # Map values back
                value_map = {expand_indices[i]: values[i] for i in range(len(expand_indices))}
            else:
                value_map = {}

            # Backpropagate for all paths
            for idx, (path, leaf) in enumerate(zip(paths, leaves)):
                if idx in value_map:
                    value = value_map[idx]
                else:
                    # Terminal node - get actual result
                    result = get_game_result(leaf.state)
                    if result is not None:
                        value = result
                        if leaf.state.turn == Color.BLACK:
                            value = -value
                    else:
                        value = leaf.value if leaf.visit_count > 0 else 0.0

                self._backpropagate(path, value)

        # Extract policies from roots
        for i, idx in enumerate(active_indices):
            policy = np.zeros(NetworkConfig.POLICY_SIZE, dtype=np.float32)
            root = active_roots[i]

            for child_idx, child in root.children.items():
                policy[child_idx] = child.visit_count

            if policy.sum() > 0:
                policy /= policy.sum()

            results[idx] = (policy, root.value)

        return results

    def _expand_batch(
        self,
        nodes: List[MCTSNode],
        histories: List[List[GameState]],
        legal_moves_list: List[List[Move]],
        add_noise: bool = False
    ) -> List[float]:
        """
        Expand multiple nodes in a single batched network call.

        Returns list of value estimates.
        """
        n = len(nodes)
        if n == 0:
            return []

        # Encode all boards
        board_tensors = []
        for node, history in zip(nodes, histories):
            encoded = encode_board(node.state, history)
            board_tensors.append(encoded)

        # Batch forward pass
        batch = torch.tensor(np.array(board_tensors), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            policy_logits_batch, values_batch = self.network(batch)

        policy_logits_batch = policy_logits_batch.cpu().numpy()
        values_batch = values_batch.cpu().numpy().flatten()

        # Process each node
        values = []
        for i, (node, legal_moves) in enumerate(zip(nodes, legal_moves_list)):
            if not legal_moves:
                node.is_expanded = True
                values.append(0.0)
                continue

            policy_logits = policy_logits_batch[i]
            value = float(values_batch[i])

            # Apply mask
            mask = get_policy_mask(node.state)
            policy_logits[mask == 0] = -float('inf')

            # Softmax
            policy_logits = policy_logits - policy_logits.max()
            policy = np.exp(policy_logits)
            policy = policy / (policy.sum() + 1e-8)

            # Add Dirichlet noise at root
            if add_noise and len(legal_moves) > 0:
                noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(legal_moves))
                noise_full = np.zeros_like(policy)
                for j, move in enumerate(legal_moves):
                    idx = move_to_policy_index(move, node.state.turn)
                    noise_full[idx] = noise[j]

                policy = (1 - self.config.dirichlet_epsilon) * policy + \
                         self.config.dirichlet_epsilon * noise_full

            # Create children
            for move in legal_moves:
                idx = move_to_policy_index(move, node.state.turn)
                child_state = make_move(node.state, move)
                child = MCTSNode(
                    state=child_state,
                    parent=node,
                    move=move,
                    prior=policy[idx]
                )
                node.children[idx] = child

            node.is_expanded = True
            values.append(value)

        return values

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            score = child.ucb_score(node.visit_count, self.config.c_puct)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _backpropagate(self, path: List[MCTSNode], value: float):
        """Backpropagate value through search path."""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value


class ParallelSelfPlay:
    """
    Manages parallel self-play games with batched inference.

    Runs N games simultaneously, batching neural network calls across all games.
    """

    def __init__(
        self,
        network: ChessNetwork,
        mcts_config: MCTSConfig,
        device: torch.device,
        num_parallel: int = 8
    ):
        self.network = network
        self.mcts_config = mcts_config
        self.device = device
        self.num_parallel = num_parallel
        self.batched_mcts = BatchedMCTS(network, mcts_config, device)

    def play_games(
        self,
        num_games: int,
        verbose: bool = False,
        return_moves: bool = False
    ) -> List:
        """
        Play multiple games using parallel self-play.

        Args:
            num_games: Total number of games to play
            verbose: Print progress
            return_moves: Include move history for PGN export

        Returns:
            List of game results (training_data or (training_data, moves, result))
        """
        all_results = []
        games_completed = 0

        while games_completed < num_games:
            # Determine batch size for this round
            batch_size = min(self.num_parallel, num_games - games_completed)

            # Play a batch of games
            batch_results = self._play_game_batch(batch_size, return_moves)
            all_results.extend(batch_results)
            games_completed += batch_size

            if verbose:
                print(f"Completed {games_completed}/{num_games} games")

        return all_results

    def _play_game_batch(
        self,
        batch_size: int,
        return_moves: bool = False
    ) -> List:
        """Play a batch of games in parallel."""

        # Initialize games
        games = []
        for _ in range(batch_size):
            games.append({
                'state': create_starting_position(),
                'history': [],
                'game_data': [],
                'move_history': [],
                'done': False,
                'result': None,
                'move_count': 0
            })

        # Play until all games are done
        while not all(g['done'] for g in games):
            # Collect active games
            active_indices = [i for i, g in enumerate(games) if not g['done']]

            if not active_indices:
                break

            active_states = [games[i]['state'] for i in active_indices]
            active_histories = [games[i]['history'] for i in active_indices]

            # Determine temperatures
            temps = []
            for i in active_indices:
                if games[i]['move_count'] < self.mcts_config.temperature_threshold:
                    temps.append(self.mcts_config.temperature)
                else:
                    temps.append(0.0)

            # Batch MCTS search
            search_results = self.batched_mcts.search_batch(
                active_states,
                active_histories,
                add_noise=True
            )

            # Process results and make moves
            for idx, i in enumerate(active_indices):
                game = games[i]
                policy, value = search_results[idx]
                temp = temps[idx]

                # Select move
                legal_moves = generate_legal_moves(game['state'])

                if not legal_moves:
                    game['done'] = True
                    game['result'] = get_game_result(game['state']) or 0.0
                    continue

                move = self._select_move(legal_moves, policy, game['state'].turn, temp)

                # Store training data
                game['game_data'].append((game['state'].copy(), policy, None))

                if return_moves:
                    game['move_history'].append((game['state'].copy(), move))

                # Make move
                game['history'] = [game['state']] + game['history'][:7]
                game['state'] = make_move(game['state'], move)
                game['move_count'] += 1

                # Check game over
                result = get_game_result(game['state'])
                if result is not None:
                    game['done'] = True
                    game['result'] = result
                elif game['move_count'] > 300:
                    game['done'] = True
                    game['result'] = 0.0  # Draw by move limit

        # Finalize training data with values
        results = []
        for game in games:
            training_data = []
            for state, policy, _ in game['game_data']:
                if state.turn == Color.WHITE:
                    value = game['result']
                else:
                    value = -game['result']
                training_data.append((state, policy, value))

            if return_moves:
                results.append((training_data, game['move_history'], game['result']))
            else:
                results.append(training_data)

        return results

    def _select_move(
        self,
        legal_moves: List[Move],
        policy: np.ndarray,
        turn: Color,
        temperature: float
    ) -> Move:
        """Select a move based on policy and temperature."""

        if temperature == 0:
            # Greedy
            best_idx = policy.argmax()
            for move in legal_moves:
                if move_to_policy_index(move, turn) == best_idx:
                    return move
            # Fallback
            return legal_moves[0]
        else:
            # Sample with temperature
            move_probs = []
            for move in legal_moves:
                idx = move_to_policy_index(move, turn)
                move_probs.append((move, policy[idx]))

            probs = np.array([p ** (1 / temperature) for _, p in move_probs])
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(len(legal_moves)) / len(legal_moves)

            idx = np.random.choice(len(legal_moves), p=probs)
            return legal_moves[idx]


# Test
if __name__ == "__main__":
    import time

    print("Testing Parallel MCTS...")

    device = get_device()
    print(f"Device: {device}")

    network = ChessNetwork().to(device)
    config = MCTSConfig(num_simulations=50)

    # Test batched search
    print("\nTesting batched MCTS search...")
    batched_mcts = BatchedMCTS(network, config, device)

    states = [create_starting_position() for _ in range(8)]

    start = time.time()
    results = batched_mcts.search_batch(states)
    elapsed = time.time() - start

    print(f"Searched 8 positions in {elapsed:.2f}s ({elapsed/8:.2f}s per position)")

    # Test parallel self-play
    print("\nTesting parallel self-play (4 games, 8 parallel)...")
    parallel = ParallelSelfPlay(network, config, device, num_parallel=8)

    start = time.time()
    games = parallel.play_games(4, verbose=True)
    elapsed = time.time() - start

    print(f"Played 4 games in {elapsed:.2f}s ({elapsed/4:.2f}s per game)")
    print(f"Total positions: {sum(len(g) for g in games)}")
