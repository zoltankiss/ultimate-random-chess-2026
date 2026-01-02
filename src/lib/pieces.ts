import { PieceDefinition } from './types';

// Embedded piece definitions (from YAML files)
// This avoids needing to load YAML at runtime in the browser

export const standardPieces: Record<string, PieceDefinition> = {
  king: {
    name: "King",
    symbol: "K",
    value: 0,
    royal: true,
    moves: [
      {
        type: 'slide',
        directions: [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]],
        range: 1
      }
    ]
  },
  queen: {
    name: "Queen",
    symbol: "Q",
    value: 9,
    moves: [
      {
        type: 'slide',
        directions: [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]],
        range: 'unlimited'
      }
    ]
  },
  rook: {
    name: "Rook",
    symbol: "R",
    value: 5,
    moves: [
      {
        type: 'slide',
        directions: [[1, 0], [-1, 0], [0, 1], [0, -1]],
        range: 'unlimited'
      }
    ]
  },
  bishop: {
    name: "Bishop",
    symbol: "B",
    value: 3,
    moves: [
      {
        type: 'slide',
        directions: [[1, 1], [1, -1], [-1, 1], [-1, -1]],
        range: 'unlimited'
      }
    ]
  },
  knight: {
    name: "Knight",
    symbol: "N",
    value: 3,
    moves: [
      {
        type: 'leap',
        vectors: [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]]
      }
    ]
  },
  pawn: {
    name: "Pawn",
    symbol: "P",
    value: 1,
    moves: [
      { type: 'slide', directions: [[0, 1]], range: 1, move_only: true },
      { type: 'slide', directions: [[0, 1]], range: 2, move_only: true, initial_only: true },
      { type: 'slide', directions: [[1, 1], [-1, 1]], range: 1, capture_only: true }
    ]
  }
};

export const customPieces: Record<string, PieceDefinition> = {
  boulder: {
    name: "Boulder",
    symbol: "O",
    value: 4,
    description: "Slides like a king, but up to 2 squares in any direction",
    moves: [
      {
        type: 'slide',
        directions: [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]],
        range: 2
      }
    ]
  },
  archbishop: {
    name: "Archbishop",
    symbol: "A",
    value: 7,
    description: "Combines knight and bishop movement",
    moves: [
      { type: 'leap', vectors: [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]] },
      { type: 'slide', directions: [[1, 1], [1, -1], [-1, 1], [-1, -1]], range: 'unlimited' }
    ]
  },
  sugeknight: {
    name: "Sugeknight",
    symbol: "S",
    value: 3.5,
    description: "Like a knight, but leaps (3,1) instead of (2,1)",
    moves: [
      { type: 'leap', vectors: [[3, 1], [3, -1], [-3, 1], [-3, -1], [1, 3], [1, -3], [-1, 3], [-1, -3]] }
    ]
  },
  pope: {
    name: "Pope",
    symbol: "+",
    value: 7,
    description: "Combines sugeknight and bishop movement",
    moves: [
      { type: 'leap', vectors: [[3, 1], [3, -1], [-3, 1], [-3, -1], [1, 3], [1, -3], [-1, 3], [-1, -3]] },
      { type: 'slide', directions: [[1, 1], [1, -1], [-1, 1], [-1, -1]], range: 'unlimited' }
    ]
  },
  chancellor: {
    name: "Chancellor",
    symbol: "C",
    value: 8,
    description: "Combines knight and rook movement",
    moves: [
      { type: 'leap', vectors: [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]] },
      { type: 'slide', directions: [[1, 0], [-1, 0], [0, 1], [0, -1]], range: 'unlimited' }
    ]
  },
  amazon: {
    name: "Amazon",
    symbol: "M",
    value: 12,
    description: "Combines queen and knight - the most powerful standard piece",
    moves: [
      { type: 'leap', vectors: [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]] },
      { type: 'slide', directions: [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]], range: 'unlimited' }
    ]
  },
  zebra: {
    name: "Zebra",
    symbol: "Z",
    value: 2.5,
    description: "Leaps (3,2)",
    moves: [
      { type: 'leap', vectors: [[3, 2], [3, -2], [-3, 2], [-3, -2], [2, 3], [2, -3], [-2, 3], [-2, -3]] }
    ]
  },
  nightrider: {
    name: "Nightrider",
    symbol: "NN",
    value: 5,
    description: "Slides in knight directions - can make multiple knight jumps in a line",
    moves: [
      { type: 'slide', directions: [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]], range: 'unlimited' }
    ]
  },
  grasshopper: {
    name: "Grasshopper",
    symbol: "G",
    value: 2,
    description: "Moves along queen lines, but must hop over exactly one piece and land immediately behind it",
    moves: [
      { type: 'hopper', directions: [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]], hurdle_count: 1, land_distance: 1 }
    ]
  },
  cannon: {
    name: "Cannon",
    symbol: "X",
    value: 4.5,
    description: "Xiangqi cannon - slides like rook to move, but must hop over exactly one piece to capture",
    moves: [
      { type: 'slide', directions: [[1, 0], [-1, 0], [0, 1], [0, -1]], range: 'unlimited', move_only: true },
      { type: 'hopper', directions: [[1, 0], [-1, 0], [0, 1], [0, -1]], hurdle_count: 1, capture_only: true }
    ]
  }
};

export const allPieces: Record<string, PieceDefinition> = {
  ...standardPieces,
  ...customPieces
};

// Pieces eligible for back-rank (excluding king, rook, pawn)
export const backRankPieces = Object.keys(allPieces).filter(
  k => !['king', 'rook', 'pawn'].includes(k)
);
