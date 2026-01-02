// Piece definition types (from YAML)
export type MoveType = 'slide' | 'leap' | 'hopper';

export interface SlideMove {
  type: 'slide';
  directions: [number, number][];
  range: number | 'unlimited';
  move_only?: boolean;
  capture_only?: boolean;
  initial_only?: boolean;
}

export interface LeapMove {
  type: 'leap';
  vectors: [number, number][];
}

export interface HopperMove {
  type: 'hopper';
  directions: [number, number][];
  hurdle_count: number;
  land_distance?: number;
  capture_only?: boolean;
}

export type PieceMove = SlideMove | LeapMove | HopperMove;

export interface PieceDefinition {
  name: string;
  symbol: string;
  value: number;
  royal?: boolean;
  description?: string;
  moves: PieceMove[];
  special_rules?: Record<string, unknown>[];
}

// Game state types
export type Color = 'white' | 'black';

export interface Piece {
  type: string; // key from piece definitions (e.g., 'king', 'archbishop')
  color: Color;
  hasMoved?: boolean;
}

export interface Square {
  piece: Piece | null;
}

export type Board = (Piece | null)[][];

export interface Position {
  row: number;
  col: number;
}

export interface Move {
  from: Position;
  to: Position;
  capture?: Piece;
  castling?: 'kingside' | 'queenside';
  enPassant?: boolean;
  promotion?: string;
}

export interface GameState {
  board: Board;
  turn: Color;
  moveHistory: Move[];
  selectedSquare: Position | null;
  validMoves: Position[];
  gameOver: boolean;
  winner: Color | null;
  check: boolean;
  pieces: Record<string, PieceDefinition>;
  castlingRights: {
    white: { kingside: boolean; queenside: boolean; kingCol: number; rookCols: [number, number] };
    black: { kingside: boolean; queenside: boolean; kingCol: number; rookCols: [number, number] };
  };
  enPassantSquare: Position | null;
}
