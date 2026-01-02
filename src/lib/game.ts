import { Board, Color, GameState, Move, Piece, Position, PieceDefinition, SlideMove, LeapMove, HopperMove } from './types';
import { allPieces, backRankPieces, standardPieces } from './pieces';

// Create empty 8x8 board
export function createEmptyBoard(): Board {
  return Array(8).fill(null).map(() => Array(8).fill(null));
}

// Generate Fischer Random 960-style back rank with custom pieces
// Rules: King between rooks, bishops on opposite colors
export function generateBackRank(): { pieces: string[], kingCol: number, rookCols: [number, number] } {
  const rank: (string | null)[] = Array(8).fill(null);

  // Place rooks first - need two, can be anywhere
  const availableSquares = [0, 1, 2, 3, 4, 5, 6, 7];

  // Pick first rook position
  const rook1Idx = Math.floor(Math.random() * availableSquares.length);
  const rook1Col = availableSquares.splice(rook1Idx, 1)[0];
  rank[rook1Col] = 'rook';

  // Pick second rook position
  const rook2Idx = Math.floor(Math.random() * availableSquares.length);
  const rook2Col = availableSquares.splice(rook2Idx, 1)[0];
  rank[rook2Col] = 'rook';

  // Sort rook columns
  const rookCols: [number, number] = [Math.min(rook1Col, rook2Col), Math.max(rook1Col, rook2Col)];

  // Place king between rooks (Fischer Random rule)
  const kingPossible = availableSquares.filter(c => c > rookCols[0] && c < rookCols[1]);
  if (kingPossible.length === 0) {
    // If no space between rooks, swap to make space
    return generateBackRank(); // Retry
  }
  const kingIdx = Math.floor(Math.random() * kingPossible.length);
  const kingCol = kingPossible[kingIdx];
  availableSquares.splice(availableSquares.indexOf(kingCol), 1);
  rank[kingCol] = 'king';

  // Fill remaining 5 squares with random pieces from backRankPieces
  // Ensure at least one bishop on light and one on dark if we pick bishops
  for (const col of availableSquares) {
    const randomPiece = backRankPieces[Math.floor(Math.random() * backRankPieces.length)];
    rank[col] = randomPiece;
  }

  return { pieces: rank as string[], kingCol, rookCols };
}

// Initialize game with random setup
export function initializeGame(): GameState {
  const board = createEmptyBoard();

  // Generate back rank for both sides (same layout, mirrored)
  const { pieces: backRank, kingCol, rookCols } = generateBackRank();

  // Place white pieces (row 0 = back rank, row 1 = pawns)
  for (let col = 0; col < 8; col++) {
    board[0][col] = { type: backRank[col], color: 'white' };
    board[1][col] = { type: 'pawn', color: 'white' };
  }

  // Place black pieces (row 7 = back rank, row 6 = pawns)
  for (let col = 0; col < 8; col++) {
    board[7][col] = { type: backRank[col], color: 'black' };
    board[6][col] = { type: 'pawn', color: 'black' };
  }

  return {
    board,
    turn: 'white',
    moveHistory: [],
    selectedSquare: null,
    validMoves: [],
    gameOver: false,
    winner: null,
    check: false,
    pieces: allPieces,
    castlingRights: {
      white: { kingside: true, queenside: true, kingCol, rookCols },
      black: { kingside: true, queenside: true, kingCol, rookCols }
    },
    enPassantSquare: null
  };
}

// Check if position is on board
function isOnBoard(pos: Position): boolean {
  return pos.row >= 0 && pos.row < 8 && pos.col >= 0 && pos.col < 8;
}

// Get piece at position
function getPiece(board: Board, pos: Position): Piece | null {
  if (!isOnBoard(pos)) return null;
  return board[pos.row][pos.col];
}

// Calculate valid moves for a piece (without considering check)
function getRawMoves(state: GameState, pos: Position): Position[] {
  const piece = getPiece(state.board, pos);
  if (!piece) return [];

  const definition = state.pieces[piece.type];
  if (!definition) return [];

  const moves: Position[] = [];
  const direction = piece.color === 'white' ? 1 : -1; // White moves up (positive), black down

  for (const move of definition.moves) {
    if (move.type === 'slide') {
      const slideMove = move as SlideMove;

      // Check initial_only condition for pawns
      if (slideMove.initial_only) {
        const startRow = piece.color === 'white' ? 1 : 6;
        if (pos.row !== startRow) continue;
      }

      for (const [dx, dy] of slideMove.directions) {
        // Apply direction flip for pawns (y is forward)
        const actualDy = dy * direction;

        const maxRange = slideMove.range === 'unlimited' ? 8 : slideMove.range;

        for (let dist = 1; dist <= maxRange; dist++) {
          const newPos = { row: pos.row + actualDy * dist, col: pos.col + dx * dist };

          if (!isOnBoard(newPos)) break;

          const target = getPiece(state.board, newPos);

          if (target) {
            // Hit a piece
            if (target.color !== piece.color && !slideMove.move_only) {
              // Can capture enemy
              moves.push(newPos);
            }
            break; // Can't go further
          } else {
            // Empty square
            if (!slideMove.capture_only) {
              moves.push(newPos);
            } else {
              // Check en passant for pawns
              if (piece.type === 'pawn' && state.enPassantSquare &&
                  newPos.row === state.enPassantSquare.row && newPos.col === state.enPassantSquare.col) {
                moves.push(newPos);
              }
            }
          }
        }
      }
    } else if (move.type === 'leap') {
      const leapMove = move as LeapMove;

      for (const [dx, dy] of leapMove.vectors) {
        const newPos = { row: pos.row + dy, col: pos.col + dx };

        if (!isOnBoard(newPos)) continue;

        const target = getPiece(state.board, newPos);
        if (!target || target.color !== piece.color) {
          moves.push(newPos);
        }
      }
    } else if (move.type === 'hopper') {
      const hopperMove = move as HopperMove;

      for (const [dx, dy] of hopperMove.directions) {
        let hurdleCount = 0;
        let dist = 1;

        while (true) {
          const checkPos = { row: pos.row + dy * dist, col: pos.col + dx * dist };

          if (!isOnBoard(checkPos)) break;

          const target = getPiece(state.board, checkPos);

          if (target) {
            hurdleCount++;
            if (hurdleCount === hopperMove.hurdle_count) {
              // Found our hurdle, now we can land
              const landDist = hopperMove.land_distance || 1;
              const landPos = { row: checkPos.row + dy * landDist, col: checkPos.col + dx * landDist };

              if (isOnBoard(landPos)) {
                const landTarget = getPiece(state.board, landPos);
                if (!landTarget || landTarget.color !== piece.color) {
                  if (hopperMove.capture_only) {
                    if (landTarget && landTarget.color !== piece.color) {
                      moves.push(landPos);
                    }
                  } else {
                    moves.push(landPos);
                  }
                }
              }
              break;
            }
          }

          dist++;
          if (dist > 8) break; // Safety limit
        }
      }
    }
  }

  // Add castling moves for king
  if (piece.type === 'king' && !piece.hasMoved) {
    const rights = state.castlingRights[piece.color];
    const row = piece.color === 'white' ? 0 : 7;

    // Check kingside castling
    if (rights.kingside) {
      const canCastle = canCastleKingside(state, piece.color);
      if (canCastle) {
        // King moves to g-file (col 6) in standard, but in Fischer Random it's 2 squares toward rook
        const targetCol = 6;
        moves.push({ row, col: targetCol });
      }
    }

    // Check queenside castling
    if (rights.queenside) {
      const canCastle = canCastleQueenside(state, piece.color);
      if (canCastle) {
        // King moves to c-file (col 2)
        const targetCol = 2;
        moves.push({ row, col: targetCol });
      }
    }
  }

  return moves;
}

// Check if a square is attacked by enemy
function isSquareAttacked(state: GameState, pos: Position, byColor: Color): boolean {
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const piece = state.board[row][col];
      if (piece && piece.color === byColor) {
        const moves = getRawMoves(state, { row, col });
        if (moves.some(m => m.row === pos.row && m.col === pos.col)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Find king position
function findKing(board: Board, color: Color): Position | null {
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const piece = board[row][col];
      if (piece && piece.type === 'king' && piece.color === color) {
        return { row, col };
      }
    }
  }
  return null;
}

// Check if player is in check
export function isInCheck(state: GameState, color: Color): boolean {
  const kingPos = findKing(state.board, color);
  if (!kingPos) return false;

  const enemyColor = color === 'white' ? 'black' : 'white';
  return isSquareAttacked(state, kingPos, enemyColor);
}

// Simplified castling check for Fischer Random
function canCastleKingside(state: GameState, color: Color): boolean {
  const rights = state.castlingRights[color];
  const row = color === 'white' ? 0 : 7;

  // Check rook hasn't moved
  const rookCol = rights.rookCols[1]; // Rightmost rook
  const rook = state.board[row][rookCol];
  if (!rook || rook.type !== 'rook' || rook.hasMoved) return false;

  // Check path is clear between king and target (g1/g8)
  const kingCol = rights.kingCol;
  const targetKingCol = 6;
  const targetRookCol = 5;

  // All squares between king current pos and target, and rook current pos and target must be empty (or be the king/rook)
  const minCol = Math.min(kingCol, targetKingCol, rookCol, targetRookCol);
  const maxCol = Math.max(kingCol, targetKingCol, rookCol, targetRookCol);

  for (let col = minCol; col <= maxCol; col++) {
    if (col === kingCol || col === rookCol) continue;
    if (state.board[row][col]) return false;
  }

  // Check king doesn't pass through check
  const enemyColor = color === 'white' ? 'black' : 'white';
  const step = targetKingCol > kingCol ? 1 : -1;
  for (let col = kingCol; col !== targetKingCol + step; col += step) {
    if (isSquareAttacked(state, { row, col }, enemyColor)) return false;
  }

  return true;
}

function canCastleQueenside(state: GameState, color: Color): boolean {
  const rights = state.castlingRights[color];
  const row = color === 'white' ? 0 : 7;

  // Check rook hasn't moved
  const rookCol = rights.rookCols[0]; // Leftmost rook
  const rook = state.board[row][rookCol];
  if (!rook || rook.type !== 'rook' || rook.hasMoved) return false;

  // Check path is clear
  const kingCol = rights.kingCol;
  const targetKingCol = 2;
  const targetRookCol = 3;

  const minCol = Math.min(kingCol, targetKingCol, rookCol, targetRookCol);
  const maxCol = Math.max(kingCol, targetKingCol, rookCol, targetRookCol);

  for (let col = minCol; col <= maxCol; col++) {
    if (col === kingCol || col === rookCol) continue;
    if (state.board[row][col]) return false;
  }

  // Check king doesn't pass through check
  const enemyColor = color === 'white' ? 'black' : 'white';
  const step = targetKingCol > kingCol ? 1 : -1;
  for (let col = kingCol; col !== targetKingCol + step; col += step) {
    if (isSquareAttacked(state, { row, col }, enemyColor)) return false;
  }

  return true;
}

// Get valid moves (filtering out moves that leave king in check)
export function getValidMoves(state: GameState, pos: Position): Position[] {
  const piece = getPiece(state.board, pos);
  if (!piece || piece.color !== state.turn) return [];

  const rawMoves = getRawMoves(state, pos);

  return rawMoves.filter(move => {
    // Simulate the move
    const newBoard = state.board.map(row => [...row]);
    newBoard[move.row][move.col] = piece;
    newBoard[pos.row][pos.col] = null;

    // Handle en passant capture
    if (piece.type === 'pawn' && state.enPassantSquare &&
        move.row === state.enPassantSquare.row && move.col === state.enPassantSquare.col) {
      const capturedPawnRow = piece.color === 'white' ? move.row - 1 : move.row + 1;
      newBoard[capturedPawnRow][move.col] = null;
    }

    // Check if own king is in check after move
    const testState = { ...state, board: newBoard };
    return !isInCheck(testState, piece.color);
  });
}

// Make a move and return new game state
export function makeMove(state: GameState, from: Position, to: Position): GameState {
  const piece = getPiece(state.board, from);
  if (!piece) return state;

  const newBoard = state.board.map(row => [...row]);
  const capturedPiece = getPiece(state.board, to);

  // Handle castling
  let isCastling = false;
  if (piece.type === 'king' && Math.abs(to.col - from.col) === 2) {
    isCastling = true;
    const row = from.row;
    const rights = state.castlingRights[piece.color];

    if (to.col === 6) {
      // Kingside - move rook from its position to f1/f8
      const rookCol = rights.rookCols[1];
      newBoard[row][5] = { ...newBoard[row][rookCol]!, hasMoved: true };
      if (rookCol !== 5) newBoard[row][rookCol] = null;
    } else if (to.col === 2) {
      // Queenside - move rook from its position to d1/d8
      const rookCol = rights.rookCols[0];
      newBoard[row][3] = { ...newBoard[row][rookCol]!, hasMoved: true };
      if (rookCol !== 3) newBoard[row][rookCol] = null;
    }
  }

  // Handle en passant capture
  let isEnPassant = false;
  if (piece.type === 'pawn' && state.enPassantSquare &&
      to.row === state.enPassantSquare.row && to.col === state.enPassantSquare.col) {
    isEnPassant = true;
    const capturedPawnRow = piece.color === 'white' ? to.row - 1 : to.row + 1;
    newBoard[capturedPawnRow][to.col] = null;
  }

  // Move piece
  newBoard[to.row][to.col] = { ...piece, hasMoved: true };
  newBoard[from.row][from.col] = null;

  // Handle pawn promotion (auto-promote to queen for MVP)
  if (piece.type === 'pawn' && (to.row === 7 || to.row === 0)) {
    newBoard[to.row][to.col] = { type: 'queen', color: piece.color, hasMoved: true };
  }

  // Update en passant square
  let newEnPassantSquare: Position | null = null;
  if (piece.type === 'pawn' && Math.abs(to.row - from.row) === 2) {
    newEnPassantSquare = { row: (from.row + to.row) / 2, col: from.col };
  }

  // Update castling rights
  const newCastlingRights = { ...state.castlingRights };
  if (piece.type === 'king') {
    newCastlingRights[piece.color] = {
      ...newCastlingRights[piece.color],
      kingside: false,
      queenside: false
    };
  }
  if (piece.type === 'rook') {
    const rights = newCastlingRights[piece.color];
    if (from.col === rights.rookCols[0]) {
      newCastlingRights[piece.color] = { ...rights, queenside: false };
    } else if (from.col === rights.rookCols[1]) {
      newCastlingRights[piece.color] = { ...rights, kingside: false };
    }
  }

  const move: Move = {
    from,
    to,
    capture: capturedPiece || undefined,
    castling: isCastling ? (to.col === 6 ? 'kingside' : 'queenside') : undefined,
    enPassant: isEnPassant
  };

  const newTurn: Color = state.turn === 'white' ? 'black' : 'white';

  const newState: GameState = {
    ...state,
    board: newBoard,
    turn: newTurn,
    moveHistory: [...state.moveHistory, move],
    selectedSquare: null,
    validMoves: [],
    castlingRights: newCastlingRights,
    enPassantSquare: newEnPassantSquare
  };

  // Check for check/checkmate
  newState.check = isInCheck(newState, newTurn);

  // Check for game over (no legal moves)
  let hasLegalMoves = false;
  outer: for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const p = newBoard[row][col];
      if (p && p.color === newTurn) {
        const moves = getValidMoves(newState, { row, col });
        if (moves.length > 0) {
          hasLegalMoves = true;
          break outer;
        }
      }
    }
  }

  if (!hasLegalMoves) {
    newState.gameOver = true;
    newState.winner = newState.check ? state.turn : null; // null = stalemate
  }

  return newState;
}

// Get piece symbol for display
export function getPieceSymbol(piece: Piece, pieces: Record<string, PieceDefinition>): string {
  const def = pieces[piece.type];
  return def?.symbol || '?';
}
