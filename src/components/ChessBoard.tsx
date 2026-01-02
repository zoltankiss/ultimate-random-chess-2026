'use client';

import { GameState, Position, Piece, PieceDefinition } from '@/lib/types';
import { getValidMoves, makeMove, getPieceSymbol, initializeGame } from '@/lib/game';
import { useState, useCallback } from 'react';

// Unicode chess pieces for display
const PIECE_UNICODE: Record<string, { white: string; black: string }> = {
  king: { white: '♔', black: '♚' },
  queen: { white: '♕', black: '♛' },
  rook: { white: '♖', black: '♜' },
  bishop: { white: '♗', black: '♝' },
  knight: { white: '♘', black: '♞' },
  pawn: { white: '♙', black: '♟' },
};

// For custom pieces, we'll show the symbol in a colored circle
function PieceDisplay({ piece, pieces }: { piece: Piece; pieces: Record<string, PieceDefinition> }) {
  const unicode = PIECE_UNICODE[piece.type];

  if (unicode) {
    return (
      <span className="text-4xl select-none" style={{ color: piece.color === 'white' ? '#fff' : '#000' }}>
        {piece.color === 'white' ? unicode.white : unicode.black}
      </span>
    );
  }

  // Custom piece - show symbol
  const symbol = getPieceSymbol(piece, pieces);
  const def = pieces[piece.type];

  return (
    <div
      className={`w-9 h-9 rounded-full flex items-center justify-center text-sm font-bold border-2
        ${piece.color === 'white' ? 'bg-white text-gray-800 border-gray-400' : 'bg-gray-800 text-white border-gray-600'}`}
      title={def?.name || piece.type}
    >
      {symbol}
    </div>
  );
}

interface SquareProps {
  row: number;
  col: number;
  piece: Piece | null;
  isSelected: boolean;
  isValidMove: boolean;
  isLastMove: boolean;
  isCheck: boolean;
  onClick: () => void;
  pieces: Record<string, PieceDefinition>;
}

function Square({ row, col, piece, isSelected, isValidMove, isLastMove, isCheck, onClick, pieces }: SquareProps) {
  const isLight = (row + col) % 2 === 1;

  let bgColor = isLight ? 'bg-amber-100' : 'bg-amber-700';
  if (isSelected) bgColor = 'bg-blue-400';
  else if (isLastMove) bgColor = isLight ? 'bg-yellow-200' : 'bg-yellow-600';
  else if (isCheck && piece?.type === 'king') bgColor = 'bg-red-500';

  return (
    <div
      className={`w-12 h-12 sm:w-14 sm:h-14 md:w-16 md:h-16 flex items-center justify-center cursor-pointer relative ${bgColor}`}
      onClick={onClick}
    >
      {piece && <PieceDisplay piece={piece} pieces={pieces} />}
      {isValidMove && (
        <div className={`absolute w-3 h-3 rounded-full ${piece ? 'ring-4 ring-green-500 ring-opacity-50 w-full h-full' : 'bg-green-500 opacity-50'}`} />
      )}
    </div>
  );
}

export default function ChessBoard() {
  const [gameState, setGameState] = useState<GameState>(() => initializeGame());

  const handleSquareClick = useCallback((row: number, col: number) => {
    if (gameState.gameOver) return;

    const clickedPos: Position = { row, col };
    const clickedPiece = gameState.board[row][col];

    // If we have a selected piece and clicked on a valid move
    if (gameState.selectedSquare) {
      const isValid = gameState.validMoves.some(m => m.row === row && m.col === col);
      if (isValid) {
        // Make the move
        const newState = makeMove(gameState, gameState.selectedSquare, clickedPos);
        setGameState(newState);
        return;
      }
    }

    // If clicked on own piece, select it
    if (clickedPiece && clickedPiece.color === gameState.turn) {
      const validMoves = getValidMoves(gameState, clickedPos);
      setGameState({
        ...gameState,
        selectedSquare: clickedPos,
        validMoves
      });
    } else {
      // Deselect
      setGameState({
        ...gameState,
        selectedSquare: null,
        validMoves: []
      });
    }
  }, [gameState]);

  const handleNewGame = () => {
    setGameState(initializeGame());
  };

  const lastMove = gameState.moveHistory[gameState.moveHistory.length - 1];

  // Get piece roster for display
  const getPieceRoster = () => {
    const roster: Record<string, number> = {};
    for (let col = 0; col < 8; col++) {
      const piece = gameState.board[0][col];
      if (piece) {
        const name = gameState.pieces[piece.type]?.name || piece.type;
        roster[name] = (roster[name] || 0) + 1;
      }
    }
    return Object.entries(roster).map(([name, count]) =>
      count > 1 ? `${name} x${count}` : name
    ).join(', ');
  };

  return (
    <div className="flex flex-col items-center gap-4 p-4">
      <h1 className="text-2xl font-bold text-gray-800">Ultimate Random Chess</h1>

      <div className="text-sm text-gray-600 max-w-md text-center">
        Pieces: {getPieceRoster()}
      </div>

      <div className="flex items-center gap-4">
        <div className={`px-3 py-1 rounded ${gameState.turn === 'white' ? 'bg-white text-black border-2 border-black' : 'bg-gray-300 text-gray-500'}`}>
          White {gameState.turn === 'white' && '(to move)'}
        </div>
        <div className={`px-3 py-1 rounded ${gameState.turn === 'black' ? 'bg-gray-800 text-white' : 'bg-gray-300 text-gray-500'}`}>
          Black {gameState.turn === 'black' && '(to move)'}
        </div>
      </div>

      {gameState.check && !gameState.gameOver && (
        <div className="text-red-600 font-bold">Check!</div>
      )}

      {gameState.gameOver && (
        <div className="text-xl font-bold text-center">
          {gameState.winner ? (
            <span className={gameState.winner === 'white' ? 'text-gray-800' : 'text-gray-600'}>
              {gameState.winner.charAt(0).toUpperCase() + gameState.winner.slice(1)} wins by checkmate!
            </span>
          ) : (
            <span className="text-gray-600">Stalemate - Draw!</span>
          )}
        </div>
      )}

      {/* Board - render from white's perspective (row 7 at top for black, row 0 at bottom for white) */}
      <div className="border-4 border-amber-900 shadow-lg">
        {[7, 6, 5, 4, 3, 2, 1, 0].map(row => (
          <div key={row} className="flex">
            {[0, 1, 2, 3, 4, 5, 6, 7].map(col => {
              const piece = gameState.board[row][col];
              const isSelected = gameState.selectedSquare?.row === row && gameState.selectedSquare?.col === col;
              const isValidMove = gameState.validMoves.some(m => m.row === row && m.col === col);
              const isLastMove = lastMove && ((lastMove.from.row === row && lastMove.from.col === col) ||
                                               (lastMove.to.row === row && lastMove.to.col === col));
              const isCheck = gameState.check && piece?.type === 'king' && piece?.color === gameState.turn;

              return (
                <Square
                  key={`${row}-${col}`}
                  row={row}
                  col={col}
                  piece={piece}
                  isSelected={isSelected}
                  isValidMove={isValidMove}
                  isLastMove={isLastMove}
                  isCheck={isCheck}
                  onClick={() => handleSquareClick(row, col)}
                  pieces={gameState.pieces}
                />
              );
            })}
          </div>
        ))}
      </div>

      <button
        onClick={handleNewGame}
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
      >
        New Game
      </button>

      {/* Piece legend */}
      <div className="mt-4 p-4 bg-gray-100 rounded-lg max-w-lg">
        <h3 className="font-bold mb-2">Custom Pieces in this Game:</h3>
        <div className="grid grid-cols-2 gap-2 text-sm">
          {Object.entries(gameState.pieces)
            .filter(([key]) => !['king', 'queen', 'rook', 'bishop', 'knight', 'pawn'].includes(key))
            .filter(([key]) => gameState.board[0].some(p => p?.type === key))
            .map(([key, def]) => (
              <div key={key} className="flex items-center gap-2">
                <span className="font-mono bg-gray-200 px-1 rounded">{def.symbol}</span>
                <span className="font-semibold">{def.name}</span>
                {def.description && (
                  <span className="text-gray-500 text-xs">- {def.description}</span>
                )}
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}
