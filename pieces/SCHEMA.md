# Ultimate Random Chess - Piece Definition Schema

## Coordinate System

- Origin: the piece's current position
- `[x, y]` where:
  - `+x` = right (toward h-file for white)
  - `-x` = left (toward a-file for white)
  - `+y` = forward (toward opponent)
  - `-y` = backward (toward own side)

## Piece Definition

```yaml
piece_id:
  name: "Human Readable Name"
  symbol: "X"              # 1-2 character symbol for notation
  value: 5                 # Relative piece value (pawn=1, queen=9)
  royal: false             # If true, losing this piece loses the game
  description: "..."       # Optional description
  moves: [...]             # List of move patterns
  special_rules: {...}     # Optional special rules
```

## Move Types

### 1. Slide

Moves along a direction, can be blocked by pieces.

```yaml
- type: slide
  directions:              # List of [x, y] unit vectors
    - [1, 0]              # Example: right
    - [0, 1]              # Example: forward
  range: unlimited         # "unlimited" or integer (max squares)
  move_only: false         # If true, cannot capture with this move
  capture_only: false      # If true, must capture to use this move
  initial_only: false      # If true, only available from starting position
```

### 2. Leap

Jumps directly to target square, ignoring pieces in between.

```yaml
- type: leap
  vectors:                 # List of [x, y] destination offsets
    - [2, 1]              # Example: knight move
    - [1, 2]              # Example: knight move
  move_only: false
  capture_only: false
  initial_only: false
```

### 3. Hopper

Slides along a direction but must jump over exactly N pieces.

```yaml
- type: hopper
  directions:              # Same as slide
    - [1, 0]
  hurdle_count: 1          # Number of pieces to hop over
  land_distance: 1         # Squares after hurdle to land (1 = immediately after)
  move_only: false
  capture_only: false
```

## Special Rules

```yaml
special_rules:
  en_passant: true         # Can capture en passant
  promotion:
    rank: 8                # Rank where promotion occurs
    options: [queen, rook, bishop, knight]
  castling:                # For kings
    kingside: true
    queenside: true
```

## Examples

### Standard Knight
```yaml
knight:
  name: "Knight"
  symbol: "N"
  value: 3
  moves:
    - type: leap
      vectors:
        - [1, 2]
        - [1, -2]
        - [-1, 2]
        - [-1, -2]
        - [2, 1]
        - [2, -1]
        - [-2, 1]
        - [-2, -1]
```

### Pawn (complex asymmetric movement)
```yaml
pawn:
  name: "Pawn"
  symbol: "P"
  value: 1
  moves:
    - type: slide
      directions: [[0, 1]]
      range: 1
      move_only: true
    - type: slide
      directions: [[0, 1]]
      range: 2
      move_only: true
      initial_only: true
    - type: slide
      directions: [[1, 1], [-1, 1]]
      range: 1
      capture_only: true
  special_rules:
    en_passant: true
    promotion:
      rank: 8
      options: [queen, rook, bishop, knight]
```

### Grasshopper (hopper example)
```yaml
grasshopper:
  name: "Grasshopper"
  symbol: "G"
  value: 2
  moves:
    - type: hopper
      directions: [[1,0], [-1,0], [0,1], [0,-1], [1,1], [1,-1], [-1,1], [-1,-1]]
      hurdle_count: 1
      land_distance: 1
```
