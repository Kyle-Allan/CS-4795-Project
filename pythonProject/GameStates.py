def is_winner(board, player):
    """Check if the given player has won on the board."""
    win_positions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    return any(all(board[pos] == player for pos in win) for win in win_positions)

def generate_states(board, turn, valid_states):
    """Recursive function to generate all valid game states."""
    if tuple(board) in valid_states:  # Avoid duplicates
        return

    # Add current board to valid states
    valid_states.add(tuple(board))

    # Check if the game is over (win or draw)
    if is_winner(board, 1) or is_winner(board, -1) or board.count(0) == 0:
        return

    # Generate all possible next states
    for i in range(9):
        if board[i] == 0:  # Empty spot
            board[i] = turn  # Place the current player's move
            next_turn = -1 if turn == 1 else 1  # Alternate turn
            generate_states(board, next_turn, valid_states)
            board[i] = 0  # Undo move (backtracking)

def generate_all_valid_states():
    """Generate all unique valid Tic Tac Toe game states."""
    initial_board = [0] * 9  # Start with an empty board
    valid_states = set()
    generate_states(initial_board, 1, valid_states)  # 1 (X) always starts
    return valid_states

def print_board(board):
    """Helper function to print the board neatly using 1, 0, and -1."""
    for i in range(0, 9, 3):
        row = ' | '.join(f"{board[j]:2}" for j in range(i, i + 3))  # Format each cell
        print(row)
        if i < 6:  # Add horizontal separator after each row except the last
            print("---+----+---")
    print()


# Generate all valid states
all_states = generate_all_valid_states()
print(f"Total unique valid states: {len(all_states)}\n")

# Display some states (optional)
for state in list(all_states):  # Display the first 10 states
    print_board(state)
