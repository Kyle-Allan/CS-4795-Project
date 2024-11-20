class Node:
    """Class to represent a single node in the minimax tree."""
    def __init__(self, board, turn):
        self.board = board  # Current board state
        self.turn = turn  # Current player's turn (1 for X, -1 for O)
        self.children = []  # List of child nodes
        self.label = None  # Label for this node

    def __repr__(self):
        return f"Node(Board: {self.board}, Turn: {self.turn}, Label: {self.label})"

def is_winner(board, player):
    """Check if the given player has won on the board."""
    win_positions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    return any(all(board[pos] == player for pos in win) for win in win_positions)

def label_node(node):
    """Recursively label each node in the minimax tree."""
    # Terminal node: label based on the game outcome
    if not node.children:
        if is_winner(node.board, 1):
            node.label = "Win"
        elif is_winner(node.board, -1):
            node.label = "Loss"
        elif node.board.count(0) == 0:
            node.label = "Draw"
        return node.label

    # Recursively label child nodes
    child_labels = [label_node(child) for child in node.children]

    # Assign label to this node based on child labels
    if "Win" in child_labels or "Winning" in child_labels:
        node.label = "Winning"
    elif all(label in {"Losing", "Loss"} for label in child_labels):
        node.label = "Losing"
    elif "Draw" in child_labels or "Drawing" in child_labels:
        if "Win" not in child_labels and "Winning" not in child_labels:
            node.label = "Drawing"
    else:
        node.label = "Neutral"  # Fallback if no other label applies

    return node.label

def build_minimax_tree(board, turn):
    """Recursively build the minimax tree for Tic-Tac-Toe."""
    # Create the current node
    node = Node(board[:], turn)

    # If the game is over (win, loss, or draw), stop recursion
    if is_winner(board, 1) or is_winner(board, -1) or board.count(0) == 0:
        return node

    # Generate all possible next states
    for i in range(9):
        if board[i] == 0:  # Empty spot
            board[i] = turn  # Place the current player's move
            child = build_minimax_tree(board, -turn)  # Alternate turn
            node.children.append(child)
            board[i] = 0  # Undo move (backtracking)

    return node

def print_tree(node, depth=0):
    """Recursively print the minimax tree."""
    indent = "  " * depth
    print(f"{indent}Node(Board: {node.board}, Turn: {node.turn}, Label: {node.label})")
    for child in node.children:
        print_tree(child, depth + 1)

# Initialize the board
initial_board = [0] * 9  # Start with an empty board
turn = 1  # X (1) starts the game

# Build the minimax tree
minimax_tree = build_minimax_tree(initial_board, turn)

# Label the tree
label_node(minimax_tree)

# Print the labeled minimax tree
print_tree(minimax_tree)
