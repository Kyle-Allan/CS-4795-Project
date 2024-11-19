import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

def display_board(board):
    """Display the Tic-Tac-Toe board in a human-readable format."""
    symbols = {1: 'X', -1: 'O', 0: ' '}
    print(f"""
    {symbols[board[0]]} | {symbols[board[1]]} | {symbols[board[2]]}
    ---------
    {symbols[board[3]]} | {symbols[board[4]]} | {symbols[board[5]]}
    ---------
    {symbols[board[6]]} | {symbols[board[7]]} | {symbols[board[8]]}
    """)


def check_winner(board):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] != 0:
            return board[condition[0]]  # Return 1 for X, -1 for O
    return 0  # No winner
def generate_valid_game_states():
    boards = []
    labels = []

    # Function to check for winners
    def check_winner(board):
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        for condition in win_conditions:
            if board[condition[0]] == board[condition[1]] == board[condition[2]] != 0:
                return board[condition[0]]  # 1 for X, -1 for O
        return 0  # No winner

    # Recursive function to simulate games
    def simulate_game(board, turn):
        winner = check_winner(board)
        if winner == 1:
            boards.append(board[:])
            labels.append("Win")
            print("Winning Board:")
            display_board(board)
            return
        elif winner == -1:
            boards.append(board[:])
            labels.append("Loss")
            print("Losing Board:")
            display_board(board)
            return
        elif 0 not in board:  # Draw
            boards.append(board[:])
            labels.append("Draw")
            print("Draw Board:")
            display_board(board)
            return

        # Simulate placing pieces in empty spots
        for i in range(9):
            if board[i] == 0:
                board[i] = turn  # Place current player's piece
                simulate_game(board, -turn)  # Switch turn
                board[i] = 0  # Undo move

    # Start with an empty board
    empty_board = [0] * 9
    simulate_game(empty_board, 1)  # Start with X (1)
    return np.array(boards), np.array(labels)

# Generate data
boards, labels = generate_valid_game_states()
print("Generated Dataset Size:", len(boards))


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(boards, labels, test_size=0.2, random_state=42)

# Train a logistic regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# Test the classifier with a new board state
new_board = [1, 1, 1, 0, -1, 1, 0, 0, 0]  # Example board
print('example')
display_board(new_board)
predicted_label = clf.predict([new_board])[0]
print("Predicted Game Outcome for the board:", predicted_label)


for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        print("Misclassified Board:")
        display_board(X_test[i])
        print(f"True Label: {y_test[i]}, Predicted Label: {y_pred[i]}")
