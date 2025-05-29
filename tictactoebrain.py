from tictactoe import win, x, y
import random
import pickle

# Q-table to store state-action values
Q = {}

# Hyperparameters
alpha = 0.2         # Learning rate
gamma = 0.95        # Discount factor
epsilon = 0.5       # Exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.9995  # Slower decay

def get_state(x, y):
    state = ''
    for i in range(9):
        if x[i] == 'X':
            state += 'X'
        elif y[i] == 'O':
            state += 'O'
        else:
            state += ' '
    return state

def get_available_positions(x, y):
    available = []
    for i in range(9):
        if x[i] == '' and y[i] == '':
            available.append(i)
    return available

def can_win_or_block(x_board, y_board, player, available_positions):
    wins = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [2, 4, 6], [0, 4, 8]              # Diagonals
    ]
    player_mark = 'X' if player == 'X' else 'O'
    opponent_mark = 'O' if player == 'X' else 'X'
    
    # Check for winning move
    for combo in wins:
        marks = 0
        for pos in combo:
            if player_mark == 'X':
                mark_at_pos = x_board[pos]
            else:
                mark_at_pos = y_board[pos]
            if mark_at_pos == player_mark:
                marks += 1
        
        empty = []
        for pos in combo:
            if x_board[pos] == '' and y_board[pos] == '':
                empty.append(pos)
        
        if marks == 2 and len(empty) == 1:
            return empty[0]  # Winning move
    
    # Check for blocking move
    for combo in wins:
        marks = 0
        for pos in combo:
            if opponent_mark == 'O':
                mark_at_pos = y_board[pos]
            else:
                mark_at_pos = x_board[pos]
            if mark_at_pos == opponent_mark:
                marks += 1
        
        empty = []
        for pos in combo:
            if x_board[pos] == '' and y_board[pos] == '':
                empty.append(pos)
        
        if marks == 2 and len(empty) == 1:
            return empty[0]  # Blocking move
    
    return None

def would_opponent_win(x_board, y_board, opponent_mark):
    wins = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [2, 4, 6], [0, 4, 8]              # Diagonals
    ]
    for combo in wins:
        marks = 0
        for pos in combo:
            if opponent_mark == 'O':
                mark_at_pos = y_board[pos]
            else:
                mark_at_pos = x_board[pos]
            if mark_at_pos == opponent_mark:
                marks += 1
        empty = []
        for pos in combo:
            if x_board[pos] == '' and y_board[pos] == '':
                empty.append(pos)
        if marks == 2 and len(empty) == 1:
            return True
    return False

def choose_action(state, available_positions, epsilon_value, x_board=None, y_board=None, player=None):
    # During training, both 'X' and 'O' use the heuristic 20% of the time
    should_use_heuristic = False
    if x_board is not None:
        if y_board is not None:
            if random.random() < 0.2:  # Reduced to 20%
                should_use_heuristic = True
    
    if should_use_heuristic:
        move = can_win_or_block(x_board, y_board, player, available_positions)
        if move is not None:
            return move

    # Epsilon-greedy policy
    if random.uniform(0, 1) < epsilon_value:
        return random.choice(available_positions)
    else:
        # Get Q-values for each available position
        q_values = []
        for pos in available_positions:
            q_value = Q.get((state, pos), 0.0)
            q_values.append(q_value)

        # Find the maximum Q-value
        max_q = q_values[0]
        for q in q_values:
            if q > max_q:
                max_q = q

        # Find all positions with the maximum Q-value
        best_positions = []
        for i in range(len(available_positions)):
            if q_values[i] == max_q:
                best_positions.append(available_positions[i])

        # Randomly choose one of the best positions
        return random.choice(best_positions)

def step(x, y, action, player, available_positions):
    # Check if the opponent would win on their next turn (before this move)
    opponent_mark = 'O' if player == 'X' else 'X'
    would_win_before = would_opponent_win(x, y, opponent_mark)

    # Execute the move
    if player == 'X':
        x[action] = 'X'
    else:
        y[action] = 'O'
    available_positions.remove(action)
    next_state = get_state(x, y)
    result = win(print_result=False, x_board=x, y_board=y)

    # Check if the opponent would win after this move
    would_win_after = would_opponent_win(x, y, opponent_mark)

    # Assign reward
    if result == player:  # Current player wins
        reward = 0.5
        done = True
    elif result in ['X', 'O']:  # Opponent wins
        reward = -1
        done = True
    elif result == 'tie':  # Tie
        reward = 0
        done = True
    else:  # Game continues
        # Reward for blocking: +0.5 if this move prevented the opponent from winning
        if would_win_before and not would_win_after:
            reward = 1
        else:
            reward = 0
        done = False
    return next_state, reward, done

def train(num_games=200000):  # Increased to 200,000 games
    global epsilon
    o_wins = 0
    x_wins = 0
    ties = 0
    for game in range(num_games):
        x_train = [''] * 9
        y_train = [''] * 9
        available_positions = list(range(9))
        # Bias toward 'O' starting (70% chance)
        if random.random() < 0.7:
            turn = 0  # 'O' starts
        else:
            turn = 1  # 'X' starts
        done = False
        state = get_state(x_train, y_train)

        for _ in range(9):
            if done:
                break
            player = 'X' if turn == 1 else 'O'
            action = choose_action(state, available_positions, epsilon, x_train, y_train, player)
            next_state, reward, done = step(x_train, y_train, action, player, available_positions)
            next_available = get_available_positions(x_train, y_train)
            max_next_q = 0.0
            if next_available:
                next_q_values = []
                for pos in next_available:
                    q_value = Q.get((next_state, pos), 0.0)
                    next_q_values.append(q_value)
                max_next_q = next_q_values[0]
                for q in next_q_values:
                    if q > max_next_q:
                        max_next_q = q
            current_q = Q.get((state, action), 0.0)
            Q[(state, action)] = current_q + alpha * (reward + gamma * max_next_q - current_q)
            state = next_state
            turn = 1 - turn

        result = win(print_result=False, x_board=x_train, y_board=y_train)
        if result == 'O':
            o_wins += 1
        elif result == 'X':
            x_wins += 1
        elif result == 'tie':
            ties += 1

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if (game + 1) % 10000 == 0:
            print(f"Games: {game + 1}, O Wins: {o_wins}, X Wins: {x_wins}, Ties: {ties}, Q-table size: {len(Q)}")

def save_q_table(filename='q_table.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(Q, f)

def load_q_table(filename='q_table.pkl'):
    global Q
    try:
        with open(filename, 'rb') as f:
            Q = pickle.load(f)
    except FileNotFoundError:
        print("Q-table file not found. Please train the model first.")

def get_best_move(x, y):
    state = get_state(x, y)
    available_positions = get_available_positions(x, y)
    if not available_positions:
        return None
    q_values = []
    for pos in available_positions:
        q_value = Q.get((state, pos), 0.0)
        q_values.append(q_value)
    print(f"State: {state}, Available: {available_positions}, Q-values: {q_values}")
    return choose_action(state, available_positions, epsilon_value=0)

if __name__ == "__main__":
    print("Training the CPU...")
    train(num_games=200000)
    save_q_table()
    print("Training complete. Q-table saved.")
