from connect4 import Connect4Env


def evaluate_board(env: Connect4Env, player: int) -> float:
    """
    Evaluate the board for the given player.

    Args:
        env (Connect4Env): The Connect4 environment which contains the current state of the game.
        player (int): The player to evaluate for (0 or 1).

    Returns:
        float: The evaluation score for the player (how good is the current state for the player).
    """
    return 0  # Placeholder for the evaluation function


def minimax(env: Connect4Env, origin_player: int, depth: int) -> tuple[float, int]:
    """
    Minimax algorithm for the Connect4 game.

    Args:
        env (Connect4Env): The Connect4 environment which contains the current state of the game.
        origin_player (int): The player for whom we are evaluating the moves (0 or 1).
        depth (int): The depth to search in the minimax algorithm.

    Returns:
        tuple: A tuple containing the evaluation score and the best move for the current player.
    """
    # If we are at the maximum depth or the game is over, return the evaluation score
    if depth == 0 or env.is_over():
        return evaluate_board(env, origin_player), None
    # Otherwise, we need to explore the possible moves and recurcively call minimax
    raise NotImplementedError("Minimax function is not implemented yet.")


def get_best_move(env: Connect4Env, depth: int = 2) -> int:
    """
    Get the best move for the current player using the minimax algorithm.
    
    Args:
        env (Connect4Env): The Connect4 environment.
        depth (int): The depth to search in the minimax algorithm.

    Returns:
        int: The best move for the current player (the column index).
    """
    print("Thinking...")
    score, move = minimax(env, env.current_player, depth)
    print(f"Best move: {move} with score: {score}")
    return move
