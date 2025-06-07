import matplotlib.pyplot as plt

from connect4 import Connect4Env
from minimax import get_best_move


def play_against_minimax(env: Connect4Env, depth: int = 2):
    """
    Allows a human to play against the minimax agent.
    Human is player 1, minimax is player 0.
    """
    assert env.render_mode == 'rgb_array', "Render mode must be 'rgb_array' for visualization."
    env.reset()
    done = False
    print("You are player 1. Enter column number (0-indexed) to play.")
    while not done:
        rgb = env.render()
        if rgb is not None:
            plt.imshow(rgb)
            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.1)
            plt.clf()
        # print(env.board)  # Optionally keep this for debugging
        if env.current_player == 0:
            move = get_best_move(env, depth)
        else:
            legal_moves = env.get_moves()
            move = None
            while move not in legal_moves:
                try:
                    move = int(input(f"Your move (choose from {legal_moves}): "))
                    if move not in legal_moves:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter a valid integer.")
        env.move(move)
        done = env.is_over()
        if done:
            rgb = env.render()
            if rgb is not None:
                plt.imshow(rgb)
                plt.axis('off')
                plt.show()
            # print(env.board)
            if env.winner == 1 and env.current_player == 1:
                print("You win!")
            elif env.winner == 1 and env.current_player == 0:
                print("Minimax agent wins!")
            else:
                print("It's a draw!")
            break

if __name__ == "__main__":
    env = Connect4Env(render_mode='rgb_array')
    play_against_minimax(env, depth=3)
