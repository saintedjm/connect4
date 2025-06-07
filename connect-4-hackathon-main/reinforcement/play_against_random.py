import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from setup import setup

def play_against_random(human_as_player=1):
    env = gym.make('Connect4-v0', render_mode='rgb_array')
    (obs, _), info = env.reset()
    done = False
    player = 0  # 0 = Player 0, 1 = Player 1

    print(f"You are Player {human_as_player}. Random is Player {1 - human_as_player}.")
    print("Enter a column number (0–6) when it's your turn.\n")

    while not done:
        # Render the board
        rgb = env.render()
        if rgb is not None:
            plt.imshow(rgb)
            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.1)
            plt.clf()

        legal_actions = info['legal_actions']

        if player == human_as_player:
            print(f"Your turn. Legal moves: {legal_actions}")
            while True:
                try:
                    action = int(input("Your move (column): "))
                    if action in legal_actions:
                        break
                    print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter a valid integer.")
        else:
            action = np.random.choice(legal_actions)
            print(f"Random Player {player} plays: {action}")

        (next_obs, _), reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        obs = next_obs
        info = next_info
        player = 1 - player

    # Final render
    rgb = env.render()
    if rgb is not None:
        plt.imshow(rgb)
        plt.axis('off')
        plt.show()

    # print(reward)
    # print(info['legal_actions'])
    
    # Game result
    if abs(reward[0]) >= 1e-6:    # reward is not [0, 0]
        if player != human_as_player:
            print("You win!")
        else:
            print("Random player wins!")
    else:
        print("it is a Draw!")

    env.close()

if __name__ == "__main__":
    setup()
    play_against_random(1)

