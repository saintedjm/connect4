import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from rl_task import DQN, preprocess_obs, DEVICE
from setup import setup

def load_agent(model_path, obs_shape, n_actions):
    agent = DQN(obs_shape, n_actions).to(DEVICE)
    agent.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.eval()
    return agent

def agent_select_action(agent, state, legal_actions):
    state_v = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        q_values = agent(state_v).cpu().numpy()[0]
    mask = np.full_like(q_values, -np.inf)
    mask[legal_actions] = 0
    q_values = q_values + mask
    action = int(np.argmax(q_values))
    return action

def play_against_agent(model_path="dqn_policy_net.pth"):
    env = gym.make('Connect4-v0', render_mode='rgb_array')
    obs_shape = env.observation_space[0].shape
    n_actions = env.action_space.n
    agent = load_agent(model_path, obs_shape, n_actions)

    (obs, _), info = env.reset()
    done = False
    player = 0  # agent is player 0, human is player 1

    print("Agent is Player 0 (Red). You are Player 1 (Black). Enter column number (0-6) to play.")

    while not done:
        rgb = env.render()
        if rgb is not None:
            plt.imshow(rgb)
            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.1)
            plt.clf()
        legal_actions = info['legal_actions']
        if player == 0:
            # Agent's turn
            state = preprocess_obs(obs)
            action = agent_select_action(agent, state, legal_actions)
            print(f"Agent plays: {action}")
        else:
            # Human's turn
            print(f"Legal moves: {legal_actions}")
            while True:
                try:
                    action = int(input("Your move (column): "))
                    if action in legal_actions:
                        break
                    print("Invalid move. Try again.")
                except Exception:
                    print("Please enter a valid integer.")

        (next_obs, _), reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        obs = next_obs
        info = next_info
        player = 1 - player

    rgb = env.render()
    if rgb is not None:
        plt.imshow(rgb)
        plt.axis('off')
        plt.show()
    if abs(reward[0]) >= 1e-6:    # reward is not [0, 0]
        print("Agent wins!" if player == 1 else "You win!")
    else:
        print("Draw!")
    env.close()

if __name__ == "__main__":
    setup()
    play_against_agent()
