# Connect4 Reinforcement Learning Agent

This is a hackathon project where I trained a reinforcement learning agent to play Connect4 using Deep Q-Learning. I used PyTorch for the neural network and Gymnasium to build the environment.

The goal was to apply core concepts like Q-learning, cost functions, and backpropagation, which I learned from resources like 3Blue1Brown and research papers. While PyTorch handled most low-level details, working with it helped me understand the logic behind those operations.

## Files

- `connect4_rl_task.ipynb`: main training notebook
- `rl_task.py`: training logic
- `play_against_random.py`: test against a random agent
- `play_against_rl.py`: test RL vs RL
- `dqn_policy_net.pth`: trained model
- `setup.py`: helper code
- `gym_env/`: environment folder

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
