# Deep Q-Learning AI with PyTorch

This project was built during a hackathon as an exploration into reinforcement learning and neural networks. I implemented a Deep Q-Learning (DQN) agent using PyTorch to better understand how AI can learn to make decisions over time through trial and error.

---

## Motivation

As a second-semester electrical engineering student, I’ve been exploring neural networks and reinforcement learning out of personal interest. I wanted to apply concepts I’ve learned from videos (like 3Blue1Brown's series on neural networks) and research papers, including:

- **Backpropagation**
- **Cost functions**
- **Q-learning fundamentals**
- **Exploration vs exploitation trade-offs**

Even though I didn’t implement everything from scratch, this project helped me see how these theories translate into real models. PyTorch handled a lot of the low-level operations (like autograd and tensor updates), but working with it gave me a clearer picture of how those internals work in practice.

---

## What the AI Does

The agent learns from its environment using rewards and penalties. It tries to maximize future rewards by:

- Approximating the Q-value function with a neural network
- Using experience replay for better sample efficiency
- Gradually shifting from exploration to exploitation (epsilon-greedy strategy)

---

## Tech Stack

- Python 3
- PyTorch
- NumPy

---

## Files

- `train.py` – main training loop  
- `dqn_agent.py` – agent class and logic  
- `model.py` – PyTorch neural network  
- `environment.py` – simplified or Gym-compatible environment  

---

## How to Run

Make sure you have Python and PyTorch installed, then:

```bash
python train.py
