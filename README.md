# Treasure Hunt Game – Deep Q-Learning Project

## Overview

This project implements a reinforcement learning agent, a pirate, that learns to find the optimal path to a treasure within a maze. The agent uses deep Q-learning, a form of deep reinforcement learning, to navigate an 8x8 grid while minimizing steps and avoiding getting stuck. The goal is for the pirate to consistently reach the treasure faster than a human player.

This project was developed as part of a course module focused on AI training, exploration vs. exploitation, and reinforcement learning implementation using Keras and NumPy in a Jupyter Notebook environment.

## Reflection

### What work did I do on this project?

In this project, I was provided with a base environment that included the `TreasureMaze` class and some scaffolding for training a reinforcement learning agent. I designed and implemented the DQN model architecture, created the training loop logic from scratch, and added enhancements like reward shaping, loop detection, and epsilon decay to improve learning. I also handled memory replay, model evaluation, and training performance diagnostics. Most of the code involving the neural network logic, training cycle, exploration strategies, and performance evaluation was self-authored.

### What do computer scientists do and why does it matter?

Computer scientists build and optimize the systems that make modern technology run. We analyze problems, model them computationally, and write efficient algorithms that scale. This matters because the solutions we write, whether in AI, systems, or data, power everything from healthcare to real-time navigation. In this project, it was about applying reinforcement learning to teach an agent how to navigate autonomously, which is conceptually similar to real-world applications like robotics, autonomous driving, and dynamic resource allocation.

### How do I approach a problem as a computer scientist?

I start by understanding the constraints, identifying the moving parts, and thinking in terms of systems and data flow. I decompose the problem into modular, testable units and figure out the best tools or algorithms to solve each part. For the RL agent, I focused on balancing exploration vs. exploitation, tuning hyperparameters, and designing a reward function that aligns with the behavior I wanted the agent to learn. I iterate, test frequently, and optimize based on empirical feedback.

### What are my ethical responsibilities to the end user and the organization?

From a technical standpoint, my responsibility is to ensure the systems I help build are robust, transparent, and aligned with both user expectations and organizational objectives. That includes protecting data, avoiding biased training data or unintended behaviors, and being accountable for model decisions. Especially in AI, where outcomes can be opaque, I believe in building in ways to inspect, audit, and constrain models so that edge cases or misuse don’t lead to harm or exploitation.

## Project Structure

- TreasureMaze.py – Represents the environment and maze layout.
- GameExperience.py – Records episodes (sequences of state transitions).
- Angel_RiveraMoreira_ProjectTwo.ipynb – Main notebook where deep Q-learning is implemented and trained.



## Key Features

- Custom maze environment (8x8 matrix with valid/invalid cells).
- Q-learning agent using a Keras-based model.
- Reward shaping based on distance to the treasure and loop detection.
- Replay memory and experience sampling for more stable training.
- Epsilon-greedy strategy for balancing exploration and exploitation.
- Training loop with early stopping based on average win rate.
- Evaluation of the trained model using `play_game()`.



## How the Agent Learns

The pirate learns by taking actions in the maze and receiving rewards based on:
- Reaching the treasure (+100)
- Getting stuck or losing (-20)
- Improving proximity to the treasure (+2 per cell closer)
- Repeating steps or visiting the same cells (-0.5 to -1)

The agent uses Deep Q-Learning with experience replay to learn from these rewards and improve its decision-making policy over time.



## How to Run

1. Load all required `.py` files and the `.ipynb` notebook.
2. Run all cells in the notebook.
3. The agent trains over several epochs and evaluates performance every 100 epochs.
4. Training stops early if the agent consistently wins 90%+ of test games.
5. The final model is saved as `treasure_model_final.h5`.



## Known Limitations

- Training may plateau if the agent falls into movement loops.
- Requires tuning of hyperparameters like `epsilon`, learning rate, and discount factor.
- Highly dependent on maze layout,new mazes may require retraining.



## Ethical Consideration

This game simulation is used to demonstrate learning mechanisms and has no real-world data implications. However, the principles of training agents, avoiding bias, and ensuring fairness are broadly applicable in real-world AI use cases.



## Author

Angel M. Rivera Moreira  
Milestone Two
CS370 – Artificial Intelligence