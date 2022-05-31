# pong-dqn-rl
Deep Q-Learning example for Pong. 

This is an implementation of the Duel Double DQN algorithm in PyTorch for solving the OpenAI GYM Atari Pong environment. The agent learns to play to an impressive standard in just 900 episodes.

# Requirements

* Python 3.10
* Python packages; 
    * `torch` for neural networks
    * `gym` for RL environments
    * `opencv-python` for image processing
    * `configparser` for configuration
    * `black`, `isort` & `mypy` for formatting

# Method
Run the commands in the repository `Makefile` for training, formatting and testing the code.

# Resources
(Playing Atari with Deep Reinforcement Learning)[https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf]
(Dueling Network Architectures for Deep Reinforcement Learning)[https://arxiv.org/abs/1511.06581]