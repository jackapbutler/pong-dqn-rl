# pong-dqn-rl
Deep Dueling Neural Network Reinforcement Learning for Pong. 

This is an implementation of the Duel Double DQN algorithm in PyTorch for solving the OpenAI GYM Atari Pong environment which learns to play at an impressive level within approximately 1,000 episodes.

# Requirements

* Python 3.10
* Python tools;
    * `Poetry` for environment management
    * `Pyenv` for managing Python versions
* Python packages; 
    * `torch` for neural networks
    * `gym` for RL environments
    * `opencv-python` for image processing
    * `configparser` for configuration
    * `black`, `isort` & `mypy` for formatting

# Method
Run the commands in the repository `Makefile` for training, formatting and testing the code.

# Experiments
All experiments are configured in the `config.ini` file, ran in `pong_dqn_rl/training.py` with models, parameters and logs being saved to the `models` directory.

# Resources
[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

[Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
