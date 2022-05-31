"""This script provides the methods for training the agent in the Pong environment"""
import collections
import configparser
import json
import statistics
import time
from typing import Deque

import gym
import os
import numpy as np
import torch
from rl_agent import Agent

config = configparser.ConfigParser()
config.read("config.ini")

TRAIN_MODEL = True
LOAD_FROM_FILE = False
SAVE_MODELS = True
RENDER_GAME = None  # 'human' to watch
MODEL_PATH = f"models/{config['SAVING']['model_tag']}"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

environment: gym.Env = gym.make(config["GAME"]["env"], render_mode=RENDER_GAME)
agent: Agent = Agent(environment)

if LOAD_FROM_FILE:
    agent.online_model.load_state_dict(torch.load(f"{MODEL_PATH}/model.pkl"))

    with open(f"{MODEL_PATH}/parameters.json") as outfile:
        parameters = json.load(outfile)
        agent.epsilon = parameters.get("epsilon")

last_100_ep_reward: Deque[float] = collections.deque(maxlen=100)
total_step = 1  # Cumulative sum of all steps in episodes

for episode in range(0, int(config["TRAINING"]["max_episode"])):

    startTime = time.time()  # Keep time
    state = environment.reset()  # Reset env

    state = agent.preProcess(state)  # Process image

    # Stack state . Every state contains 4 time contionusly frames
    # We stack frames like 4 channel image
    state = np.stack((state, state, state, state))

    total_max_q_val = 0  # Total max q vals
    total_reward = 0  # Total reward for each episode
    total_loss = 0  # Total loss for each episode
    for step in range(int(config["TRAINING"]["max_step"])):

        # Select and perform an action
        action = agent.act(state)  # Act
        next_state, reward, done, info = environment.step(action)  # Observe

        next_state = agent.preProcess(next_state)  # Process image

        # Stack state . Every state contains 4 time contionusly frames
        # We stack frames like 4 channel image
        next_state = np.stack((next_state, state[0], state[1], state[2]))

        # Store the transition in memory
        agent.storeResults(state, action, reward, next_state, done)  # Store to mem

        # Move to the next state
        state = next_state  # Update state

        if TRAIN_MODEL:
            # Perform one step of the optimization (on the target network)
            if len(agent.memory) < int(config["TRAINING"]["min_memory"]):
                loss, max_q = [0, 0]
            else:
                # Train with random state from memory
                loss, max_q = agent.train()
        else:
            loss, max_q = [0, 0]

        total_loss += loss
        total_max_q_val += max_q
        total_reward += reward
        total_step += 1
        if total_step % 1000 == 0:
            agent.adaptiveEpsilon()  # Decrease epsilon

        if done:  # Episode completed
            currentTime = time.time()  # Keep current time
            time_passed = currentTime - startTime  # Find episode duration
            parameters = {"epsilon": agent.epsilon}

            if SAVE_MODELS and episode % int(config["SAVING"]["interval"]) == 0:
                torch.save(agent.online_model.state_dict(), f"{MODEL_PATH}/model.pkl")

                with open(f"{MODEL_PATH}/parameters.json", "w") as outfile:
                    json.dump(parameters, outfile)

            if TRAIN_MODEL:
                # update target model
                agent.target_model.load_state_dict(agent.online_model.state_dict())

            last_100_ep_reward.append(total_reward)
            avg_max_q_val = total_max_q_val / step

            print(
                f"Episode:{episode} Reward:{total_reward} Last_100_Avg_Rew:{statistics.mean(last_100_ep_reward)} Loss:{total_loss} Avg_Max_Q:{avg_max_q_val} Epsilon:{agent.epsilon,} Duration:{time_passed} Step/Total:{step}/{total_step}"
            )
            break
