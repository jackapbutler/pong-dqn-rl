"""This script provides the methods for training the agent in the Pong environment"""
import json
import torch
import gym
import numpy as np
import collections
import time
import json
import rl_agent
import configparser

TRAIN_MODEL = True
LOAD_FROM_FILE = False
SAVE_MODELS = True
RENDER_GAME = True

config = configparser.ConfigParser()
config.read("config.ini")

environment: gym.Env = (
    gym.make(config["GAME"]["env"], render_mode="human")
    if RENDER_GAME
    else gym.make(config["GAME"]["env"], render_mode="human")
)

agent: rl_agent.Agent = rl_agent.Agent(environment)

if LOAD_FROM_FILE:
    agent.online_model.load_state_dict(
        torch.load(config["SAVING"]["path"] + config["SAVING"]["file_episode"] + ".pkl")
    )

    with open(
        config["SAVING"]["path"] + config["SAVING"]["file_episode"] + ".json"
    ) as outfile:
        param = json.load(outfile)
        agent.epsilon = param.get("epsilon")

    startEpisode = int(config["SAVING"]["file_episode"]) + 1

else:
    startEpisode = 1

last_100_ep_reward = collections.deque(maxlen=100)  # Last 100 episode rewards
total_step = 1  # Cumulkative sum of all steps in episodes
for episode in range(startEpisode, int(config["TRAINING"]["max_episode"])):

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
            agent.adaptiveEpsilon()  # Decrase epsilon

        if done:  # Episode completed
            currentTime = time.time()  # Keep current time
            time_passed = currentTime - startTime  # Find episode duration
            current_time_format = time.strftime(
                "%H:%M:%S", time.gmtime()
            )  # Get current dateTime as HH:MM:SS
            epsilonDict = {
                "epsilon": agent.epsilon
            }  # Create epsilon dict to save model as file

            if (
                SAVE_MODELS and episode % int(config["SAVING"]["interval"]) == 0
            ):  # Save model as file
                weightsPath = f"{config['SAVING']['path']}{episode}.pkl"
                epsilonPath = f"{config['SAVING']['path']}{episode}.json"

                torch.save(agent.online_model.state_dict(), weightsPath)
                with open(epsilonPath, "w") as outfile:
                    json.dump(epsilonDict, outfile)

            if TRAIN_MODEL:
                agent.target_model.load_state_dict(
                    agent.online_model.state_dict()
                )  # Update target model

            last_100_ep_reward.append(total_reward)
            avg_max_q_val = total_max_q_val / step

            outStr = "Episode:{} Time:{} Reward:{:.2f} Loss:{:.2f} Last_100_Avg_Rew:{:.3f} Avg_Max_Q:{:.3f} Epsilon:{:.2f} Duration:{:.2f} Step:{} CStep:{}".format(
                episode,
                current_time_format,
                total_reward,
                total_loss,
                np.mean(last_100_ep_reward),
                avg_max_q_val,
                agent.epsilon,
                time_passed,
                step,
                total_step,
            )
            print(outStr)

            if SAVE_MODELS:
                outputPath = (
                    config["SAVING"]["PATH"] + "out" + ".txt"
                )  # Save outStr to file
                with open(outputPath, "a") as outfile:
                    outfile.write(outStr + "\n")

            break
