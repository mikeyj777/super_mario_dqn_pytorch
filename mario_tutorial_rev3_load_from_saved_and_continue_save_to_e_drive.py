import itertools
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

from visual_preprocessing import *
from dqn_logging import MetricLogger
from agent_rev2 import Mario


def get_env():

    world = random.randint(1, 8)
    stage = random.randint(1, 4)
    
    # debug xxx apple
    # focus on training on 1-1.  training across all worlds not working too well
    world = 1
    stage = 1

    mario_call = f'SuperMarioBros-{world}-{stage}-v0'

    print(f'mario is now training on world {world}, stage {stage}')

    # Initialize Super Mario environment (in v0.26 change render mode to 'human' to see results on the screen)
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make(mario_call, new_step_api=True)
    else:
        env = gym_super_mario_bros.make(mario_call, render_mode='rgb', apply_api_compatibility=True)

    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    env.reset()
    next_state, reward, done, trunc, info = env.step(action=0)
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

    # Apply Wrappers to environment
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)

    return env

env = get_env()

save_dir = Path("e:/coding/super_mario_dqn_387/checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
mario.load_state_dict(path_state_dict='e:/coding/super_mario_dqn_387/trained_model/mario_net.chkpt')

logger = MetricLogger(save_dir)

episodes = 40
for e in itertools.count():

    state = env.reset()[0]

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)
        
        if done and not info['flag_get']:
            reward = -300
            if info['viewport_y'] > 1:
                reward = -1000
                print('off-screen accident')

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            apple = 1
            break
    
    # every 1000th episode, have mario train on a new level:
    if e % 1000 == 0 and e > 0:
        env = get_env()

    logger.log_episode()

    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
        print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

apple = 1