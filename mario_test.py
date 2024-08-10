import itertools
import gym_super_mario_bros
from agent_rev2 import Mario

from visual_preprocessing import *

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)

# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=None)
mario.load_state_dict(path_state_dict='trained_model/xxx_testing_mario_net.chkpt')
mario.net.eval()
mario.exploration_rate = -1

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)

env = FrameStack(env, num_stack=4)

for e in itertools.count():
    state = env.reset()[0]

    # Play the game!
    while True:
        env.render()
        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        state, reward, done, trunc, info = env.step(action)

        # Check if end of game
        if done or info["flag_get"]:
            break

        apple = 1

    apple = 1