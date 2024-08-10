import itertools
import gym_super_mario_bros
from agent import Mario

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=None)
mario.load_state_dict(path_state_dict='trained_model/mario_net_580.chkpt')
mario.net.eval()
mario.exploration_rate = -1


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