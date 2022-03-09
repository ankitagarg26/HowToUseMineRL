import gym
import minerl

from pyvirtualdisplay import Display
from colabgymrender.recorder import Recorder
from time import sleep

display = Display(visible=0, size=(1400, 900))
display.start()

class Agent():
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        
    def act(self, obs):
        action = self.action_space.noop()
        action['camera'] = [0, 0.03*obs["compass"]["angle"]]
        action['back'] = 0
        action['forward'] = 1
        action['jump'] = 1
        action['attack'] = 1
        
        return action
    
if __name__ == '__main__':
    env = gym.make('MineRLNavigateDense-v0')
    env = Recorder(env, './video', fps=60)
    obs  = env.reset()

    agent = Agent(env.action_space, env.observation_space)

    done = False
    net_reward = 0

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(
            action)
        net_reward += reward

    env.release()
    env.close()

    print("Total reward: ", net_reward)