import gym
import minerl

from pyvirtualdisplay import Display
from colabgymrender.recorder import Recorder
from time import sleep

display = Display(visible=0, size=(1400, 900))
display.start()

#Creates an agent that returns random action from the action space defined by the environment.
class RandomAgent():
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        
    def act(self, obs):
        return self.action_space.sample()
    
if __name__ == '__main__':
    env = gym.make("MineRLTreechop-v0")
    env = Recorder(env, './video', fps=30) #used for rendering the environment as a video. File is saved in the provided directory once environment is released. 
    obs = env.reset()

    agent = RandomAgent(env.action_space, env.observation_space)

    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)

    env.release()
    env.close()