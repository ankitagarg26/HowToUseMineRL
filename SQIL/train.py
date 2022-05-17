# from pyvirtualdisplay import Display
import numpy as np
import torch
import gym
import minerl
import random
import torch.nn.functional as F
import math
import random

from torch import nn
from sklearn.cluster import KMeans
from tqdm import tqdm
from minerl.data import BufferedBatchIter
from collections import deque
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from src import replay_memory
from src import network

# display = Display(visible=0, size=(400, 300))
# display.start();


def sample_from_buffer(online_memory_replay, expert_memory_replay, online_memory_batch_size, expert_memory_batch_size):
    expert_batch = expert_memory_replay.sample(expert_memory_batch_size)
    expert_batch_state = []
    expert_batch_action = []
    expert_batch_next_state = []
    expert_batch_reward = []
    expert_batch_done = []

    for x in expert_batch:
        #         for x in t:
        expert_batch_state.append(x[0])
        expert_batch_action.append(x[1])
        expert_batch_next_state.append(x[3])
        expert_batch_reward.append(x[2])
        expert_batch_done.append(x[4])

    expert_batch_state = np.array([expert_batch_state[i]["pov"] for i in range(len(expert_batch_state))])
    expert_batch_state = torch.from_numpy(expert_batch_state.transpose(0, 3, 1, 2).astype(np.float32) / 255)

    expert_batch_next_state = np.array([expert_batch_next_state[i]["pov"] for i in range(len(expert_batch_next_state))])
    expert_batch_next_state = torch.from_numpy(expert_batch_next_state.transpose(0, 3, 1, 2).astype(np.float32) / 255)

    expert_batch_action = torch.Tensor([expert_batch_action[i] for i in range(len(expert_batch_action))])
    expert_batch_reward = torch.Tensor([expert_batch_reward[i] for i in range(len(expert_batch_reward))]).unsqueeze(1)
    expert_batch_done = torch.Tensor([expert_batch_done[i] for i in range(len(expert_batch_done))]).unsqueeze(1)

    if online_memory_replay.size() == 0:
        return expert_batch_state, expert_batch_action, expert_batch_reward, expert_batch_next_state, expert_batch_done

    online_batch = online_memory_replay.sample(online_memory_batch_size)
    online_batch_state = []
    online_batch_action = []
    online_batch_next_state = []
    online_batch_reward = []
    online_batch_done = []
    for x in online_batch:
        #         for x in t:
        online_batch_state.append(x[0])
        online_batch_action.append(x[1])
        online_batch_next_state.append(x[3])
        online_batch_reward.append(x[2])
        online_batch_done.append(x[4])

    online_batch_state = np.array([online_batch_state[i]["pov"] for i in range(len(online_batch_state))])
    online_batch_state = torch.from_numpy(online_batch_state.transpose(0, 3, 1, 2).astype(np.float32) / 255)

    online_batch_next_state = np.array([online_batch_next_state[i]["pov"] for i in range(len(online_batch_next_state))])
    online_batch_next_state = torch.from_numpy(online_batch_next_state.transpose(0, 3, 1, 2).astype(np.float32) / 255)

    online_batch_action = torch.Tensor([online_batch_action[i] for i in range(len(online_batch_action))]).unsqueeze(1)
    online_batch_reward = torch.Tensor([online_batch_reward[i] for i in range(len(online_batch_reward))]).unsqueeze(1)
    online_batch_done = torch.Tensor([online_batch_done[i] for i in range(len(online_batch_done))]).unsqueeze(1)

    batch_state = torch.cat([online_batch_state, expert_batch_state], dim=0)
    batch_next_state = torch.cat([online_batch_next_state, expert_batch_next_state], dim=0)
    batch_action = torch.cat([online_batch_action, expert_batch_action], dim=0)
    batch_reward = torch.cat([online_batch_reward, expert_batch_reward], dim=0)
    batch_done = torch.cat([online_batch_done, expert_batch_done], dim=0)

    return batch_state, batch_action, batch_reward, batch_next_state, batch_done


DATA_DIR = "/home/ankitagarg/minerl/data/"
ENVIRONMENT = 'MineRLTreechopVectorObf-v0'
NUM_ACTION_CENTROIDS = 64
REPLAY_MEMORY = 80000

NUM_OF_EPOCHS = 1000 #50
NUM_STEPS = 300 #1000
GAMMA = 0.99
REPLAY_START_SIZE = 30000 #1000

decay = 0.999
update_steps = 10000
batch_size = 32
min_epsilon = 0.1
learning_rate = 0.0001

action_centroids = np.load('./action_centroids.npy')

def train(env, writer, onlineQNetwork, targetQNetwork, expert_memory_replay, online_memory_replay):
    learn_steps = 0
    begin_learn = False
    epsilon = 0.9
    print('training')

    for epoch in range(NUM_OF_EPOCHS):
        state = env.reset()
        episode_reward = 0
        loss_values = 0
        for time_steps in range(NUM_STEPS):
            network_state = torch.from_numpy(state['pov'].transpose(2, 0, 1)[None].astype(np.float32) / 255)
            selected_action = onlineQNetwork.choose_action(network_state, epsilon)
            action = action_centroids[selected_action]
            minerl_action = {"vector": action}

            next_state, reward, done, _ = env.step(minerl_action)
            episode_reward += reward

            online_memory_replay.add((state, selected_action, 0, next_state, done))

            if online_memory_replay.size() > REPLAY_START_SIZE:
                if begin_learn is False:
                    print('learn begin!')
                    begin_learn = True
                learn_steps += 1
                if learn_steps % update_steps == 0:
                    print('updating target network')
                    targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

                batch_state, batch_action, batch_reward, batch_next_state, batch_done = sample_from_buffer(
                    online_memory_replay, expert_memory_replay, batch_size, batch_size)

                with torch.no_grad():
                    next_q = targetQNetwork(batch_next_state)
                    next_v = targetQNetwork.getV(next_q)
                    y = batch_reward + (1 - batch_done) * GAMMA * next_v

                loss = F.mse_loss(onlineQNetwork(batch_state).gather(1, batch_action.long()), y)
                optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=learning_rate)

                loss_values += loss.item()
                optimizer.zero_grad()
                loss.backward()
                for param in onlineQNetwork.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
                
                writer.add_scalar('loss', loss.item(), global_step=learn_steps)

            if done:
                break

            state = next_state

        print('epoch:', epoch, ' episode reward:', episode_reward, flush=True)
        print('epoch:', epoch, ' loss:', loss_values / NUM_STEPS, flush=True)
        writer.add_scalar('episode reward', episode_reward, global_step=epoch)
        if epoch % 1000 == 0:
            epsilon = max(min_epsilon, epsilon * decay)
            # learning_rate = learning_rate/1.5
        if epoch % 10 == 0:
            torch.save(onlineQNetwork.state_dict(), 'sqil-policy.para')
            online_memory_replay.save('online_memory_replay')

    online_memory_replay.save('online_memory_replay')
    torch.save(onlineQNetwork.state_dict(), 'sqil-policy.para')

if __name__ == "__main__":
    print('Inside main', flush=True)
    writer = SummaryWriter('logs/sqil')
    onlineQNetwork = network.SoftQNetwork((3, 64, 64), NUM_ACTION_CENTROIDS, alpha=4)
#     onlineQNetwork.load_state_dict(torch.load('sqil-policy.para'))
    targetQNetwork = network.SoftQNetwork((3, 64, 64), NUM_ACTION_CENTROIDS, alpha=4)
    targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

    print('Initializing Environment',flush=True)
    env = gym.make(ENVIRONMENT)

    expert_memory_replay = replay_memory.Memory(REPLAY_MEMORY)
    expert_memory_replay.load('expert_memory_replay')
    online_memory_replay = replay_memory.Memory(REPLAY_MEMORY)
#     online_memory_replay.load('online_memory_replay')

    train(env, writer, onlineQNetwork, targetQNetwork, expert_memory_replay, online_memory_replay)


