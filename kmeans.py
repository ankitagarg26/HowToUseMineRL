import gym
import minerl
import tqdm
import numpy as np

from pyvirtualdisplay import Display
from colabgymrender.recorder import Recorder
from time import sleep
from sklearn.cluster import KMeans

display = Display(visible=0, size=(1400, 900))
display.start()

dat = minerl.data.make('MineRLTreechopVectorObf-v0', data_dir='/home/ankitagarg/minerl/data/')

act_vectors = []
NUM_CLUSTERS = 30

# Load the dataset storing 1000 batches of actions
for _, act, _, _, _ in tqdm.tqdm(dat.batch_iter(16, 32, 2, preload_buffer_size=20)):
    act_vectors.append(act['vector'])
    if len(act_vectors) > 1000:
        break

# Reshape these the action batches
acts = np.concatenate(act_vectors).reshape(-1, 64)
kmeans_acts = acts[:100000]

# Use sklearn to cluster the demonstrated actions
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(kmeans_acts)

# Now we have 32 actions that represent reasonable action for agent to take. Next, we will use these actions to explore the environment.
i, net_reward, done, env = 0, 0, False, gym.make('MineRLTreechopVectorObf-v0')
env = Recorder(env, './video', fps=60)
obs = env.reset()

while not done:
    # Let's use a frame skip of 4
    if i % 4 == 0:
        action = {
            'vector': kmeans.cluster_centers_[np.random.choice(NUM_CLUSTERS)]
        }
        
    obs, reward, done, info = env.step(action)
    env.render()
    
    if reward > 0:
        print("+{} reward!".format(reward))
    net_reward += reward
    i += 1

env.release()

print("Total reward: ", net_reward)