# Scaling Imitation Learning in Minecraft

A network is trained directly from expert demonstrations to predict an action given an observation. It does not require any interaction with the environment during training and is unhindered by the sparsity of rewards.

Continuous control by quantization is implemented to reduce the action space. 
1. Only up to 3 simultaneous sub-actions are allowed.
2. The camera movement is discretized into four actions of moving the camera by 22.5 degree in each direction 
3. Redundant actions like turning left and right at the same time are removed. 

Technical Report can be found [here](https://arxiv.org/abs/2007.02701)

Code referred from [here](https://github.com/amiranas/minerl_imitation_learning)

### Experiment Results ###
After training the model for 3M steps, the model was evaluated on environment MineRLObtainIronPickaxe-v0 for 100 episodes.

![alt text](https://github.com/ankitagarg26/HowToUseMineRL/blob/main/minerl_imitation_learning/reward.jpg "Episode Rewards")
