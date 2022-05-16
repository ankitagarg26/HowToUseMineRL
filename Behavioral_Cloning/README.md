Using human demonstration data, a convolutional neural network is trained to take the current state as input and give the action as output. 

In order to discretize actions, Kmeans is used to cluster actions into 24 groups. 

For experimentation, MineRLTreechopVectorObf-v0 environment data is used to train the model. The following graphs show the loss(left) and return(right) over 140000 iterations.

![Training Loss](Behavioral_Cloning/training_loss.png) ![Training Returns](Behavioral_Cloning/training_return.png)
