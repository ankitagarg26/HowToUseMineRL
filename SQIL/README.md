## SQIL ##

`ExpertDemonstration.ipynb`: Randomly samples human demonstration data and saves it into file numpy file expert_memory_replay.npy

`SQIL.ipynb`: Learns policy using SQIL algorithm mentioned [here](https://arxiv.org/abs/1905.11108). 

Code is referred from https://github.com/s-shiroshita/minerl2020_sqil_submission

### Experimentation Result ###

Model is trained for 80k steps on environment MineRLTreechopVectorObf-v0.

The below graph shows the increase in q-value with training.

![alt text](https://github.com/ankitagarg26/HowToUseMineRL/blob/main/SQIL/q-value.jpg "Q-value")

After every 1000 iterations, the model is evaluated. The following graph shows change in average reward obtained while evaluation during training.  

![alt text](https://github.com/ankitagarg26/HowToUseMineRL/blob/main/SQIL/returns.jpg "Average Returns")

