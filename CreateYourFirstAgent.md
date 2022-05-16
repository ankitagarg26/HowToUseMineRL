### Creating Your First Agent ###

1. Import the necessary packages.
    ```
    import gym
    import minerl
    ```
2. To create the environment, call the function **gym.make** and choose any environments included in the minerl package. 
    ```
    env = gym.make('MineRLNavigateDense-v0')
    ```
3. Reset the environment.
    ```
    obs = env.reset()
    ``` 
    *obs* is a data type dictionary that contains information about the current state of the environment.

    In the case of the MineRLNavigate-v0 environment, three observations are returned: pov (an RGB image of the agent’s first person perspective); compassAngle (a float giving the angle of the agent to its (approximate) target); and inventory (a dictionary containing the amount of 'dirt' blocks in the agent’s inventory).

4. Use OpenAI Gym **env.step** method to take actions through the environment.
    ```
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
    ```
    The code above generates a random agent that traverses the environment until time runs out or the agent dies.
    
 You can find the template code [here](https://github.com/ankitagarg26/HowToUseMineRL/blob/main/random_agent.py). 
 
 ### Other Samples ###
 
[no_op_actions.py](https://github.com/ankitagarg26/HowToUseMineRL/blob/main/no_op_actions.py): 
shows the code for creating an agent that take the noop action with a few modifications targeted towards the goal.

[kmeans.py](https://github.com/ankitagarg26/HowToUseMineRL/blob/main/kmeans.py): 
shows how kmeans can be used to quantize the human demonstrations and give agent n discrete actions representative of actions taken by humans when solving the environment.
