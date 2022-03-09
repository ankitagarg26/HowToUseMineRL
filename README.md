# HowToUseMineRL

MineRL is a research project started at Carnegie Mellon University aimed at developing various aspects of AI within minecraft.

To accesss MineRL dataset and Gym environments, you will need to install the main python package minerl.

## Installation on Linux

### Install JDK 1.8 on your system

  ```
    sudo add-apt-repository ppa:openjdk-r/ppa
    sudo apt-get update
    sudo apt-get install openjdk-8-jdk
  ```

### Install the minerl package
```
pip3 install --upgrade minerl
```
Note: You may need the user flag: pip3 install --upgrade minerl --user to install properly.

### Install xvfb

In order to run minerl environments without a head i.e. without any external display monitor you need to installa software renderer.

```
sudo apt install xvfb
```

### Install Environment ###
You can use the **environment.yml** file to install the conda environment on your system, which will install all of the Python libraries required for the sample codes in this repository.

Run the below command to copy the environment. 
```
conda env create -f environment.yml
```
Before running above command make sure conda is installed on the system. If not, you can follow the instructions [here](https://phoenixnap.com/kb/how-to-install-anaconda-ubuntu-18-04-or-20-04) to install conda.

## Run sample code

Run the sample code to test the installation.
```
xvfb-run python3 firstAgent.py
```

## Start with MineRL ##
**random_agent.py**: shows the code for creating an agent that takes random actions.

**no_op_actions.py**: shows the code for creating an agent that take the noop action with a few modifications targeted towards the goal.

**kmeans.py**: shows how kmeans can be used to quantize the human demonstrations and give agent n discrete actions representative of actions taken by humans when solving the environment.

## Official Doc and Reference

MineRL official documentation: 

https://minerl.readthedocs.io/en/latest/

https://github.com/minerllabs/minerl

https://minerl.io/workshop/






