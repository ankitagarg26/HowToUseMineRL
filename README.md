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

## Run sample code

Run the sample code to test the installation.
```
xvfb-run python3 firstAgent.py
```

## Official Doc and Reference

MineRL official documentation: https://minerl.readthedocs.io/en/latest/






