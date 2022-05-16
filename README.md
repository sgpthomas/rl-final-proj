# Reinforcement Learning

Code for replicating AI Economist results + some tweaks to explore the general setup.

## Files

 - `run.py` starts different trials using parameters specified in a yaml file. Calling `./run.py --config <config.yml> --output <experiment name>` will create a directory called `<experiment name>` and start two phase training using Raylib and tensorflow. The structure of this was heavily inspired by [training_script.py](https://github.com/salesforce/ai-economist/blob/master/tutorials/rllib/training_script.py).
 - `tf_models.py` defines a LSTM neural network that the agents use as their mdoel when learning. This is copied verbatim from the ai-economist source.
 - `env_wrapper.py` converts between the internal environment model that ai-economist uses and the environment model that Raylib uses. This is taken directly from ai-economist. We did not write this.
 - `learn_to_build.py` is an `ai-economist` component that adds the behavior of skill increasing every time a house was built. This was heavily inspired by the related `build.py` component in the ai-economist.
 - `learn_to_gather.py` is a component that adds the increasing skill for gathering operations. It also makes gathering an explicit actions that agents have to take instead of it happening automatically when moving to scare with resources.
 - `initial_coin.py` is a scenario that allows you to specify the starting coin for each agent.
 - `nonlinear_disutility.py` is a scenario that slightly modifies the stone and wood environment to use a nonlinear disutility function.
