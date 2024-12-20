# MAgent2 RL Final Project
## Overview
In this final project, you will develop and train a reinforcement learning (RL) agent using the MAgent2 platform. The task is to solve a specified MAgent2 environment `battle`, and your trained agent will be evaluated on all following three types of opponents:

1. Random Agents: Agents that take random actions in the environment.
2. A Pretrained Agent: A pretrained agent provided in the repository.
3. A Final Agent: A stronger pretrained agent

### Agent Training with Functional Policy and Hyperparameter Reward

#### Functional Policy
The agent's decision-making process is based on a **Functional Policy**, which uses mathematical functions or flexible models to map observations to actions. This approach allows the agent to:

- **Adapt to Complex Environments**: Quickly adjust to dynamic and unpredictable scenarios.
- **Optimize Decision-Making**: Enhance performance by leveraging sophisticated policy structures.

#### Hyperparameter Reward Optimization
The reward system for the agent is tuned through hyperparameter optimization to guide learning effectively. Key aspects include:

- **step_reward**: reward after every step.
- **attack_opponent_reward**: reward added for attacking an opponent.
- **dead_penalty**: reward given to an agent when it gets eliminated (dies).


<p align="center">
  <img src="assets/my_random.gif" width="300" alt="random agent"  title="Against Random Agent"/>
  <img src="assets/my_pretrained.gif" width="300" alt="Against Pretrained Agent" />
  <img src="assets/my_final.gif" width="300" alt="Against Final Agent" />
</p>



## Installation
clone this repo and install with
```
pip install -r requirements.txt
```

## Demos
See `final_training.py` for a training.

## Evaluation
Refer to `eval.py` for the evaluation code, you might want to modify it with your specific codebase.

## References

1. [MAgent2 GitHub Repository](https://github.com/Farama-Foundation/MAgent2)
2. [MAgent2 API Documentation](https://magent2.farama.org/introduction/basic_usage/)

For further details on environment setup and agent interactions, please refer to the MAgent2 documentation.
