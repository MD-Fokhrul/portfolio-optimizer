# Yield-Variance Portfolio Optimization with Deep Reinforcement Learning

## Overview
Final project for Professor Iddo Drori's COMSW4775 - Deep Learning @ Columbia University.
Reinforcement Learning agent for portfolio optimization using DDPG and min-variance optimization.  


## Authors
- Eden Dolev
- Ben Segev
- Lingyu An

## Dependencies
### Required
- python 3.7
- pytorch 1.3.0
- gym 0.15.3
- numpy 1.17.2
- pandas 0.25.3

### Optional
- comet_ml 2.0.16

## How to run

Run `python main.py [...optional run args]`

#### Run args
- dataset_name: [dj/sp500]
- data_dir
- lr: learning rate
- init_cash: starting cash balance for portfolio environment
- episodes: number of training episodes
- limit_days: limit data to days per episode (latest days in dataset)
- limit_iters: hard limit on total iterations (for debugging)
- num_sample_stocks: limit data to subset of stocks
- discount_factor: q learning discount factor
- minibatch_size: policy learning batch size
- warmup_iters: number of iterations for random action generation in the start of training (should be at least minbatch_size for effective learning)
- random_process_theta: theta hyperparameter for Ornstein Uhlenback Process for action noise
- log_interval: reporting interval
- log_comet: use comet ml for datalogging [True/False]
- comet_tags: tags for comet ml
- force_cpu: force pytorch to use cpu even if cuda available [True/False]

## Project structure
- main.py
Main training script. Instantiates agent and environment and runs training loop.

- model
Agent related classes and utilities. DDPG, Actor, Critic, etc.

- env
Portfolio/environment related classes and utilities. Portfolio, gym environment, etc.

- dataset
Dataset loader and utilities.

- data
Data directory with dow jones and S&P500 stock prices over period of time.

## References
- Deep Reinforcement Learning Approach for Stock Trading - Zhuoran Xiong et al. | [paper on arxiv.org](https://arxiv.org/pdf/1811.07522.pdf?fbclid=IwAR24YGM8Jv-855TIIJlKC268cfZqQoIMJQPAiljv_RfpdUVyITcuHaVv30k)
- Create custom gym environments from scratch - Adam King [Blog post on towardsdatascience.com](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)
- A blundering guide to making a deep actor-critic bot for stock trading [Blog post on towardsdatascience.com](https://towardsdatascience.com/a-blundering-guide-to-making-a-deep-actor-critic-bot-for-stock-trading-c3591f7e29c2)

### Code references
- https://github.com/notadamking/Stock-Trading-Environment
- https://github.com/hackthemarket/gym-trading
- https://github.com/edolev89/pytorch-ddpg
- https://github.com/hust512/DQN-DDPG_Stock_Trading