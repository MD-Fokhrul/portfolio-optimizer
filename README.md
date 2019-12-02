# Yield-Variance Portfolio Optimization with Deep Reinforcement Learning

## Overview
Final project for Professor Iddo Drori's COMSW4775 - Deep Learning @ Columbia University.
- Reinforcement Learning agent for portfolio optimization using DDPG and min-variance optimization.
- LSTM-CNN future stock price change and market factor prediction
- Regression optimizer to predict t+1 price changes from t price change and market factor data 


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

## DDPG

## Overview
Before the competition was announced this was the focal point of the project. It is able to make portfolio optimization decisions for t+1 based on price information from t.
However, since the objective has changed with Kaggle to having to predict 152 days into the future, there's need to predict future prices which is not something an RL agent is designed to do.
We did adapt the agent to work optimization for the 152 predicted price data and try to optimize the portfolio. However, since in that case we already "know" the future prices (i.e. predicted prices) it becomes a closed optimization problem which would not require an agent. 

### How to run
Run `python main.py [...optional run args]`

### Run args
- dataset_name: name of dataset csv file in data dir
- data_dir
- lr: learning rate
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
- results_root_dir: directory to save results in
- checkpoints_root_dir: directory to save checkpoints in
- checkpoints_interval: interval at which to save checkpoints
- save_checkpoints: save checkpoints [True/False]
- load_model: load model path
- modes: combination of [test/train]
- plot_stocks: plot graph of stock prices [True/False]
- visualize_portfolio: visualize portfolio holdings per epoch [True/False]


## LSTM-CNN

### How to run
Run `python prices_main.py [...optional run args]`

### Run args
- dataset_name: name of dataset csv file in data dir
- data_dir
- lr: learning rate
- epochs: number of training epochs
- limit_days: limit data to days per episode (latest days in dataset)
- val_days: number of days to use as validation set
- target_size: if target size is different from input size
- days_lookback_window: days to use for sliding LSTM lookback window
- test_predict_days: number of days to predict/generate after final input day
- num_sample_stocks: limit data to subset of stocks
- batch_size: learning batch size
- log_interval: reporting interval
- log_comet: use comet ml for datalogging [True/False]
- comet_tags: tags for comet ml
- force_cpu: force pytorch to use cpu even if cuda available [True/False]
- results_root_dir: directory to save results in
- checkpoints_root_dir: directory to save checkpoints in
- checkpoints_interval: interval at which to save checkpoints
- save_checkpoints: save checkpoints [True/False]
- load_model: load model path
- modes: combination of [test/train]
- log_interval: interval to log at batch level
- log_batches: log at batch level [True/False]
- log_epochs: log at epoch level [True/False]

## Regression Estimator

### How to run
Run `python estimator_main.py [...optional run args]`

### Run args
- dataset_name: name of dataset csv file in data dir
- data_dir
- lr: learning rate
- epochs: number of training epochs
- limit_days: limit data to days per episode (latest days in dataset)
- target_size: if target size is different from input size
- days_lookback_window: days to use for sliding LSTM lookback window
- test_split_days: number of days to use for test set
- num_sample_stocks: limit data to subset of stocks
- batch_size: learning batch size
- log_interval: reporting interval
- force_cpu: force pytorch to use cpu even if cuda available [True/False]
- results_root_dir: directory to save results in
- checkpoints_root_dir: directory to save checkpoints in
- checkpoints_interval: interval at which to save checkpoints
- save_checkpoints: save checkpoints [True/False]
- load_model: load model path
- modes: combination of [test/train]
- log_interval: interval to log at batch level
- log_batches: log at batch level [True/False]
- log_epochs: log at epoch level [True/False]

## Project structure

### DDPG
- main.py
Main DDPG training script

- model
DDPG agent related classes and utilities. DDPG, Actor, Critic, etc

- env
DDPG portfolio/environment related classes and utilities. Portfolio, gym environment, etc

### LSTM
- prices_main.py
Main LSTM training script

- future prices
LSTM future price prediction related classes and utilities. LSTM, cNN, dataloader, train and test

### Regressor
- estimator_main.py
Main regressor training script

- estimator
Estimator related classes and utilites. Xls dataloader and model

### Common

- dataset
Dataset loader and utilities

- data
Data directory

## References
- Deep Reinforcement Learning Approach for Stock Trading - Zhuoran Xiong et al. | [paper on arxiv.org](https://arxiv.org/pdf/1811.07522.pdf?fbclid=IwAR24YGM8Jv-855TIIJlKC268cfZqQoIMJQPAiljv_RfpdUVyITcuHaVv30k)
- Create custom gym environments from scratch - Adam King [Blog post on towardsdatascience.com](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)
- A blundering guide to making a deep actor-critic bot for stock trading [Blog post on towardsdatascience.com](https://towardsdatascience.com/a-blundering-guide-to-making-a-deep-actor-critic-bot-for-stock-trading-c3591f7e29c2)

### Code references
- https://github.com/notadamking/Stock-Trading-Environment
- https://github.com/hackthemarket/gym-trading
- https://github.com/edolev89/pytorch-ddpg
- https://github.com/hust512/DQN-DDPG_Stock_Trading
- D360 driving challenge bootstrap code 