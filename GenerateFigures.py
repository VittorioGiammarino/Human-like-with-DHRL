#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:43:19 2021

@author:
"""

from utils import Plot
import numpy as np

# Humans
with open('Data_and_models/Humans/Real_Reward_eval_human.npy', 'rb') as f:
    Reward_human_eval = np.load(f, allow_pickle=True).tolist()
with open('Data_and_models/Humans/Real_Traj_eval_human.npy', 'rb') as f:
    Trajectories_human_eval = np.load(f, allow_pickle=True).tolist()
with open('Data_and_models/Data_preprocessed/coins_location_eval.npy', 'rb') as f:
    coins_location_eval = np.load(f, allow_pickle=True)

Rand_traj = np.argmax(Reward_human_eval)    
Plot.Human_plot_and_stats(Trajectories_human_eval, Reward_human_eval, coins_location_eval, Rand_traj)

# Deep Q learning
with open('Data_and_models/Deep_Q_Learning/Training_DQN.npy', 'rb') as f:
    Training_DQN = np.load(f, allow_pickle=True).tolist()
    
with open('Data_and_models/RL_algorithms/DeepQ_Learning/Q_learning_evaluation.npy', 'rb') as f:
    DQN_Evaluation = np.load(f, allow_pickle=True).tolist()

Rand_traj=Rand_traj%10
coins_location = coins_location_eval[Rand_traj,:,:]
Plot.DQN_plot_and_stats(Training_DQN[0][0], DQN_Evaluation, coins_location)

# Behavioral Cloning
with open('Data_and_models/Results_main/BC_from_human_evaluation_results.npy', 'rb') as f:
    BC_evaluation = np.load(f, allow_pickle=True).tolist()
    
Plot.BC_plot_and_stats(BC_evaluation, coins_location)

# HIL random Initialized
with open('Data_and_models/Results_main/HIL_from_human_evaluation_results_random_init.npy', 'rb') as f:
    HIL_evaluation_random_init = np.load(f, allow_pickle=True).tolist()
    
Plot.HIL_random_init_plot_and_stats(HIL_evaluation_random_init,coins_location)    

# HIL pre_init
with open('Data_and_models/Results_main/HIL_from_human_evaluation_results.npy', 'rb') as f:
    HIL_evaluation_pre_init = np.load(f, allow_pickle=True).tolist()
    
Plot.HIL_pre_init_plot_and_stats(HIL_evaluation_pre_init,coins_location)

# DOC
with open('Data_and_models/DOC/Training_DOC.npy', 'rb') as f:
    Training_DOC = np.load(f, allow_pickle=True).tolist()
    
with open('Data_and_models/RL_algorithms/Option_critic_with_DQN/DeepSoftOC_learning_evaluation.npy', 'rb') as f:
    DOC_evaluation = np.load(f, allow_pickle=True).tolist()
    
Plot.DOC_plot_and_stats(Training_DOC, DOC_evaluation, coins_location)

Plot.DOC_DQN_comparison(Training_DOC, Training_DQN[0][0])
    