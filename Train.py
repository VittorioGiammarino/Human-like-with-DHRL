#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 20:21:57 2021

@author:
"""

from utils import Deep_Qlearning
from utils import Imitation_Learning
from utils.Training import NeuralTD
from utils.Training import Option_Critic_with_DQN

import numpy as np

# Humans
with open('Data_and_models/Data_preprocessed/Reward_human_eval.npy', 'rb') as f:
    Reward_human_eval = np.load(f, allow_pickle=True).tolist()
with open('Data_and_models/Data_preprocessed/Trajectories_human_eval.npy', 'rb') as f:
    Trajectories_human_eval = np.load(f, allow_pickle=True).tolist()
with open('Data_and_models/Data_preprocessed/coins_location_eval.npy', 'rb') as f:
    coins_location_eval = np.load(f, allow_pickle=True)
 
# DQN training
seed = 36
coins_location = coins_location_eval[np.argmax(Reward_human_eval),:,:]
s0 = Trajectories_human_eval[np.argmax(Reward_human_eval)][0,:] #initial state
NEpisodes = 200
reset = 'standard'
    
DQN_agent = Deep_Qlearning.Q_learning_NN(seed, coins_location)
reward_per_episode, traj, network_weights = DQN_agent.Training_buffer(NEpisodes, seed, reset, s0)

# Behavioral Cloning
with open('Data_and_models/Data_preprocessed/Rotation_human_eval.npy', 'rb') as f:
    Rotation_human_eval = np.load(f, allow_pickle=True).tolist()

index = np.argmax(Reward_human_eval)
size_data = len(Trajectories_human_eval[index])-1
T_set = Trajectories_human_eval[index][0:size_data,:]
Heading_set = Rotation_human_eval[index][0:size_data]
BC_agent = Imitation_Learning.BC(T_set, Heading_set)
BC_agent.encode_data()
BC_agent.init_Model()
BC_agent.train(1)

# HIL random Initialized
N=10
option_space = 2
initialization = 'random'
HIL_random_init = Imitation_Learning.HIL(T_set, Heading_set, N, initialization, option_space)
HIL_random_init.encode_data()
pi_hi_batch_random_init, pi_lo_batch_random_init, pi_b_batch_random_init = HIL_random_init.train()

# HIL pre-init
initialization = 'pre-train'
HIL_random_init = Imitation_Learning.HIL(T_set, Heading_set, N, initialization, option_space)
HIL_random_init.encode_data()
pi_hi_batch_pre_init, pi_lo_batch_pre_init, pi_b_batch_pre_init = HIL_random_init.train()

# Neural TD0
pi_hi_net_w = pi_hi_batch_pre_init.get_weights()
pi_lo_net_w = []
pi_b_net_w = []
for i in range(option_space):
    pi_lo_net_w.append(pi_lo_batch_pre_init[i].get_weights())
    pi_b_net_w.append(pi_b_batch_pre_init[i].get_weights())
    
NEpisodesTD = 300
seed = 13
TD = NeuralTD.TD0_NN(seed, coins_location, option_space)
critic_nets = []
critic_nets = TD.Options_Evaluation(NEpisodesTD, seed, pi_hi_net_w, pi_lo_net_w, pi_b_net_w, reset = 'standard', initial_state = s0)

# DOC
NEpisodesDOC = 500
seed = 31 
DOC_learning_results = []

deep_option_critic = Option_Critic_with_DQN.DOC(seed, coins_location, option_space, pi_hi_net_w, pi_lo_net_w, pi_b_net_w, critic_nets)
reward_per_episode_OC, traj_OC, Option_OC, Termination_OC, pi_hi, pi_lo, pi_b  = deep_option_critic.Training(NEpisodesDOC, seed, reset = 'standard', initial_state = s0)
DOC_learning_results.append([reward_per_episode_OC, traj_OC, Option_OC, Termination_OC, pi_hi, pi_lo, pi_b])

    

