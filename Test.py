#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 10:42:30 2021

@author:
"""

from utils import Deep_Qlearning as DQL
from utils import Imitation_Learning
from utils.Testing import Hierarchical_policy_evaluation as Hpe
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import BatchBW_HIL

import numpy as np
import multiprocessing
import multiprocessing.pool

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
    
Nseed = multiprocessing.cpu_count()

# Humans
with open('Data_and_models/Humans/Real_Reward_eval_human.npy', 'rb') as f:
    Reward_human_eval = np.load(f, allow_pickle=True).tolist()
with open('Data_and_models/Humans/Real_Traj_eval_human.npy', 'rb') as f:
    Trajectories_human_eval = np.load(f, allow_pickle=True).tolist()
    
print('Human average reward = {}'.format(np.mean(Reward_human_eval)))
print('Human standard deviation = {}'.format(np.std(Reward_human_eval)))
print('Human best trajectory reward = {}'.format(np.max(Reward_human_eval)))
print('Human worst trajectory reward = {}'.format(np.min(Reward_human_eval)))

with open('Data_and_models/Data_preprocessed/Reward_human_eval.npy', 'rb') as f:
    Reward_human_eval_processed = np.load(f, allow_pickle=True).tolist()
with open('Data_and_models/Data_preprocessed/Trajectories_human_eval.npy', 'rb') as f:
    Trajectories_human_eval_processed = np.load(f, allow_pickle=True).tolist()
with open('Data_and_models/Data_preprocessed/coins_location_eval.npy', 'rb') as f:
    coins_location_eval = np.load(f, allow_pickle=True)
 
coins_location = coins_location_eval[np.argmax(Reward_human_eval_processed),:,:]
s0 = Trajectories_human_eval_processed[np.argmax(Reward_human_eval_processed)][0,:] #initial state

# DQN
with open('Data_and_models/Deep_Q_Learning/DQN.npy', 'rb') as f:
    DQN = np.load(f, allow_pickle=True).tolist()

def evaluate(seed, coins_location, NEpisodes, initial_state, net_weights, length_episode):
    agent_NN_Q_learning = DQL.Q_learning_NN(seed, coins_location, import_net = True, weights = net_weights)
    reward_per_episode, traj, network_weights = agent_NN_Q_learning.Evaluation(NEpisodes, initial_state, seed, length_episode)
    return reward_per_episode, traj, network_weights

NEpisodes = 100
length_episode = 6000
pool = MyPool(Nseed)
args = [(seed, coins_location, NEpisodes, s0, DQN, length_episode) for seed in range(Nseed)]
print("running for {} seeds".format(Nseed))
DQN_evaluation_results = pool.starmap(evaluate, args) 
pool.close()
pool.join()

averageDQN = []
totRew = []
MaxDQN = []
MinDQN = []

for i in range(len(DQN_evaluation_results)):
    totRew.append(DQN_evaluation_results[i][0])
    MaxDQN.append(np.max(DQN_evaluation_results[i][0]))
    MinDQN.append(np.min(DQN_evaluation_results[i][0]))
    
TotAve_DQN = np.mean(totRew)
STD_DQN = np.std(totRew)
MAX_DQN = np.max(MaxDQN)
MIN_DQN = np.min(MinDQN)
    
print('DQN average reward = {}'.format(TotAve_DQN))
print('DQN standard deviation = {}'.format(STD_DQN))
print('DQN best trajectory reward = {}'.format(MAX_DQN))
print('DQN worst trajectory reward = {}'.format(MIN_DQN))    

# Behavioral Cloning
with open('Data_and_models/Data_preprocessed/Rotation_human_eval.npy', 'rb') as f:
    Rotation_human_eval = np.load(f, allow_pickle=True).tolist()

index = np.argmax(Reward_human_eval_processed)
size_data = len(Trajectories_human_eval_processed[index])-1
T_set = Trajectories_human_eval_processed[index][0:size_data,:]
Heading_set = Rotation_human_eval[index][0:size_data]
BC_agent = Imitation_Learning.BC(T_set, Heading_set)
BC_agent.encode_data()
BC_agent.init_Model()
BC_agent.train(1)
BC_net_weights = BC_agent.model.get_weights()

def evaluateBC(seed, coins_location, NEpisodes, initial_state, weights, length_episode):
    eval_class = Imitation_Learning.BC_evaluate(weights)
    reward_per_episode, traj, control = eval_class.evaluate(coins_location, NEpisodes, initial_state, seed, length_episode)
    return reward_per_episode, traj, control

pool = MyPool(Nseed)
args = [(seed, coins_location, NEpisodes, s0, BC_net_weights, length_episode) for seed in range(Nseed)]
print("running BC for {} seeds".format(Nseed))
BC_evaluation_results = pool.starmap(evaluateBC, args) 
pool.close()
pool.join()

averageBC = []
BC = []
maxBC = []
minBC = []

for i in range(len(BC_evaluation_results)):
    averageBC.append(np.mean(BC_evaluation_results[i][0]))
    BC.append(BC_evaluation_results[i][0])
    maxBC.append(np.max(BC_evaluation_results[i][0]))
    minBC.append(np.min(BC_evaluation_results[i][0]))

BCAve = np.mean(BC)
STDBC = np.std(BC)
MAXBC = np.max(maxBC)
MINBC = np.min(minBC)    

print('BC average reward = {}'.format(BCAve))
print('BC standard deviation = {}'.format(STDBC))
print('BC best trajectory reward = {}'.format(MAXBC))
print('BC worst trajectory reward = {}'.format(MINBC))

# HIL random init
pi_hi_HIL_random_init = BatchBW_HIL.NN_PI_HI.load('Data_and_models/Models_HIL/Saved_Model_Batch/pi_hi_NN_randominit').get_weights()
pi_lo_HIL_random_init = []
pi_b_HIL_random_init = []
option_space = 2
for i in range(option_space):
    pi_lo_HIL_random_init.append(BatchBW_HIL.NN_PI_LO.load('Data_and_models/Models_HIL/Saved_Model_Batch/pi_lo_NN_{}_pi_hi_NN_randominit'.format(i)).get_weights())
    pi_b_HIL_random_init.append(BatchBW_HIL.NN_PI_B.load('Data_and_models/Models_HIL/Saved_Model_Batch/pi_b_NN_{}_pi_hi_NN_randominit'.format(i)).get_weights())
    

def evaluateHIL(seed, coins_location, option_space, pi_hi_weights, pi_lo_weights, pi_b_weights, reset, NEpisodes, init_state, length_episode):
    eval_class = Hpe.HP_eval(seed, coins_location, option_space, pi_hi_weights, pi_lo_weights, pi_b_weights)
    traj, Option, Termination, reward_per_episode = eval_class.evaluate(NEpisodes, seed, length_episode, reset = reset, initial_state = init_state)
    return reward_per_episode, traj, Option, Termination


pool = MyPool(Nseed)
args = [(seed, coins_location, option_space, pi_hi_HIL_random_init, pi_lo_HIL_random_init, pi_b_HIL_random_init, 'standard', NEpisodes, s0, length_episode) for seed in range(Nseed)]
print("running HIL random init for {} seeds".format(Nseed))
HIL_random_init_evaluation_results = pool.starmap(evaluateHIL, args) 
pool.close()
pool.join()

HIL_random_init = []
maxHIL_random_init = []
minHIL_random_init = []

for i in range(len(HIL_random_init_evaluation_results)):
    HIL_random_init.append(HIL_random_init_evaluation_results[i][0])
    maxHIL_random_init.append(np.max(HIL_random_init_evaluation_results[i][0]))
    minHIL_random_init.append(np.min(HIL_random_init_evaluation_results[i][0]))

HIL_random_initAve = np.mean(HIL_random_init)
STDHIL_random_init = np.std(HIL_random_init)
MAXHIL_random_init = np.max(maxHIL_random_init)
MINHIL_random_init = np.min(minHIL_random_init)    

print('HIL random init average reward = {}'.format(HIL_random_initAve))
print('HIL random init standard deviation = {}'.format(STDHIL_random_init))
print('HIL random init best trajectory reward = {}'.format(MAXHIL_random_init))
print('HIL random init worst trajectory reward = {}'.format(MINHIL_random_init))

# HIL pre-init
pi_hi_HIL_pre_init = BatchBW_HIL.NN_PI_HI.load('Data_and_models/Models_HIL/Saved_Model_Batch/pi_hi_NN_preinit').get_weights()
pi_lo_HIL_pre_init = []
pi_b_HIL_pre_init = []
option_space = 2
for i in range(option_space):
    pi_lo_HIL_pre_init.append(BatchBW_HIL.NN_PI_LO.load('Data_and_models/Models_HIL/Saved_Model_Batch/pi_lo_NN_{}_pi_hi_NN_preinit'.format(i)).get_weights())
    pi_b_HIL_pre_init.append(BatchBW_HIL.NN_PI_B.load('Data_and_models/Models_HIL/Saved_Model_Batch/pi_b_NN_{}_pi_hi_NN_preinit'.format(i)).get_weights())
    

def evaluateHIL(seed, coins_location, option_space, pi_hi_weights, pi_lo_weights, pi_b_weights, reset, NEpisodes, init_state, length_episode):
    eval_class = Hpe.HP_eval(seed, coins_location, option_space, pi_hi_weights, pi_lo_weights, pi_b_weights)
    traj, Option, Termination, reward_per_episode = eval_class.evaluate(NEpisodes, seed, length_episode, reset = reset, initial_state = init_state)
    return reward_per_episode, traj, Option, Termination


pool = MyPool(Nseed)
args = [(seed, coins_location, option_space, pi_hi_HIL_pre_init, pi_lo_HIL_pre_init, pi_b_HIL_pre_init, 'standard', NEpisodes, s0, length_episode) for seed in range(Nseed)]
print("running HIL pre-init for {} seeds".format(Nseed))
HIL_pre_init_evaluation_results = pool.starmap(evaluateHIL, args) 
pool.close()
pool.join()

HIL_pre_init = []
maxHIL_pre_init = []
minHIL_pre_init = []

for i in range(len(HIL_pre_init_evaluation_results)):
    HIL_pre_init.append(HIL_pre_init_evaluation_results[i][0])
    maxHIL_pre_init.append(np.max(HIL_pre_init_evaluation_results[i][0]))
    minHIL_pre_init.append(np.min(HIL_pre_init_evaluation_results[i][0]))

HIL_pre_initAve = np.mean(HIL_pre_init)
STDHIL_pre_init = np.std(HIL_pre_init)
MAXHIL_pre_init = np.max(maxHIL_pre_init)
MINHIL_pre_init = np.min(minHIL_pre_init)    

print('HIL pre init average reward = {}'.format(HIL_pre_initAve))
print('HIL pre init standard deviation = {}'.format(STDHIL_pre_init))
print('HIL pre init best trajectory reward = {}'.format(MAXHIL_pre_init))
print('HIL pre init worst trajectory reward = {}'.format(MINHIL_pre_init))

# DOC
with open('Data_and_models/DOC/DOC_pi_hi.npy', 'rb') as f:
    pi_hi_DOC = np.load(f, allow_pickle=True).tolist()

with open('Data_and_models/DOC/DOC_pi_lo.npy', 'rb') as f:
    pi_lo_DOC = np.load(f, allow_pickle=True).tolist()
    
with open('Data_and_models/DOC/DOC_pi_b.npy', 'rb') as f:
    pi_b_DOC = np.load(f, allow_pickle=True).tolist()


def evaluateDOC(seed, coins_location, option_space, pi_hi_weights, pi_lo_weights, pi_b_weights, reset, NEpisodes, init_state, length_episode):
    eval_class = Hpe.HP_eval(seed, coins_location, option_space, pi_hi_weights, pi_lo_weights, pi_b_weights)
    traj, Option, Termination, reward_per_episode = eval_class.evaluate(NEpisodes, seed, length_episode, reset = reset, initial_state = init_state)
    return reward_per_episode, traj, Option, Termination


pool = MyPool(Nseed)
args = [(seed, coins_location, option_space, pi_hi_DOC, pi_lo_DOC, pi_b_DOC, 'standard', NEpisodes, s0, length_episode) for seed in range(Nseed)]
print("running DOC for {} seeds".format(Nseed))
DOC_evaluation_results = pool.starmap(evaluateDOC, args) 
pool.close()
pool.join()

DOC = []
maxDOC = []
minDOC = []

for i in range(len(DOC_evaluation_results)):
    DOC.append(DOC_evaluation_results[i][0])
    maxDOC.append(np.max(DOC_evaluation_results[i][0]))
    minDOC.append(np.min(DOC_evaluation_results[i][0]))

DOCAve = np.mean(DOC)
STDDOC = np.std(DOC)
MAXDOC = np.max(maxDOC)
MINDOC = np.min(minDOC)    

print('DOC average reward = {}'.format(DOCAve))
print('DOC standard deviation = {}'.format(STDDOC))
print('DOC best trajectory reward = {}'.format(MAXDOC))
print('DOC worst trajectory reward = {}'.format(MINDOC))


with open('Data_and_models/RL_algorithms/DeepQ_Learning/Q_learning_evaluation.npy', 'wb') as f:
    np.save(f, DQN_evaluation_results)

with open('Data_and_models/Results_main/BC_from_human_evaluation_results.npy', 'wb') as f:
    np.save(f, BC_evaluation_results)
    
with open('Data_and_models/Results_main/HIL_from_human_evaluation_results_random_init.npy', 'wb') as f:
    np.save(f, HIL_random_init_evaluation_results)

with open('Data_and_models/Results_main/HIL_from_human_evaluation_results.npy', 'wb') as f:
    np.save(f, HIL_pre_init_evaluation_results)
    
with open('Data_and_models/RL_algorithms/Option_critic_with_DQN/DeepSoftOC_learning_evaluation.npy', 'wb') as f:
    np.save(f, DOC_evaluation_results)







