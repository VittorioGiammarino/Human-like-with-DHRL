#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 21:59:06 2021

@author:
"""
import numpy as np
from tensorflow import keras
import tensorflow as tf 
from sklearn.preprocessing import OneHotEncoder
import BatchBW_HIL
import World

# %% Behavioral Cloning

class BC:
    def __init__(self, Traj, Labels):
        self.T_set = Traj
        self.Heading_set = Labels
            
    def encode_data(self):
        # encode psi
        psi = self.T_set[:,2].reshape(len(self.T_set[:,2]),1)
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded_psi = onehot_encoder.fit_transform(psi)
        # encode closest coin direction
        closest_coin_direction = self.T_set[:,3].reshape(len(self.T_set[:,3]),1)
        onehot_encoded_closest_coin_direction = onehot_encoder.fit_transform(closest_coin_direction)
        coordinates = self.T_set[:,0:2].reshape(len(self.T_set[:,0:2]),2)
        Data_set = np.concatenate((coordinates,onehot_encoded_psi,onehot_encoded_closest_coin_direction),1)
        observation_space_size = Data_set.shape[1]
        action_size = len(np.unique(self.Heading_set))
        
        self.encoded_Data = Data_set
        self.obs_space_size = observation_space_size
        self.action_size = action_size
        
    def init_Model(self):
        model_BC = keras.Sequential([             
                keras.layers.Dense(512, activation='relu', input_shape=(self.obs_space_size,),
                                   kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=0),
                                   bias_initializer=keras.initializers.Zeros()),                                
                keras.layers.Dense(self.action_size, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=2)),
                keras.layers.Softmax()
                                 ])              
        
        model_BC.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
        
        self.model = model_BC
        
    def train(self, verb):
        self.model.fit(self.encoded_Data, self.Heading_set, epochs=200, verbose = verb)
        
class BC_evaluate:
    def __init__(self, weights):
        self.hidden = len(weights[1])
        self.obs_space_size = weights[0].shape[0]
        self.action_size = len(weights[-1])
        self.model = BC_evaluate.init_Model(self)
        self.model.set_weights(weights)
        
    def init_Model(self):
        model_BC = keras.Sequential([             
                keras.layers.Dense(self.hidden, activation='relu', input_shape=(self.obs_space_size,),
                                   kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=0),
                                   bias_initializer=keras.initializers.Zeros()),                                
                keras.layers.Dense(self.action_size, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=2)),
                keras.layers.Softmax()
                                 ])              
        
        model_BC.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
        
        return model_BC
        
    def evaluate(self, coins_location, NEpisodes, initial_state, seed, length_ep):
        reward_per_episode =[]
        traj = [[None]*1 for _ in range(NEpisodes)]
        control = [[None]*1 for _ in range(NEpisodes)]
        np.random.seed(seed)
        observation_space_size = self.obs_space_size
        env = World.Foraging.env(coins_location, init_state = initial_state)
        model = self.model
    
        for i_episode in range(NEpisodes):
            x = np.empty((0, observation_space_size))
            u = np.empty((0, 1))
            current_state = env.reset('standard', init_state = initial_state)
            coordinates = current_state[0:2]
            psi = current_state[2]
            psi_encoded = np.zeros(2) #psi dimension = 2
            psi_encoded[int(psi)]=1
            coin_dir_encoded = np.zeros(9) # coin dir dimension = 9
            coin_dir = current_state[3]
            coin_dir_encoded[int(coin_dir)]=1
            current_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))            
            cum_reward = 0 
            x = np.append(x, current_state_encoded.reshape(1, observation_space_size), 0)
            
            for t in range(length_ep):
                # draw action
                prob_u = model(current_state_encoded.reshape(1, observation_space_size)).numpy()
                prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                for i in range(1,prob_u_rescaled.shape[1]):
                    prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                action = np.amin(np.where(draw_u<prob_u_rescaled)[1])
                                   
                obs, reward = env.step(action)
                current_state = obs
                coordinates = current_state[0:2]
                psi = current_state[2]
                psi_encoded = np.zeros(2)
                psi_encoded[int(psi)]=1
                coin_dir_encoded = np.zeros(9)
                coin_dir = current_state[3]
                coin_dir_encoded[int(coin_dir)]=1
                current_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))   
                x = np.append(x, current_state_encoded.reshape(1, observation_space_size), 0)
                u = np.append(u, [[action]], 0)
                cum_reward = cum_reward + reward
                                    
            reward_per_episode.append(cum_reward)
            traj[i_episode] = x    
            control[i_episode] = u
        
        return  reward_per_episode, traj, control
    
            
class HIL:
    def __init__(self, Traj, Labels, N, initialization, option_space):
        self.T_set = Traj
        self.Heading_set = Labels
        self.option_space = option_space
        self.M_step_epoch = 10
        self.size_batch = 32
        self.optimizer = keras.optimizers.Adamax(learning_rate=1e-1)
        self.N = N
        self.initialization = initialization
        
    def encode_data(self):
        # encode psi
        psi = self.T_set[:,2].reshape(len(self.T_set[:,2]),1)
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded_psi = onehot_encoder.fit_transform(psi)
        # encode closest coin direction
        closest_coin_direction = self.T_set[:,3].reshape(len(self.T_set[:,3]),1)
        onehot_encoded_closest_coin_direction = onehot_encoder.fit_transform(closest_coin_direction)
        coordinates = self.T_set[:,0:2].reshape(len(self.T_set[:,0:2]),2)
        Data_set = np.concatenate((coordinates,onehot_encoded_psi,onehot_encoded_closest_coin_direction),1)
        observation_space_size = Data_set.shape[1]
        action_size = len(np.unique(self.Heading_set))
        
        self.encoded_Data = Data_set
        self.obs_space_size = observation_space_size
        self.action_size = action_size
        
    def train(self):
        Agent_BatchHIL = BatchBW_HIL.BatchHIL(self.encoded_Data, self.Heading_set, self.option_space, self.M_step_epoch, self.size_batch, self.optimizer, NN_init=self.initialization)
        pi_hi_batch, pi_lo_batch, pi_b_batch = Agent_BatchHIL.Baum_Welch(self.N)
        
        return pi_hi_batch, pi_lo_batch, pi_b_batch
        