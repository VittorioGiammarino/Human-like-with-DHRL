#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 15:51:59 2021

@author:
"""

import numpy as np
import tensorflow as tf 
from tensorflow import keras
import World
import BatchBW_HIL
import tensorflow.keras.backend as kb

# %%
class ReplayBuffer():
    def __init__(self, max_size, input_dims, input_dims_encoded, seed):
        np.random.seed(seed)
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.state_memory_encoded = np.zeros((self.mem_size, input_dims_encoded), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory_encoded = np.zeros((self.mem_size, input_dims_encoded), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.new_action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_option_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, state_encoded, action, reward, state_, state_encoded_, option_, action_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.state_memory_encoded[index] = state_encoded
        self.new_state_memory[index] = state_
        self.new_state_memory_encoded[index] = state_encoded_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.new_action_memory[index] = action_
        self.new_option_memory[index] = option_
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_encoded = self.state_memory_encoded[batch]
        states_ = self.new_state_memory[batch]
        states_encoded_ = self.new_state_memory_encoded[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        actions_ = self.new_action_memory[batch]
        options_ = self.new_option_memory[batch]
        
        return states, states_encoded, actions, rewards, states_, states_encoded_, options_, actions_
    
def NN_model(input_size, output_size, seed_init):
    model = keras.Sequential([             
            keras.layers.Dense(256, activation='relu', input_shape=(input_size,),
                               kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=seed_init),
                               bias_initializer=keras.initializers.Zeros()),        
            keras.layers.Dense(256, activation='relu',
                               kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=seed_init),
                               bias_initializer=keras.initializers.Zeros()),                         
            keras.layers.Dense(output_size, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=seed_init))
                             ])         

    model.compile(optimizer='adam',
                       loss=tf.keras.losses.MeanSquaredError(),
                       metrics=['accuracy'])     
    
    return model
        
        
class TD0_NN:
    def __init__(self, seed, coins_location, option_space):
        self.env = World.Foraging.env(coins_location)
        np.random.seed(seed)
        self.observation_space_size = self.env.observation_size
        self.option_space = option_space
        self.coordinates = 2
        self.view = 2
        self.closest_coin_dir = 9
        self.observation_space_size_encoded = self.coordinates + self.view + self.closest_coin_dir
        self.zeta = 0
        
        DQN = []
        Buffer = []
        DDQN = []
        DVN = []
        DVN_target = []
        
        for options in range(self.option_space):
            DQN.append(NN_model(self.observation_space_size, self.env.action_size, 1))
            Buffer.append(ReplayBuffer(30000, self.observation_space_size, self.observation_space_size_encoded, seed))
            DDQN.append(NN_model(self.observation_space_size, self.env.action_size, 2))
            DVN.append(NN_model(self.observation_space_size, 1, 3))
            DVN_target.append(NN_model(self.observation_space_size, 1, 4))
            
        self.Q_net = DQN
        self.DQ_net = DDQN
        self.Buffer = Buffer
        self.V_net = DVN
        self.V_net_target = DVN_target        

    
    def Options_Evaluation(self, NEpisodes, seed, pi_hi_weights, pi_lo_weights, pi_b_weights, reset = 'random', initial_state = np.array([0,0,0,8])):
        
        pi_hi = BatchBW_HIL.NN_PI_HI(self.option_space, self.observation_space_size_encoded)
        pi_hi_batch = pi_hi.NN_model()
        pi_hi_batch.set_weights(pi_hi_weights)
        
        pi_lo_batch = []
        pi_b_batch = []
        pi_lo_class = BatchBW_HIL.NN_PI_LO(self.env.action_size, self.observation_space_size_encoded)
        pi_b_class = BatchBW_HIL.NN_PI_B(2, self.observation_space_size_encoded)
        for i in range(self.option_space):
            pi_lo_batch.append(pi_lo_class.NN_model())
            pi_lo_batch[i].set_weights(pi_lo_weights[i])

            pi_b_batch.append(pi_b_class.NN_model())
            pi_b_batch[i].set_weights(pi_b_weights[i])
            
        gamma = 0.99 
        reward_per_episode =[]
        batch_size = 512
        self.pi_lo_batch = pi_lo_batch
        self.pi_hi_batch = pi_hi_batch
        self.pi_b_batch = pi_b_batch
        
        for i_episode in range(NEpisodes):
            
            T = i_episode + 1
            o_tot = np.empty((0,0),int)
            b_tot = np.empty((0,0),int)
            x = np.empty((0, self.observation_space_size))

            current_state = self.env.reset(reset, initial_state)
            coordinates = current_state[0:2]
            psi = current_state[2]
            psi_encoded = np.zeros(self.view)
            psi_encoded[int(psi)]=1
            coin_dir_encoded = np.zeros(self.closest_coin_dir)
            coin_dir = current_state[3]
            coin_dir_encoded[int(coin_dir)]=1
            current_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))
            cum_reward = 0 
            
            # Initial Option
            prob_o = self.pi_hi_batch(current_state_encoded.reshape(1, self.observation_space_size_encoded)).numpy()
            prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
            for i in range(1,prob_o_rescaled.shape[1]):
                prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
            draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
            option = np.amin(np.where(draw_o<prob_o_rescaled))
            o_tot = np.append(o_tot,option)            
            
            # draw action
            prob_u = self.pi_lo_batch[option](current_state_encoded.reshape(1, self.observation_space_size_encoded)).numpy()
            prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
            for i in range(1,prob_u_rescaled.shape[1]):
                prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
            draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
            current_action = np.amin(np.where(draw_u<prob_u_rescaled)[1])
            
            
            for t in range(3000):
                
                obs, reward = self.env.step(current_action)
                new_state = obs
                
                coordinates = new_state[0:2]
                psi = new_state[2]
                psi_encoded = np.zeros(self.view)
                psi_encoded[int(psi)]=1
                coin_dir_encoded = np.zeros(self.closest_coin_dir)
                coin_dir = new_state[3]
                coin_dir_encoded[int(coin_dir)]=1
                new_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))
                
                # Termination
                prob_b = self.pi_b_batch[option](new_state_encoded.reshape(1, self.observation_space_size_encoded)).numpy()
                prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                for i in range(1,prob_b_rescaled.shape[1]):
                    prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                temp = np.where(draw_b<=prob_b_rescaled)[1]
                if temp.size == 0:
                    draw_b = 1
                    b = np.argmax(prob_b)
                else:
                    b = np.amin(np.where(draw_b<=prob_b_rescaled)[1])
                    
                b_tot = np.append(b_tot,b)
                if b == 1:
                    b_bool = True
                    #cost = self.eta
                else:
                    b_bool = False      
                  
                #new option    
                o_prob_tilde = np.empty((1,self.option_space))
                if b_bool == True:
                    o_prob_tilde = self.pi_hi_batch(new_state_encoded.reshape(1, self.observation_space_size_encoded)).numpy()
                else:
                    o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                    o_prob_tilde[0,option] = 1 - self.zeta + self.zeta/self.option_space
                
                prob_o = o_prob_tilde
                prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                for i in range(1,prob_o_rescaled.shape[1]):
                    prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                new_option = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                o_tot = np.append(o_tot,new_option)
                
                # draw next action
                prob_u = self.pi_lo_batch[new_option](new_state_encoded.reshape(1, self.observation_space_size_encoded)).numpy()
                prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                for i in range(1,prob_u_rescaled.shape[1]):
                    prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                temp = np.where(draw_u<=prob_u_rescaled)[1]
                if temp.size == 0:
                    draw_u = 1
                    new_action = np.argmax(prob_u)
                else:
                    new_action = np.amin(np.where(draw_u<prob_u_rescaled)[1])  
                    
                
                self.Buffer[option].store_transition(current_state, current_state_encoded, current_action, reward, new_state, new_state_encoded, new_option, new_action)
                                 
                current_state = new_state
                current_state_encoded = new_state_encoded
                x = np.append(x, current_state.reshape(1, self.observation_space_size), 0)
                current_action = new_action
                cum_reward = cum_reward + reward  
                
                if self.Buffer[option].mem_cntr>batch_size and np.mod(T,5)==0:
                    
                    states, current_states_encoded, actions, rewards, new_states, new_states_encoded, new_options, new_actions = self.Buffer[option].sample_buffer(batch_size)
                    
                    m = new_actions.shape[0]
                    auxiliary_vector_options = np.zeros((m, self.option_space))
                    auxiliary_vector_actions = np.zeros((m, self.env.action_size))
                    for k in range(m):
                            auxiliary_vector_options[k, new_options[k]] = 1
                            auxiliary_vector_actions[k, new_actions[k]] = 1
                    
                    Q_ = 0 
                    DQ_ = 0
                    for i_option in range(self.option_space):
                        Q_ = Q_ + auxiliary_vector_options[:,i_option]*kb.sum(auxiliary_vector_actions*self.Q_net[i_option](new_states),1)
                        DQ_ = DQ_ + auxiliary_vector_options[:,i_option]*kb.sum(auxiliary_vector_actions*self.DQ_net[i_option](new_states),1)
                    
                    learned_value = (rewards + gamma*Q_).numpy()
                    Dlearned_value = (rewards + gamma*DQ_).numpy()

                    y = self.Q_net[option](states).numpy()
                    y[np.arange(batch_size),actions] = learned_value[np.arange(batch_size)]
                    self.Q_net[option].fit(states, keras.utils.normalize(y,1), epochs=1, verbose = 0)
                    
                    yD = self.DQ_net[option](states).numpy()
                    yD[np.arange(batch_size),actions] = Dlearned_value[np.arange(batch_size)]
                    self.DQ_net[option].fit(states, keras.utils.normalize(yD,1), epochs=1, verbose = 0)
                    
                option = new_option
                
                
            print("Episode {}: cumulative reward = {} (seed = {}, option = {})".format(i_episode, cum_reward, seed, option))
            reward_per_episode.append(cum_reward)
        
            
        network_weights = []
        Dnetwork_weights = []
        for i in range(self.option_space):
            network_weights.append(self.Q_net[i].get_weights())
            Dnetwork_weights.append(self.DQ_net[i].get_weights())
            
        return network_weights, Dnetwork_weights 


