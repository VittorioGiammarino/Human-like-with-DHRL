#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 12:31:43 2021

@author: 
"""

import numpy as np
import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as kb
import World
import BatchBW_HIL

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
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.cost_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, state_encoded, action, reward, cost, state_, state_encoded_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.state_memory_encoded[index] = state_encoded
        self.new_state_memory[index] = state_
        self.new_state_memory_encoded[index] = state_encoded_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.cost_memory[index] = cost
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_encoded = self.state_memory_encoded[batch]
        states_ = self.new_state_memory[batch]
        states_encoded_ = self.new_state_memory_encoded[batch]
        rewards = self.reward_memory[batch]
        cost = self.cost_memory[batch]
        actions = self.action_memory[batch]
        
        return states, states_encoded, actions, rewards, cost, states_, states_encoded_
    
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

class DOC:
    def __init__(self, seed, coins_location, option_space, pi_hi_weights, pi_lo_weights, pi_b_weights, DQN_weights):
        self.env = World.Foraging.env(coins_location)
        np.random.seed(seed)
        self.observation_space_size = self.env.observation_size
        self.option_space = option_space
        self.zeta = 0
        self.eta = 0.00001
        self.coordinates = 2
        self.view = 2
        self.closest_coin_dir = 9
        self.observation_space_size_encoded = self.coordinates + self.view + self.closest_coin_dir
        
        Buffer = []     
        DQN = []   
        DDQN = []
        DVN = []
        DVN_target = []
        for options in range(self.option_space):
            Buffer.append(ReplayBuffer(10*3000, self.observation_space_size, self.observation_space_size_encoded, seed))
            DQN.append(NN_model(self.observation_space_size, self.env.action_size, 1))
            DDQN.append(NN_model(self.observation_space_size, self.env.action_size, 2))
            DVN.append(NN_model(self.observation_space_size, 1, 3))
            DVN_target.append(NN_model(self.observation_space_size, 1, 4))
            
        self.Buffer = Buffer
        
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
            
            DQN[i].set_weights(DQN_weights[0][i])
            DDQN[i].set_weights(DQN_weights[1][i])
            
        self.pi_hi_batch = pi_hi_batch
        self.pi_lo_batch = pi_lo_batch
        self.pi_b_batch = pi_b_batch
        self.Q_net = DQN
        self.DQ_net = DDQN
        self.V_net = DVN
        self.V_net_target = DVN_target
        self.optimizer_pi_lo = keras.optimizers.Adamax(learning_rate=1e-3)
        self.optimizer_pi_b = keras.optimizers.Adamax(learning_rate=1e-5)

    
    def Training(self, NEpisodes, seed, reset = 'random', initial_state = np.array([0,0,0,8])):
        
        gamma = 0.99 
        tau = 0.005
        reward_per_episode =[]
        batch_size = 512
        np.random.seed(seed)
        traj = [[None]*1 for _ in range(NEpisodes)]
        Option = [[None]*1 for _ in range(NEpisodes)]
        Termination = [[None]*1 for _ in range(NEpisodes)]
        pi_hi = [[None]*1 for _ in range(NEpisodes)]
        pi_lo = [[None]*1 for _ in range(NEpisodes)]
        pi_b = [[None]*1 for _ in range(NEpisodes)]
        
        
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
            x = np.append(x, current_state.reshape(1, self.observation_space_size), 0)
            
            # initial cost
            cost = 0
            
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
                
                self.Buffer[option].store_transition(current_state, current_state_encoded, current_action, reward, cost, new_state, new_state_encoded)
                
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
                    cost = self.eta
                else:
                    b_bool = False                
                 
                # training here TODO
                if self.Buffer[option].mem_cntr>batch_size and np.mod(T,5)==0:
                    
                    states, states_encoded, actions, rewards, costs, new_states, new_states_encoded = self.Buffer[option].sample_buffer(batch_size)
                    rewards_hat = rewards-costs
                    for option_i in range(self.option_space):
                        Q_new_states_values = self.Q_net[option_i](new_states)
                        DQ_new_states_values = self.DQ_net[option_i](new_states)
                        y_v = kb.sum(self.pi_lo_batch[option_i](new_states_encoded)*Q_new_states_values,1)
                        y_vD = kb.sum(self.pi_lo_batch[option_i](new_states_encoded)*DQ_new_states_values,1)
                        y_vFinal = np.min(np.concatenate((y_v.numpy().reshape(batch_size,1), y_vD.numpy().reshape(batch_size,1)),1),1)
                        self.V_net[option_i].fit(new_states, y_vFinal, epochs=1, verbose = 0)
                        v_weights = self.V_net[option_i].get_weights()
                        v_target_weights = self.V_net_target[option_i].get_weights()
                        
                        if T==1:
                            self.V_net_target[option_i].set_weights(v_weights)
                        else:
                            for i in range(len(v_target_weights)):
                                v_target_weights[i] = (1-tau)*v_target_weights[i]+tau*v_weights[i]
                            self.V_net_target[option_i].set_weights(v_target_weights)
                        
                        
                        
                        
                    target_same_option = gamma*self.pi_b_batch[option](new_states_encoded)[:,0]*self.V_net_target[option](new_states)[:,0] #kb.sum(self.pi_lo_batch[option](new_states_encoded)*Q_new_states_values,1) 
                    V_new_states_values = 0
                    for i_option in range(self.option_space):
                        V_new_states_values = V_new_states_values + self.pi_hi_batch(new_states_encoded)[:,i_option]*self.V_net_target[i_option](new_states)[:,0]
                    target_new_option = gamma*self.pi_b_batch[option](new_states_encoded)[:,1]*V_new_states_values
                    y_j = (rewards_hat + target_same_option + target_new_option).numpy()
                    
                    y = self.Q_net[option](states).numpy()
                    y[np.arange(batch_size),actions] = y_j[np.arange(batch_size)]
                    self.Q_net[option].fit(states, keras.utils.normalize(y,1), epochs=1, verbose = 0)
                    
                    yD = self.DQ_net[option](states).numpy()
                    yD[np.arange(batch_size),actions] = y_j[np.arange(batch_size)]
                    self.DQ_net[option].fit(states, keras.utils.normalize(yD,1), epochs=1, verbose = 0)                    
                    
                    m = actions.shape[0]
                    n = self.env.action_size
                    auxiliary_vector = np.zeros((m,n))
                    for k in range(m):
                        auxiliary_vector[k,actions[k]] = 1
                    
                    # auxiliary_vector = self.pi_lo_batch[option](states_encoded)
                    
                    Q_values = self.Q_net[option](states)
                    DQ_values = self.DQ_net[option](states)
                    Q_vFinal = np.min(np.concatenate((Q_values.numpy().reshape(batch_size,self.env.action_size,1), DQ_values.numpy().reshape(batch_size,self.env.action_size,1)),2),2)
                    
                    weights_lo = []
                    with tf.GradientTape() as tape:
                        weights_lo.append(self.pi_lo_batch[option].trainable_weights)                    
                        tape.watch(weights_lo)
                        loss_lo = -tf.math.reduce_mean(auxiliary_vector*kb.log(kb.clip(self.pi_lo_batch[option](states_encoded),1e-10,1.0))*Q_vFinal)
            
                    grads_lo = tape.gradient(loss_lo,weights_lo)

                    self.optimizer_pi_lo.apply_gradients(zip(grads_lo[0][:], self.pi_lo_batch[option].trainable_weights))
                    
                    
                    weights_b = []
                    with tf.GradientTape() as tape:
                        weights_b.append(self.pi_b_batch[option].trainable_weights)                    
                        tape.watch(weights_b)
                        Q_new_states_values_expected = self.V_net_target[option](new_states)[:,0]
                        V_new_states_values = 0
                        for i_option in range(self.option_space):
                            V_new_states_values = V_new_states_values + self.pi_hi_batch(new_states_encoded)[:,i_option]*self.V_net_target[i_option](new_states)[:,0]
                        loss_b = kb.sum(self.pi_b_batch[option](new_states_encoded)[:,1]*(Q_new_states_values_expected - V_new_states_values + self.eta*kb.ones((batch_size,))))/batch_size
            
                    grads_b = tape.gradient(loss_b,weights_b)

                    self.optimizer_pi_b.apply_gradients(zip(grads_b[0][:], self.pi_b_batch[option].trainable_weights))                    
                 
                    
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
                option = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                o_tot = np.append(o_tot,option)
                
                # draw next action
                prob_u = self.pi_lo_batch[option](new_state_encoded.reshape(1, self.observation_space_size_encoded)).numpy()
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
                
                current_state = new_state
                current_state_encoded = new_state_encoded
                x = np.append(x, current_state.reshape(1, self.observation_space_size), 0)
                current_action = new_action
                cum_reward = cum_reward + reward                
             
            print("Episode {}: cumulative reward = {} (seed = {})".format(i_episode, cum_reward, seed))
            reward_per_episode.append(cum_reward)
            traj[i_episode] = x
            Option[i_episode] = o_tot
            Termination[i_episode] = b_tot
            pi_hi[i_episode] = self.pi_hi_batch.get_weights()
            pi_lo_temp = []
            pi_b_temp = []
            for i in range(self.option_space):
                pi_lo_temp.append(self.pi_lo_batch[i].get_weights())
                pi_b_temp.append(self.pi_b_batch[i].get_weights())
                
            pi_lo[i_episode] = pi_lo_temp
            pi_b[i_episode] = pi_b_temp
            
        return reward_per_episode, traj, Option, Termination, pi_hi, pi_lo, pi_b
    








