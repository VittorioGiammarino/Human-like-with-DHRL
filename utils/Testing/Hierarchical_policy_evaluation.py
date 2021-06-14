#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 10:04:09 2021

@author:
"""
import numpy as np
import World
import BatchBW_HIL

    
class HP_eval:
    def __init__(self, seed, coins_location, option_space, pi_hi_weights, pi_lo_weights, pi_b_weights):
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
                        
        self.pi_hi_batch = pi_hi_batch
        self.pi_lo_batch = pi_lo_batch
        self.pi_b_batch = pi_b_batch


    def evaluate(self, NEpisodes, seed, length_episode, reset = 'random', initial_state = np.array([0,0,0,8])):
 
        reward_per_episode =[]
        np.random.seed(seed)
        traj = [[None]*1 for _ in range(NEpisodes)]
        Option = [[None]*1 for _ in range(NEpisodes)]
        Termination = [[None]*1 for _ in range(NEpisodes)]
        
        
        for i_episode in range(NEpisodes):
            
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
            
            for t in range(length_episode):
                
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
                else:
                    b_bool = False                
                 
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
                    new_action = np.amin(np.where(draw_u<=prob_u_rescaled)[1])  
                
                current_state = new_state
                current_state_encoded = new_state_encoded
                x = np.append(x, current_state.reshape(1, self.observation_space_size), 0)
                current_action = new_action
                cum_reward = cum_reward + reward                
             
                
            reward_per_episode.append(cum_reward)
            traj[i_episode] = x
            Option[i_episode] = o_tot
            Termination[i_episode] = b_tot
        
        return traj, Option, Termination, reward_per_episode
