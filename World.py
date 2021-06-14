#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:13:05 2020

@author:
"""
import numpy as np

class Foraging:        
    def GetDirectionFromAngle(angle):           
        if angle<0:
            angle = angle + 360
        slots = np.arange(22.5,410,45)
        label_direction = np.min(np.where(angle<=slots)[0])
        if label_direction==8:
            label_direction = 0            
         
        return label_direction
        
    class env:
        def __init__(self,  coins_location, init_state = np.array([0,0,0,8])):
            self.state = init_state
            self.coins_location_initial = 0.1*coins_location
            self.coin_location = 0.1*coins_location
            self.observation_size = len(self.state)
            self.action_size = 8

        def reset(self, version = 'standard', init_state = np.array([0,0,0,8])):
            if version == 'standard':
                self.state = init_state
                self.coin_location = self.coins_location_initial
            else:
                state = 0.1*np.random.randint(-100,100,2)
                init_state = np.concatenate((state, np.array([0,8])))
                self.state = init_state
                self.coin_location = self.coins_location_initial
                
            return self.state
                
        def Transition(state,action):
            Transition = np.zeros((9,2))
            Transition[0,0] = state[0] + 0.1
            Transition[0,1] = state[1] + 0
            Transition[1,0] = state[0] + 0.1
            Transition[1,1] = state[1] + 0.1
            Transition[2,0] = state[0] + 0
            Transition[2,1] = state[1] + 0.1
            Transition[3,0] = state[0] - 0.1
            Transition[3,1] = state[1] + 0.1
            Transition[4,0] = state[0] - 0.1
            Transition[4,1] = state[1] + 0
            Transition[5,0] = state[0] - 0.1
            Transition[5,1] = state[1] - 0.1
            Transition[6,0] = state[0] + 0
            Transition[6,1] = state[1] - 0.1
            Transition[7,0] = state[0] + 0.1
            Transition[7,1] = state[1] - 0.1
            Transition[8,:] = state
            state_plus1 = Transition[int(action),:]
            
            return state_plus1     
    
        def step(self, action):
            
            r=0
            state_partial = self.state[0:2]
            # given action, draw next state
            state_plus1_partial = Foraging.env.Transition(state_partial, action)
                
            if state_plus1_partial[0]>10 or state_plus1_partial[0]<-10:
                state_plus1_partial[0] = state_partial[0] 

            if state_plus1_partial[1]>10 or state_plus1_partial[1]<-10:
                state_plus1_partial[1] = state_partial[1]                 
                    
            # Update psi and reward and closest coin direction
            dist_from_coins = np.linalg.norm(self.coin_location-state_plus1_partial,2,1)
            l=0
            psi = 0
                
            if np.min(dist_from_coins)<=0.8:
                index_min = np.argmin(dist_from_coins,0)
                closer_coin_position = self.coin_location[index_min,:]
                closer_coin_relative_position = np.array([closer_coin_position[0]-state_plus1_partial[0],closer_coin_position[1]-state_plus1_partial[1]])
                angle = np.arctan2(closer_coin_relative_position[1],closer_coin_relative_position[0])*180/np.pi
                coin_direction = Foraging.GetDirectionFromAngle(angle)  
            else:
                coin_direction = 8   
            
            for p in range(len(dist_from_coins)):
                if dist_from_coins[p]<=0.8:
                    psi = 1
                if dist_from_coins[p]<=0.3:
                    self.coin_location = np.delete(self.coin_location, l, 0)
                    r = r+1
                else:
                    l=l+1
                    
            state_plus1 = np.concatenate((state_plus1_partial, [psi], [coin_direction]))
            self.state = state_plus1
            
            return state_plus1, r
        
        
    
    
