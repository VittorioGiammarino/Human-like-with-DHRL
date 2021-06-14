#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:24:37 2021

@author: 
"""
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import numpy as np

def Human_plot_and_stats(Trajectories, Reward_eval_human, coins_location, Rand_traj):
    
    time = np.linspace(0,480,len(Trajectories[Rand_traj][:,0]))         
    sigma1 = 0.5
    circle1 = ptch.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
    sigma2 = 1.1
    circle2 = ptch.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
    sigma3 = 1.8
    circle3 = ptch.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
    sigma4 = 1.3
    circle4 = ptch.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
    fig, ax = plt.subplots()
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    plot_data = plt.scatter(0.1*Trajectories[Rand_traj][:,0], 0.1*Trajectories[Rand_traj][:,1], c=time, marker='o', cmap='cool') 
    plt.plot(0.1*coins_location[2,:,0], 0.1*coins_location[2,:,1], 'xb')
    cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
    cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Best Human traj, Reward {}'.format(Reward_eval_human[Rand_traj]))
    plt.savefig('Figures/FiguresExpert/Best_human_traj.eps', format='eps')
    plt.show()  
    
    sigma1 = 0.5
    circle1 = ptch.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
    sigma2 = 1.1
    circle2 = ptch.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
    sigma3 = 1.8
    circle3 = ptch.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
    sigma4 = 1.3
    circle4 = ptch.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
    fig, ax = plt.subplots()
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4) #-1], marker='o', cmap='cool')
    plt.plot(0.1*coins_location[2,:,0], 0.1*coins_location[2,:,1], 'xb')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Figures/FiguresExpert/Coins_Only.eps', format='eps')
    plt.show()  
    
    print('Human average reward = {}'.format(np.mean(Reward_eval_human)))
    print('Human standard deviation = {}'.format(np.std(Reward_eval_human)))
    print('Human best trajectory reward = {}'.format(np.max(Reward_eval_human)))
    print('Human worst trajectory reward = {}'.format(np.min(Reward_eval_human)))
    
def DQN_plot_and_stats(Training_DQN, DQN_Evaluation, coins_location):
    averageDQN = []
    totRew = []
    MaxDQN = []
    MinDQN = []

    for i in range(len(DQN_Evaluation)):
        averageDQN.append(np.mean(DQN_Evaluation[i][0]))
        totRew.append(DQN_Evaluation[i][0])
        MaxDQN.append(np.max(DQN_Evaluation[i][0]))
        MinDQN.append(np.min(DQN_Evaluation[i][0]))
    best_index_agent = np.argmax(MaxDQN)
    
    best_reward_index=np.argmax(DQN_Evaluation[best_index_agent][0])
    best_episode=DQN_Evaluation[best_index_agent][1][best_reward_index]
    
    time = np.linspace(0,480,len(best_episode[:,0]))  
    sigma1 = 0.5
    circle1 = ptch.Circle((6.0, 7.5), 2*sigma1, color='k',  fill=False)
    sigma2 = 1.1
    circle2 = ptch.Circle((-1.5, -5.0), 2*sigma2, color='k',  fill=False)
    sigma3 = 1.8
    circle3 = ptch.Circle((-5.0, 3.0), 2*sigma3, color='k',  fill=False)
    sigma4 = 1.3
    circle4 = ptch.Circle((4.9, -4.0), 2*sigma4, color='k',  fill=False)
    fig, ax = plt.subplots()
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    plot_data = plt.scatter(best_episode[:,0], best_episode[:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
    plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
    cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
    cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Best DQN Traj, reward {}'.format(DQN_Evaluation[best_index_agent][0][best_reward_index]))
    plt.savefig('Figures/FiguresDQN/DQN_Traj_example.eps', format='eps')
    plt.show() 
    
    episodes = np.arange(0,len(Training_DQN))
    plt.plot(episodes, Training_DQN,'g', label = 'DQN agent')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training DQN')
    plt.legend()
    plt.savefig('Figures/FiguresDQN/DQN_training_trend_unsmoothed.eps', format='eps')
    plt.show() 
    
    episodes = np.arange(0,len(Training_DQN))
    z = np.polyfit(episodes, Training_DQN, 10)
    p = np.poly1d(z)
    plt.plot(episodes,p(episodes), 'g', label = 'DQN agent')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training DQN')
    plt.legend()
    plt.savefig('Figures/FiguresDQN/DQN_training_trend_smoothed.eps', format='eps')
    plt.show() 
    
    TotAve = np.mean(totRew)
    STD = np.std(totRew)
    MAX_DQN = np.max(MaxDQN)
    MIN_DQN = np.min(MinDQN)
    
    print('DQN average reward = {}'.format(TotAve))
    print('DQN standard deviation = {}'.format(STD))
    print('DQN best trajectory reward = {}'.format(MAX_DQN))
    print('DQN worst trajectory reward = {}'.format(MIN_DQN))
    
def BC_plot_and_stats(BC_evaluation, coins_location):
    
    averageBC = []
    BC = []
    maxBC = []
    minBC = []
    
    for i in range(len(BC_evaluation)):
        averageBC.append(np.mean(BC_evaluation[i][0]))
        BC.append(BC_evaluation[i][0])
        maxBC.append(np.max(BC_evaluation[i][0]))
        minBC.append(np.min(BC_evaluation[i][0]))
        
    best_index_agentBC = np.argmax(maxBC)
    best_reward_indexBC=np.argmax(BC_evaluation[best_index_agentBC][0])
    best_episodeBC=BC_evaluation[best_index_agentBC][1][best_reward_indexBC]
    
    
    time = np.linspace(0,480,len(best_episodeBC[:,0]))  
    sigma1 = 0.5
    circle1 = ptch.Circle((6.0, 7.5), 2*sigma1, color='k',  fill=False)
    sigma2 = 1.1
    circle2 = ptch.Circle((-1.5, -5.0), 2*sigma2, color='k',  fill=False)
    sigma3 = 1.8
    circle3 = ptch.Circle((-5.0, 3.0), 2*sigma3, color='k',  fill=False)
    sigma4 = 1.3
    circle4 = ptch.Circle((4.9, -4.0), 2*sigma4, color='k',  fill=False)
    fig, ax = plt.subplots()
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    plot_data = plt.scatter(best_episodeBC[:,0], best_episodeBC[:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
    plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xb')
    cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
    cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Best BC Traj, reward {}'.format(BC_evaluation[best_index_agentBC][0][best_reward_indexBC]))
    plt.savefig('Figures/FiguresBC/BC_from_human_Evaluation.eps', format='eps')
    plt.show() 
        
    BCAve = np.mean(BC)
    STD = np.std(BC)
    MAXBC = np.max(maxBC)
    MINBC = np.min(minBC)    
    
    print('BC average reward = {}'.format(BCAve))
    print('BC standard deviation = {}'.format(STD))
    print('BC best trajectory reward = {}'.format(MAXBC))
    print('BC worst trajectory reward = {}'.format(MINBC))
    
def HIL_random_init_plot_and_stats(HIL_evaluation_random_init,coins_location):
    averageHIL_random_init = []
    HILPI = []
    maxHIL_random_init = []
    minHIL_random_init = []
    
    for i in range(len(HIL_evaluation_random_init)):
        averageHIL_random_init.append(np.mean(HIL_evaluation_random_init[i][0]))
        HILPI.append(HIL_evaluation_random_init[i][0])
        maxHIL_random_init.append(np.max(HIL_evaluation_random_init[i][0]))
        minHIL_random_init.append(np.min(HIL_evaluation_random_init[i][0]))
    best_index_agentHIL_random_init = np.argmax(maxHIL_random_init)
    best_reward_indexHIL_random_init=np.argmax(HIL_evaluation_random_init[best_index_agentHIL_random_init][0])
    best_episodeHIL_random_init=HIL_evaluation_random_init[best_index_agentHIL_random_init][1][best_reward_indexHIL_random_init]
    
    time = np.linspace(0,480,len(best_episodeHIL_random_init[:,0]))  
    # Plot result
    sigma1 = 0.5
    circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
    sigma2 = 1.1
    circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
    sigma3 = 1.8
    circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
    sigma4 = 1.3
    circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
    fig, ax2 = plt.subplots()
    plot_data = plt.scatter(best_episodeHIL_random_init[:,0], best_episodeHIL_random_init[:,1], c=HIL_evaluation_random_init[best_index_agentHIL_random_init][2][best_reward_indexHIL_random_init][:], marker='o', cmap='bwr')
    #plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
    cbar = fig.colorbar(plot_data, ticks=[0, 1])
    cbar.ax.set_yticklabels(['option 1', 'option 2'])
    ax2.add_artist(circle1)
    ax2.add_artist(circle2)
    ax2.add_artist(circle3)
    ax2.add_artist(circle4)
    plt.xlim([-10, 10])
    plt.ylim([-10, 11])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('HIL agent random init, reward {}'.format(HIL_evaluation_random_init[best_index_agentHIL_random_init][0][best_reward_indexHIL_random_init]))
    plt.savefig('Figures/FiguresBatch/HIL_Traj_VS_Options_traj_new_random_init_reward{}.eps'.format(HIL_evaluation_random_init[best_index_agentHIL_random_init][0][best_reward_indexHIL_random_init]), format='eps')
    plt.show()  
    
    sigma1 = 0.5
    circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
    sigma2 = 1.1
    circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
    sigma3 = 1.8
    circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
    sigma4 = 1.3
    circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
    fig, ax2 = plt.subplots()
    plot_data = plt.scatter(best_episodeHIL_random_init[:,0], best_episodeHIL_random_init[:,1], c=time, marker='o', cmap='cool')
    plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
    cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
    cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
    ax2.add_artist(circle1)
    ax2.add_artist(circle2)
    ax2.add_artist(circle3)
    ax2.add_artist(circle4)
    plt.xlim([-10, 10])
    plt.ylim([-10, 11])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('HIL agent random init, reward {}'.format(HIL_evaluation_random_init[best_index_agentHIL_random_init][0][best_reward_indexHIL_random_init]))
    plt.savefig('Figures/FiguresBatch/HIL_Traj_VS_Time_traj_new_random_init_reward{}.eps'.format(HIL_evaluation_random_init[best_index_agentHIL_random_init][0][best_reward_indexHIL_random_init]), format='eps')
    plt.show()  
        
    HILPIAve = np.mean(HILPI)
    HILPIStd = np.std(HILPI)
    MAXHIL_random_init = np.max(maxHIL_random_init)
    MINHIL_random_init = np.min(minHIL_random_init)

    print('HIL random init average reward = {}'.format(HILPIAve))
    print('HIL random init standard deviation = {}'.format(HILPIStd))
    print('HIL random init best trajectory reward = {}'.format(MAXHIL_random_init))
    print('HIL random init worst trajectory reward = {}'.format(MINHIL_random_init))    
    
def HIL_pre_init_plot_and_stats(HIL_evaluation_pre_init,coins_location):    
    averageHIL = []
    HIL = []
    max_HIL = []
    min_HIL = []
    
    for i in range(len(HIL_evaluation_pre_init)):
        averageHIL.append(np.mean(HIL_evaluation_pre_init[i][0]))
        HIL.append(HIL_evaluation_pre_init[i][0])
        max_HIL.append(np.max(HIL_evaluation_pre_init[i][0]))
        min_HIL.append(np.min(HIL_evaluation_pre_init[i][0]))
    best_index_agentHIL = np.argmax(max_HIL)
    best_reward_indexHIL=np.argmax(HIL_evaluation_pre_init[best_index_agentHIL][0])
    best_episodeHIL=HIL_evaluation_pre_init[best_index_agentHIL][1][best_reward_indexHIL]
    
    time = np.linspace(0,480,len(best_episodeHIL[:,0]))  
    # Plot result
    sigma1 = 0.5
    circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
    sigma2 = 1.1
    circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
    sigma3 = 1.8
    circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
    sigma4 = 1.3
    circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
    fig, ax2 = plt.subplots()
    plot_data = plt.scatter(best_episodeHIL[:,0], best_episodeHIL[:,1], c=HIL_evaluation_pre_init[best_index_agentHIL][2][best_reward_indexHIL][:], marker='o', cmap='bwr')
    #plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
    cbar = fig.colorbar(plot_data, ticks=[0, 1])
    cbar.ax.set_yticklabels(['option 1', 'option 2'])
    ax2.add_artist(circle1)
    ax2.add_artist(circle2)
    ax2.add_artist(circle3)
    ax2.add_artist(circle4)
    plt.xlim([-10, 10])
    plt.ylim([-10, 11])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('HIL agent, reward {}'.format(HIL_evaluation_pre_init[best_index_agentHIL][0][best_reward_indexHIL]))
    plt.savefig('Figures/FiguresBatch/HIL_Traj_VS_Options_traj_reward{}.eps'.format(HIL_evaluation_pre_init[best_index_agentHIL][0][best_reward_indexHIL]), format='eps')
    plt.show()  
    
    sigma1 = 0.5
    circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
    sigma2 = 1.1
    circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
    sigma3 = 1.8
    circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
    sigma4 = 1.3
    circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
    fig, ax2 = plt.subplots()
    plot_data = plt.scatter(best_episodeHIL[:,0], best_episodeHIL[:,1], c=time, marker='o', cmap='cool')
    plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
    cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
    cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
    ax2.add_artist(circle1)
    ax2.add_artist(circle2)
    ax2.add_artist(circle3)
    ax2.add_artist(circle4)
    plt.xlim([-10, 10])
    plt.ylim([-10, 11])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('HIL agent, reward {}'.format(HIL_evaluation_pre_init[best_index_agentHIL][0][best_reward_indexHIL]))
    plt.savefig('Figures/FiguresBatch/HIL_Traj_VS_Time_traj_reward{}.eps'.format(HIL_evaluation_pre_init[best_index_agentHIL][0][best_reward_indexHIL]), format='eps')
    plt.show()  
    
    HILAve = np.mean(HIL)
    HILSTD = np.std(HIL)
    MAXHIL_pre_init = np.max(max_HIL)
    MINHIL_pre_init = np.min(min_HIL)
    
    print('HIL pre-init average reward = {}'.format(HILAve))
    print('HIL pre-init standard deviation = {}'.format(HILSTD))
    print('HIL pre-init best trajectory reward = {}'.format(MAXHIL_pre_init))
    print('HIL pre-init worst trajectory reward = {}'.format(MINHIL_pre_init))      
    
def DOC_plot_and_stats(Training_DOC, DOC_evaluation, coins_location):
    average_reward = []
    OC = []
    maxOC = []
    minOC = []
    for i in range(len(DOC_evaluation)):
        average_reward.append(np.mean(DOC_evaluation[i][0]))
        OC.append(DOC_evaluation[i][0])
        maxOC.append(np.max(DOC_evaluation[i][0]))
        minOC.append(np.min(DOC_evaluation[i][0]))
    best_index = np.argmax(maxOC)
    best_traj_index = np.argmax(DOC_evaluation[best_index][0])
        
    # Plot result
    sigma1 = 0.5
    circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
    sigma2 = 1.1
    circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
    sigma3 = 1.8
    circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
    sigma4 = 1.3
    circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
    fig, ax2 = plt.subplots()
    plot_data = plt.scatter(DOC_evaluation[best_index][1][best_traj_index][:,0], DOC_evaluation[best_index][1][best_traj_index][:,1], c=DOC_evaluation[best_index][2][best_traj_index][:], marker='o', cmap='bwr')
    #plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
    cbar = fig.colorbar(plot_data, ticks=[0, 1])
    cbar.ax.set_yticklabels(['option 1', 'option 2'])
    ax2.add_artist(circle1)
    ax2.add_artist(circle2)
    ax2.add_artist(circle3)
    ax2.add_artist(circle4)
    # plt.xlim([-10, 10])
    # plt.ylim([-10, 10])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('OC agent, reward {}'.format(DOC_evaluation[best_index][3][best_traj_index]))
    plt.savefig('Figures/FiguresOC/OC_Traj_VS_Options_traj_reward{}.eps'.format(DOC_evaluation[best_index][3][best_traj_index]), format='eps')
    plt.show()  
    
    time = np.linspace(0,480,len(DOC_evaluation[best_index][1][best_traj_index][:,0]))  
    # Plot result
    sigma1 = 0.5
    circle1 = plt.Circle((6, 7.5), 2*sigma1, color='k',  fill=False)
    sigma2 = 1.1
    circle2 = plt.Circle((-1.5, -5), 2*sigma2, color='k',  fill=False)
    sigma3 = 1.8
    circle3 = plt.Circle((-5, 3), 2*sigma3, color='k',  fill=False)
    sigma4 = 1.3
    circle4 = plt.Circle((4.9, -4), 2*sigma4, color='k',  fill=False)
    fig, ax2 = plt.subplots()
    plot_data = plt.scatter(DOC_evaluation[best_index][1][best_traj_index][:,0], DOC_evaluation[best_index][1][best_traj_index][:,1], c=time, marker='o', cmap='cool')
    plt.plot(0.1*coins_location[:,0], 0.1*coins_location[:,1], 'xk')
    cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 400, 500])
    cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 400', 'time = 500'])
    ax2.add_artist(circle1)
    ax2.add_artist(circle2)
    ax2.add_artist(circle3)
    ax2.add_artist(circle4)
    # plt.xlim([-10, 10])
    # plt.ylim([-10, 10])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('OC agent, reward {}'.format(DOC_evaluation[best_index][0][best_traj_index]))
    plt.savefig('Figures/FiguresOC/OC_Traj_VS_Time_traj_reward{}.eps'.format(DOC_evaluation[best_index][0][best_traj_index]), format='eps')
    plt.show()  
    
    episodes = np.arange(0,len(Training_DOC))
    plt.plot(episodes, Training_DOC, 'g', label = 'OC agent')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training OC')
    plt.legend()
    plt.savefig('Figures/FiguresOC/OC_training_trend_unsmoothed.eps', format='eps')
    plt.show() 
    
    episodes = np.arange(0,len(Training_DOC))
    z = np.polyfit(episodes, Training_DOC,10)
    p = np.poly1d(z)
    plt.plot(episodes, p(episodes), 'g', label = 'OC agent')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training OC')
    plt.legend()
    plt.savefig('Figures/FiguresOC/OC_training_trend_smoothed.eps', format='eps')
    plt.show() 
        
    OCAve = np.mean(OC)
    STDOC = np.std(OC)
    MAXOC= np.max(maxOC)
    MINOC = np.min(minOC)
    
    print('DOC average reward = {}'.format(OCAve))
    print('DOC standard deviation = {}'.format(STDOC))
    print('DOC best trajectory reward = {}'.format(MAXOC))
    print('DOC worst trajectory reward = {}'.format(MINOC))        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    