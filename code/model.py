#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 23:31:08 2020

@author: deepakravi
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

import pandas as pd


import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

import random
import urllib
import json
import time
from tqdm import tqdm


#%% BRING IN DATA

# open the master df
master_df_2 = pd.read_pickle('../data/' + 'master_df.pickle')
master_df = master_df_2.copy()
duplicates = master_df_2['GEOID'].duplicated('first')
non_duplicate_indexes = list(master_df_2[~duplicates].index)

master_df = master_df_2[~duplicates]

master_df = pd.read_csv('../data/master_df_clean.csv')

#open OD
subway_OD =   np.load('../data/' + 'subway_OD.npy')
adjacent_OD = np.load('../data/' + 'adjacent_OD.npy')
noise_OD = np.load('../data/' + 'noise_OD.npy')


subway_OD_2 = np.take(a = subway_OD, axis = 0, indices =  non_duplicate_indexes)
subway_OD_2 = np.take(a = subway_OD_2, axis = 1, indices = non_duplicate_indexes)



subway_OD = subway_OD_2
noise_OD = noise_OD * .04

full_OD = subway_OD + adjacent_OD + noise_OD
#to normalize the flow over time, let's half and equate the input/output
half_OD = full_OD * .5
OD = half_OD + half_OD.transpose()
count_tracts = len(full_OD)

#clean up some discrepancies
master_df['ct_first_positives'].loc[master_df['E_TOTPOP'] < master_df['ct_first_positives']] = 0

indices = list(master_df[master_df['E_TOTPOP'] == 0].index)
np_idx = np.array([indices for x in range(count_tracts)])
np_idxt = np_idx.transpose()



prune_OD = np.ones(shape = (count_tracts,count_tracts), dtype=float)
np.put_along_axis(prune_OD, np_idxt , 0.0, axis = 0)
np.put_along_axis(prune_OD, np_idx , 0.0, axis = 1)

OD = OD*prune_OD



#%% PREP MODEL

# initialize the population vector from the origin-destination flow matrix
N_k_check = np.abs(np.diagonal(OD) + OD.sum(axis=0) - OD.sum(axis=1))


N_k = master_df['E_TOTPOP'].to_numpy() #Population vector


#vector of total population
master_df['E_TOTPOP'].to_numpy()

#number of locations --> number of nodes
count_tracts = len(master_df)


locs_len = len(N_k)                 # number of locations
SIR = np.zeros(shape=(locs_len, 3)) # make a numpy array with 3 columns for keeping track of the S, I, R groups
SIR[:,0] = N_k                      # initialize the S group with the respective populations

#first_infections = np.where(SIR[:, 0]<=thresh, SIR[:, 0]//20, 0)   # for demo purposes, randomly introduce infections
first_infections = master_df['ct_first_positives'].to_numpy()

SIR[:, 0] = SIR[:, 0] - first_infections
SIR[:, 1] = SIR[:, 1] + first_infections                           # move infections to the I group

# row normalize the SIR matrix for keeping track of group proportions
row_sums = SIR.sum(axis=1)
SIR_n = SIR / row_sums[:, np.newaxis]
SIR_n = np.nan_to_num(SIR_n)


# initialize parameters
beta = 1.6
gamma = 0.04
public_trans = 0.5                                 # alpha
R0 = beta/gamma

#beta_vec = np.random.gamma(1.6, 2, locs_len) #Original
gamma_vec = np.full(locs_len, gamma) 
beta_vec = master_df['random_R0'].to_numpy()*gamma_vec #Modified to allocate higher R0 to most affected neighborhoods


public_trans_vec = np.full(locs_len, public_trans)

# make copy of the SIR matrices 
SIR_sim = SIR.copy()
SIR_nsim = SIR_n.copy()


#%% RUN MODEL
#FAST FIX
N_k = np.where(N_k==0, 1, N_k) 

print(SIR_sim.sum(axis=0).sum() == N_k.sum())
from tqdm import tqdm_notebook
infected_pop_norm = []
susceptible_pop_norm = []
recovered_pop_norm = []

time_step = 0
#Infected matrix is the proportion being infected based on the original infected numbers
infected_mat = np.array([SIR_nsim[:,1],]*locs_len).transpose()
#multiply the infected proportion to the OD matrix to see how many infected are flowing
OD_infected = np.round(OD*infected_mat)
#sum the infected flowing to each node
inflow_infected = OD_infected.sum(axis=0)
#multiply by some scaling of flow
inflow_infected = np.round(inflow_infected*public_trans_vec)
print('total infected inflow: ', inflow_infected.sum())

#find new infected by mulpitplyign the infected_inflow by tranmission beta * succeptible * proportion of the (population+inflow) that are infected
new_infect = beta_vec*SIR_sim[:, 0]*inflow_infected/(N_k + OD.sum(axis=0))
#num recovred = some proportion of the infected
new_recovered = gamma_vec*SIR_sim[:, 1]
#new infections set to total population of node if calculated is greater than existing
new_infect = np.where(new_infect>SIR_sim[:, 0], SIR_sim[:, 0], new_infect)
#remove infected from succesptible
SIR_sim[:, 0] = SIR_sim[:, 0] - new_infect
#add new infected and remove recovered
SIR_sim[:, 1] = SIR_sim[:, 1] + new_infect - new_recovered
#add recovered
SIR_sim[:, 2] = SIR_sim[:, 2] + new_recovered
#set any negatives to 0
SIR_sim = np.where(SIR_sim<0,0,SIR_sim)
# recompute the normalized SIR matrix
#should be equal to the populations of each
row_sums = SIR_sim.sum(axis=1)
#normalized values
SIR_nsim = SIR_sim / row_sums[:, np.newaxis]
S = SIR_sim[:,0].sum()/N_k.sum()
I = SIR_sim[:,1].sum()/N_k.sum()
R = SIR_sim[:,2].sum()/N_k.sum()
print(S, I, R, (S+I+R)*N_k.sum(), N_k.sum())
print('\n')
infected_pop_norm.append(I)
susceptible_pop_norm.append(S)
recovered_pop_norm.append(R)


#%%


for time_step in tqdm(range(100)):
    infected_mat = np.array([SIR_nsim[:,1],]*locs_len).transpose()
    OD_infected = np.round(OD*infected_mat)
    inflow_infected = OD_infected.sum(axis=0)
    inflow_infected = np.round(inflow_infected*public_trans_vec)
    print('total infected inflow: ', inflow_infected.sum())
    new_infect = beta_vec*SIR_sim[:, 0]*inflow_infected/(N_k + OD.sum(axis=0))
    new_recovered = gamma_vec*SIR_sim[:, 1]
    new_infect = np.where(new_infect>SIR_sim[:, 0], SIR_sim[:, 0], new_infect)
    SIR_sim[:, 0] = SIR_sim[:, 0] - new_infect
    SIR_sim[:, 1] = SIR_sim[:, 1] + new_infect - new_recovered
    SIR_sim[:, 2] = SIR_sim[:, 2] + new_recovered
    SIR_sim = np.where(SIR_sim<0,0,SIR_sim)
    # recompute the normalized SIR matrix
    row_sums = SIR_sim.sum(axis=1)
    SIR_nsim = SIR_sim / row_sums[:, np.newaxis]
    S = SIR_sim[:,0].sum()/N_k.sum()
    I = SIR_sim[:,1].sum()/N_k.sum()
    R = SIR_sim[:,2].sum()/N_k.sum()
    print(S, I, R, (S+I+R)*N_k.sum(), N_k.sum())
    print('\n')
    infected_pop_norm.append(I)
    susceptible_pop_norm.append(S)
    recovered_pop_norm.append(R)
    
    
#%% WRITE RESULTS DF






    
    
    
    
    
    
    
    
    