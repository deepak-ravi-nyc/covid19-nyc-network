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



#%% NYC Case Data

#hosp_death_df = pd.read_csv('../../coronavirus-data/case-hosp-death.csv')
#mortality_rate = (hosp_death_df['DEATH_COUNT'].sum()/hosp_death_df['CASE_COUNT'].sum())
#mortality_rate = (hosp_death_df['DEATH_COUNT']/hosp_death_df['CASE_COUNT']).mean()
#mortality_rate = .05

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
SIR = np.zeros(shape=(locs_len, 4)) # make a numpy array with 3 columns for keeping track of the S, I, R groups
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
beta = 1.6 #transmission
gamma = 0.04 #recovery rate
recovery_rate = .08 #1 / num of days
public_trans = .5  #flow multplier
mortality_rate = .03 

                              # alpha
R0 = beta/gamma

#beta_vec = np.random.gamma(1.6, 2, locs_len) #Original
gamma_vec = np.full(locs_len, gamma) 
beta_vec = master_df['random_R0'].to_numpy()*gamma_vec #Modified to allocate higher R0 to most affected neighborhoods

recovery_rate_vec = np.full(locs_len, recovery_rate) 
public_trans_vec = np.full(locs_len, public_trans)

# make copy of the SIR matrices 
SIR_sim = SIR.copy()
SIR_nsim = SIR_n.copy()


#%% RUN MODEL

def d0(a,b):
    #https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
    c = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    return c
    

#FAST FIX May 8
N_k = np.where(N_k==0, 1, N_k) 

print(SIR_sim.sum(axis=0).sum() == N_k.sum())

infected_pop_norm = []
susceptible_pop_norm = []
recovered_pop_norm = []

SIR_dfs = []

time_step = 0
for time_step in tqdm(range(100)):
    #Infected matrix is the proportion being infected based on the original infected numbers
    infected_mat = np.array([SIR_nsim[:,1],]*locs_len).transpose()
    #multiply the infected proportion to the OD matrix to see how many infected are flowing
    OD_infected = np.round(OD*infected_mat)
    #sum the infected flowing to each node
    inflow_infected = OD_infected.sum(axis=0)
    #multiply by some scaling of flow
    inflow_infected = np.round(inflow_infected*public_trans_vec)
    #print('total infected inflow: ', inflow_infected.sum())
    
    #find new infected by mulpitplyign the infected_inflow by tranmission beta * succeptible * proportion of the (population+inflow) that are infected
    new_infect = beta_vec*SIR_sim[:, 0]*d0(inflow_infected,(N_k + OD.sum(axis=0))) #new_infect = beta_vec*SIR_sim[:, 0]*inflow_infected/(N_k + OD.sum(axis=0))
    
    
    #num recovred = some proportion of the infected
    new_recovered = recovery_rate_vec*SIR_sim[:, 1]
    #
    new_deaths = mortality_rate*SIR_sim[:,1]
    
    #This lags deaths, d=such that those infected for a certain time are suspected of death
    #if len(SIR_dfs) > mortality_lag:
    #    prev_infected = SIR_dfs[-mortality_lag][:,1]
    #    new_deaths = mortality_rate*prev_infected
    

    #new infections set to total population of node if calculated is greater than existing
    new_infect = np.where(new_infect>SIR_sim[:, 0], SIR_sim[:, 0], new_infect)
    #remove infected from succesptible
    SIR_sim[:, 0] = SIR_sim[:, 0] - new_infect
    #add new infected and remove recovered
    SIR_sim[:, 1] = SIR_sim[:, 1] + new_infect - new_recovered - new_deaths
    #add recovered
    SIR_sim[:, 2] = SIR_sim[:, 2] + new_recovered
    #add dead
    SIR_sim[:, 3] = SIR_sim[:, 3] + new_deaths
    #set any negatives to 0
    SIR_sim = np.where(SIR_sim<0,0,SIR_sim)
    # recompute the normalized SIR matrix
    #should be equal to the populations of each
    row_sums = SIR_sim.sum(axis=1)
    #normalized values
    SIR_nsim = d0(SIR_sim , row_sums[:, np.newaxis]) #SIR_nsim = SIR_sim / row_sums[:, np.newaxis]
    S = d0(SIR_sim[:,0].sum(),N_k.sum()) #S = SIR_sim[:,0].sum()/N_k.sum()
    I = d0(SIR_sim[:,1].sum(),N_k.sum()) #I = SIR_sim[:,1].sum()/N_k.sum()
    R = d0(SIR_sim[:,2].sum(),N_k.sum()) #R = SIR_sim[:,2].sum()/N_k.sum()
    D = d0(SIR_sim[:,3].sum(),N_k.sum())
    #print(S, I, R, (S+I+R)*N_k.sum(), N_k.sum())
    #print('\n')
    infected_pop_norm.append(I)
    susceptible_pop_norm.append(S)
    recovered_pop_norm.append(R)

    SIR_df = pd.DataFrame(data=SIR_sim, columns=["susceptible", "infected", "recovered", "dead"])
    #append num new infected
    SIR_df['new_infected'] = pd.Series(new_infect)
    SIR_df['new_recovered'] = pd.Series(new_recovered)
    SIR_df['new_deaths'] = pd.Series(new_deaths)
    
    startdate = "4/8/2020"
    enddate = pd.to_datetime(startdate) + pd.DateOffset(days=time_step)
    SIR_df['date'] = enddate
    #JOIN GEOIDs
    SIR_df['GEOID']= master_df['GEOID']
    
    SIR_dfs.append(SIR_df)
    
    
    
#append all the dataframes together
SIR_dfs = pd.concat(SIR_dfs)
    
    



    
#%% CREATE MAP
import json
import plotly.express as px

with open('../raw_data/tract.json') as response:
    tracts = json.load(response)


#df = pd.read_pickle("master_df.pickle")
                
#df = SIR_dfs[(SIR_dfs['date']=='2020-04-08T00:00:00.000000000') | 
#             (SIR_dfs['date']=='2020-04-20T00:00:00.000000000') ]
df = SIR_dfs.copy()
df['date_str'] = df['date'].astype('str')
df = df[df['date_str'].isin(list(df['date_str'].unique())[::7])]
                
import plotly.express as px

fig = px.choropleth_mapbox(df, geojson=tracts,
                           featureidkey = 'properties.GEOID',
                           locations='GEOID',
                           #locationmode='geojson_id',
                           color='infected',
                           color_continuous_scale="OrRd",
                           range_color=(0, 1000),
                           mapbox_style="carto-positron",
                           zoom=10, center = {"lat": 40.7128, "lon": -74.0060},
                           opacity=0.5,
                           labels={'infected':'Number Infected'},
                           animation_group = 'GEOID',
                           animation_frame = 'date_str',
                           title = 'Infected Cases in NYC'
                          )


fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
py.plot(fig, filename = 'infected' + '.html')

 
#%% PLOT COMPARISON

cum_death_cases_df = pd.read_csv('../../covid-19-nyc-data/nyc.csv')

cum_SIR_df = SIR_dfs.groupby('date').sum()
cum_SIR_df['cum_cases'] = cum_SIR_df['new_infected'].cumsum() #should only count the sum of new infections
cum_SIR_df['cum_dead'] = cum_SIR_df['new_deaths'].cumsum() #should only count the sum of new infections

#cum_SIR_df['cum_deaths'] = cum_SIR_df['dead'].cumsum()


def line_plot(title, labels, x_vals, y_vals):
    traces = []
    for idx in range(len(labels)):
        trace = go.Scattergl(
            x = x_vals[idx],
            y = y_vals[idx],
            mode = 'lines',
            name = labels[idx]
            )
        traces.append(trace)

    layout = go.Layout(
                legend = dict(x = 1, y = 1),
                title = title, 
                xaxis = dict(title = title),
                yaxis = dict()
                )

    fig = go.Figure(data=traces,layout=layout)
    py.plot(fig, filename = title + '.html')
    
    


line_plot(title = 'cumulative_comparison',
          labels = ['true_cases', 'true_deaths', 'sim_cases', 'sim_deaths'], 
          y_vals = [cum_death_cases_df['cases'], cum_death_cases_df['deaths'], cum_SIR_df['cum_cases'],  cum_SIR_df['cum_dead'] ],
          x_vals = [cum_death_cases_df['timestamp'],cum_death_cases_df['timestamp'], cum_SIR_df.index,  cum_SIR_df.index])





















    
    
    
    
    