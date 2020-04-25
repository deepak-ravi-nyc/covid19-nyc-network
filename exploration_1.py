#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:07:26 2020

@author: deepakravi
"""

import pandas as pd


import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

#%% LOAD DATA

data_dir = '../covid-19-nyc-data/'

def read_and_date(path):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['timestamp'])
    return df

age_df = read_and_date(data_dir + 'age.csv')
bed_df = read_and_date(data_dir + 'beds.csv')
zc_tests_df = read_and_date(data_dir + 'zcta.csv')
tot_df = read_and_date(data_dir + 'nyc.csv')
hosp_df = read_and_date(data_dir + 'hospitalized.csv')
gender_df = read_and_date(data_dir + 'gender.csv')
borough_df = read_and_date(data_dir + 'borough.csv')
state_df = read_and_date(data_dir + 'state.csv')



#%% PLOTTING FUNCTION
def plot_by_date(df, title, cols):  
  traces = []
  for col in cols: #pdf['athlete_id'].unique():
    #ac_focus_ac = ac_focus[ac_focus['activity'] == ac]
    trace = go.Scattergl(
                x = df['datetime'],
                y = df[col], 
                name = col,
                mode = 'lines',
                #marker = dict( opacity = .7)
                #opacity = .55,
                #marker = dict(opacity = .5,size=2)
                )
    traces.append(trace)

  layout = go.Layout(legend = dict(x = 1, y = 1), title = title , xaxis = dict( rangeslider = dict()), yaxis = dict())
  fig = go.Figure(data=traces,layout=layout)
  py.plot(fig, filename = title + '.html')
  
  return
  
age_deaths_df = age_df[age_df['type'] == 'deaths']
 
plot_by_date(df = age_deaths_df,
              title = 'deaths_comapred_by_age',
              cols = ['ages_0_17',
                      'ages_18_44',
                      'ages_45_64',
                      'ages_65_74',
                      'ages_75_older'])











