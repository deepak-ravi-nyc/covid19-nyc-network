#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 08:28:42 2020

@author: deepakravi
"""

import pandas as pd


import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff



#%% Load NYC Case Data

data_dir = '../../covid-19-nyc-data/'

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



#%% Load SVI Data

data_dir = '../raw_data/'
#SVI Index attributes
census_df = pd.read_csv(data_dir + 'svi_index_NewYork.csv')
#filter out NYC
census_df = census_df[census_df['COUNTY'].str.isin('Kings','Queens','Bronx','New York','Richmond')] 



#%% Load NTA-CTA-ZCTA Relationships
data_dir = '../raw_data/'


ct_to_zcta_df = pd.read_csv(data_dir + 'zcta_tract_rel_10.csv')
nta_to_ct_df = pd.read_excel('nyc2010census_tabulation_equiv.xlsx', index_col=None, header=5)





#%% Load Mobility Data






#%% Create a master table






