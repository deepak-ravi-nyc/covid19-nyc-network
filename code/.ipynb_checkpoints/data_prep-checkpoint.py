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
census_df = census_df[census_df['COUNTY'].isin(['Kings','Queens','Bronx','New York','Richmond'])] 



#%% Load NTA-CTA-ZCTA-ZC Relationships
data_dir = '../raw_data/'

ct_to_zcta_df = pd.read_csv(data_dir + 'zcta_tract_rel_10.csv')
#Select NYC counties
ct_to_zcta_df = ct_to_zcta_df[(ct_to_zcta_df['STATE'] == 36) & ct_to_zcta_df['COUNTY'].isin([5,81,61,85,47])]

'''
nta_to_ct_df = pd.read_excel(data_dir + 'nyc2010census_tabulation_equiv.xlsx', index_col=None, header=3)
zcta_to_zipcode = pd.read_excel(data_dir + 'zip_to_zcta_2019.xlsx', index_col=None, header=0)
zcta_to_zipcode = zcta_to_zipcode[zcta_to_zipcode['STATE']=='NY']
'''

# Merge zcta_tests to census data to nta and zcta and census_tracts
# Create a backbone, merge the ct/zcta/nta

ct_to_zcta_df.rename(columns={'ZCTA5':'zcta'}, inplace=True)
ct_to_zcta_df.rename(columns={'TRACT':'census_tract'}, inplace=True)

'''
nta_to_ct_df.rename(columns={'2010 Census Tract':'census_tract'}, inplace=True)
nta_to_ct_df.rename(columns={'Neighborhood Tabulation Area (NTA)':'nta'}, inplace=True)

zcta_to_zipcode.rename(columns={'ZIP_CODE':'zip_code'}, inplace=True)
zcta_to_zipcode.rename(columns={'ZCTA':'zcta'}, inplace=True)
zcta_to_zipcode['zcta'] = zcta_to_zipcode['zcta'].astype('int')

backbone_df = nta_to_ct_df.merge(ct_to_zcta_df, on = "census_tract", how = 'inner')
backbone_df = backbone_df.merge(zcta_to_zipcode, on = "zcta", how = 'inner')

relationships_df = backbone_df[['zcta','nta','census_tract','zip_code','Borough', 'COUNTY']]
'''

# Tabling this for now, until as needed 
# INSTEAD JUST join the census data to the case data

master_df = census_df.merge(ct_to_zcta_df, left_on = 'FIPS', right_on = 'GEOID')

#Before we merge this we need to find avg rate? base measures, population needs to be joined

#Aggregate the Census Tract data by ZCTA and then join columns as needed
census_zc_subset = master_df[['zcta', 'POPPT']].groupby('zcta').agg('sum')
zcta_tests_df = zc_tests_df.merge(census_zc_subset, on = 'zcta', how = 'left')


#%% Load Mobility Data






#%% Store necessary tables








