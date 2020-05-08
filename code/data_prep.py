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

import numpy as np
import random
import urllib
import json
import time
from tqdm import tqdm


#%% Load NYC Case Data

data_dir = '../../covid-19-nyc-data/'

def read_and_date(path):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['timestamp'])
    return df

#age_df = read_and_date(data_dir + 'age.csv')
#bed_df = read_and_date(data_dir + 'beds.csv')
zc_tests_df = read_and_date(data_dir + 'zcta.csv')
#tot_df = read_and_date(data_dir + 'nyc.csv')
#hosp_df = read_and_date(data_dir + 'hospitalized.csv')
#gender_df = read_and_date(data_dir + 'gender.csv')
#borough_df = read_and_date(data_dir + 'borough.csv')
#state_df = read_and_date(data_dir + 'state.csv')



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
'''
turns = pd.read_csv(data_dir +'mta_turnstile_200418.csv')
turns['ENTRIES'] = turns['ENTRIES'].astype('int')
turns.rename(columns={'EXITS                                                               ':'EXITS'}, inplace=True)
turns['EXITS'] = turns['EXITS'].astype('int')
                                                      
#step 1 subway inflows / outflows from each station
turns['ENTRIES'] = turns.groupby(['C/A','STATION','UNIT','SCP'])['ENTRIES'].diff()
turns['EXITS'] = turns.groupby(['C/A','STATION','UNIT','SCP'])['EXITS'].diff()
turns = turns.groupby(['STATION','DATE', 'LINENAME']).agg('sum')
turns = turns.groupby(['STATION', 'LINENAME']).agg('mean').sort_values(by = ['STATION'])


#step 2 join stations with neighborhoods by long/lat
entrances = pd.read_csv(data_dir +'DOITT_SUBWAY_STATION_01_13SEPT2010.csv')
entrances['STATION'] = entrances['NAME'].str.upper()
entrances['X'] = entrances['the_geom'].str.split(' ').str[1].str[1:].astype('double')
entrances['Y'] = entrances['the_geom'].str.split(' ').str[2].str[:-1].astype('double')
#entrances = entrances.groupby('Station Name').first()
entrances = entrances[['STATION','X','Y','LINE']].sort_values(by = ['STATION'])
'''
'''
data_dir = '../data/'
joined = pd.read_csv(data_dir +'station_locations_joined.csv')
geo = pd.read_csv(data_dir +'station_location_final.csv')
'''

#stations = joined.merge(geo[['STATION','LINE','GEOID']], on = ['STATION','LINE'], how  = 'left')
data_dir = '../data/'
stations = pd.read_csv(data_dir +'stations_final.csv')
#stations.to_csv('../data/stations_final.csv')



#entrances.to_csv('../data/station_location.csv')
#turns.to_csv('../data/turnstile_counts.csv')

#>>>>FAIL
#import d6tjoin.top1
#import d6tjoin.utils
#d6tjoin.utils.PreJoin([entrances,turns],['STATION']).stats_prejoin()
#entrances['STATION'] = entrances.index
#turns['STATION1'] = turns.index
#result = d6tjoin.top1.MergeTop1(entrances,turns,fuzzy_left_on=['STATION1'],fuzzy_right_on=['STATION1']).merge()
#<<<<


import urllib
import json
import time
from tqdm import tqdm


def get_GEOID(X,Y):
    url = ('https://geocoding.geo.census.gov/geocoder/geographies/coordinates?x='+
           str(X) + '&y='+ str(Y) + '&benchmark=4&vintage=4&format=json')
    response = urllib.request.urlopen(url)
    
    try:
        res_body = response.read()
        j = json.loads(res_body.decode("utf-8"))
        geoid = int(j['result']['geographies']['Census Tracts'][0]['GEOID'])
    except:
        geoid = 'ERROR'
    
    return geoid


dicts = []
'''
for index, row in tqdm(entrances.iterrows()):
    
    #time.sleep(1)
    geoid = get_GEOID(row['X'], row['Y'])
    di = {'X':row['X'], 'Y':row['Y'], 'GEOID':geoid}
    dicts.append(di)
    
geoids = pd.DataFrame(dicts)

entrances = entrances.merge(geoids, on = ['X', 'Y'])
'''





#%% Create Master DF
#Aggregate the Census Tract data by ZCTA and then join columns as needed
census_zc_subset = master_df[['zcta', 'POPPT']].groupby('zcta').agg('sum')
zcta_tests_df = zc_tests_df.merge(census_zc_subset, on = 'zcta', how = 'left')
zcta_tests_df = zcta_tests_df[zcta_tests_df['zcta'] < 12000] #filter out NaN data


zcta_tests_df['percent_positive'] = zcta_tests_df['positive']/zcta_tests_df['total']*100
zcta_tests_df['percent_tested'] = zcta_tests_df['total']/zcta_tests_df['POPPT']*100

zcta_tests_df['new_positive'] = zcta_tests_df.groupby('zcta')['positive'].diff()
zcta_tests_df['new_test'] = zcta_tests_df.groupby('zcta')['total'].diff()

zcta_tests_df['percent_new_positive'] = zcta_tests_df['new_positive']/zcta_tests_df['new_test']*100
zcta_tests_df['percent_new_tested'] = zcta_tests_df['new_test']/zcta_tests_df['POPPT']*100

tests_grouped = zcta_tests_df.groupby('zcta').agg({'positive':'first', 'percent_new_positive':'mean', 'new_positive':'mean'})


#organize_census tracts by ascending order, give index
master_df_2 = master_df[['zcta', 'census_tract', 'GEOID', 'E_TOTPOP', 'LOCATION']] 

#left join zcta
temp_master = master_df_2[['GEOID','zcta']].merge(tests_grouped, how = 'left', on = 'zcta')

#cases data groupy zcta total / count of instances #divide out the multiple cts per zctasmaster
temp_master = temp_master.groupby('zcta').agg({'positive':'sum', 'GEOID':'count', 'percent_new_positive':'mean', 'new_positive':'sum'}) 
temp_master.rename(columns={'GEOID':'GEOID_count'}, inplace=True)
#sum / count = individual ct values
temp_master['ct_first_positives'] = temp_master['positive']/temp_master['GEOID_count']
temp_master['ct_avg_new_positives'] = temp_master['new_positive']/temp_master['GEOID_count']
temp_master['ct_avg_new_positives_percent'] = temp_master['percent_new_positive']

#rejoin left with master_df
temp_master = master_df_2[['GEOID','zcta']].merge(temp_master, how = 'left', on = 'zcta')

#group by ct for sum rates
temp_master = temp_master.groupby('GEOID').agg({'ct_avg_new_positives':'mean',
                                                'ct_avg_new_positives_percent':'mean',
                                                'ct_first_positives':'sum'})

#join this with the main master
master_df_2 = master_df_2.merge(temp_master, how = 'left', on='GEOID')


#master_df_2 = master_df_2.drop_duplicates('GEOID') #There duplkicates becuase of zcta...
master_df_2['old_index'] = master_df_2.index
master_df_2 = master_df_2.sort_values(by = ['GEOID']).reset_index(drop = True)
master_df_2['numpy_index'] = master_df_2.index

#join input/output only subway for each community, 0 if no station
#divide by total input/output in the system to get distribution

master_df_2 = master_df_2.merge(stations[['t_station','t_entries','t_exits','GEOID']], how = 'left', on='GEOID')

#fill nans with 0s
master_df_2[['t_entries', 't_exits']] = master_df_2[['t_entries', 't_exits']].fillna(value=0)
master_df_2.rename(columns={'t_entries':'sub_output', 't_exits':'sub_input'}, inplace=True)#entries into the subway are out of the neighborhood
master_df_2['sub_input_prop'] = master_df_2['sub_input']/master_df_2['sub_input'].sum()#proportion of total subwy inputs
master_df_2['sub_output_prop'] = master_df_2['sub_output']/master_df_2['sub_output'].sum()#proportion of total subway output


import geopandas as gp

file= "../raw_data/cb_2018_36_tract_500k/cb_2018_36_tract_500k.shp"    

shape_df = gp.read_file(file) # open file

#filter_counties
shape_df = shape_df[shape_df['COUNTYFP'].astype('int').isin([5,81,61,85,47])]
shape_df["NEIGHBORS"] = None  # add NEIGHBORS column
for index, row in tqdm(shape_df.iterrows()):   
    # get 'not disjoint' countries
    neighbors = shape_df[shape_df.geometry.touches(row['geometry'])].GEOID.tolist()
    # remove own name from the list
    neighbors = [ GEOID for GEOID in neighbors if row.GEOID != GEOID]
    # add names of neighbors as NEIGHBORS value
    shape_df.at[index, "NEIGHBORS"] = ", ".join(neighbors)
    
shape_df['GEOID'] = shape_df['GEOID'].astype('int')
master_df_2 = master_df_2.merge(shape_df[['geometry','GEOID','NEIGHBORS']], on = 'GEOID', how = 'left')

master_df_2 = master_df_2.drop_duplicates('GEOID')

master_df_2.to_pickle('master_df.pickle')

#%% Orgin Destination Matrix

#step 3 construct ct x ct matrix 

import numpy as np
#build Origin-Destination Matrix where indexes correspond to the above census index
count_tracts = 2157

#step 4 randomly distribute subway inflow and outflow
subway_OD = np.empty(shape = (count_tracts,count_tracts), dtype=float)
for origin in range(count_tracts):
    
    #assign output*sub_input_prop to get a vector to replace row
    output = master_df_2.iloc[origin]['sub_output']
    proportional_input = master_df_2['sub_input_prop']*output #Outflow into tother inflow
    new_row = proportional_input.to_numpy()
    
    #set row
    subway_OD[origin,:] = new_row
    
#halve the subway data becuase we are going to do twice a day
#subway_OD = subway_OD*.5

#%% 
import numpy as np
import random
count_tracts = len(master_df_2)

#step 5 distribute additional minor inflow outflow into adjacent neighborhoods
adjacent_OD = np.empty(shape = (count_tracts,count_tracts), dtype=float)       
adjacency = np.empty(shape = (count_tracts,count_tracts), dtype=float) 
    

for origin in range(len(master_df_2)):
    for destination in range(len(master_df_2)):
        
        neighbors = master_df_2.iloc[origin]['NEIGHBORS']
        dest_GEOID = master_df_2.iloc[destination]['GEOID']
        neighbors_yes_no = str(dest_GEOID) in neighbors
        
        if neighbors_yes_no:
            adjacency[origin,destination] = 1
            
            #random value between 10 and 20 percent 
            p_multiplier = random.uniform(.1, .2)
            o_pop = master_df_2.iloc[origin]['E_TOTPOP']
            d_pop = master_df_2.iloc[destination]['E_TOTPOP']
            
            adjacent_OD[origin,destination] = np.mean([o_pop,d_pop])*p_multiplier
            
        else:
            adjacency[origin,destination] = 0
            adjacent_OD[origin,destination] = 0
        


#%%
count_tracts = 2157

noise_OD = np.empty(shape = (count_tracts,count_tracts), dtype=float)
                   
#get aggregate output, assign valu to each element in row 
def randomize(noise_OD, master_df):
    #random boi
    #randomly picks two tracts
    #applies random muliper to avg population
    #sets flow

    #in future could weight by the distance btwn neighborhoods
        
    #count_tracts = len(master_df)
        
    origin = int(random.randint(0,count_tracts-1))
    destination = int(random.randint(0,count_tracts-1))
        
        
    o_pop = master_df.iloc[origin]['E_TOTPOP']
    d_pop = master_df.iloc[destination]['E_TOTPOP']
        
    #generate random value within 10 to 40% of the input / output of some pair
    avg_pop = np.mean([o_pop,d_pop])
    mult = random.uniform(.05, .1)
    d = avg_pop*mult
    
    #print(origin)
    #print(destination)
        
    noise_OD[origin, destination] = noise_OD[origin, destination] + d
    noise_OD[destination, origin] = noise_OD[origin, destination] - d

        
    #noise_OD[x1][y1] += d
    #noise_OD[x1][y2] -= d
    #noise_OD[x2][y1] -= d
    #noise_OD[x2][y2] += d
    
    return
    
for i in tqdm(range(300000)):
    randomize(noise_OD, master_df_2)
        
#%% Components of OD

#add all components of the flow
full_OD = subway_OD + adjacent_OD + noise_OD

#to normalize the flow over time, let's half and equate the input/output
half_OD = full_OD * .5
OD = half_OD + half_OD.transpose()


#our_goal = ct x ct matrix of inflow and outflow

#inflow / outflow as a spatial measure between neighborhoods some spatial function of long and lat
#inflow outflow from subway line evenly/randomly distrbuted along all subway neghiborhoods


#%% JOIN CASE DATA

#df = pd.Series(np.random.gamma(2, 4, len(master_df_2)), name = 'random_R0').to_frame()
df = pd.Series(np.random.uniform(2, 4, len(master_df_2)), name = 'random_R0').to_frame()
df = df.sort_values(['random_R0'])
short_df = master_df[['GEOID','ct_avg_new_positives']].sort_values(['ct_avg_new_positives'])
short_df = pd.concat([short_df, df], axis = 1, ignore_index = True)
short_df.columns = ['GEOID', 'ct_avg_new_positives', 'random_R0']

master_df = master_df.merge(short_df, on = ['GEOID', 'ct_avg_new_positives'])


#%% Store necessary tables








