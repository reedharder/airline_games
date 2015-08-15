# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 03:38:34 2015

@author: Reed
"""
import os
import numpy as np
import pandas as pd
from collections import Counter
from itertools import product
#REMOVE SLC AND RENO
#HEURISTIC RULES FOR 2 CARRIER CUTTOFF: 90% of MARKET
#convert market to per day
#TUEsday 9th skype 11: 00am or thursdat
# fRI 12 th 11:00 am

os.chdir("C:/Users/Reed/Desktop/vaze_competition_paper")

#OHNO SHEEEIT
carriers = ['SEA','PDX','SFO','SAN','LAX','LAS','PHX','OAK','ONT','SMF','SJC']
pairs =[sorted([pair[0],pair[1]]) for pair in product(carriers,carriers) if pair[0]!=pair[1] ]
txtpairs = list(set(["_".join(pair) for pair in pairs]))
route_demands = pd.read_csv('route_demand_Q1.csv')

def create_market(row):
    market = [row['ORIGIN'], row['DESTINATION']]
    market.sort()
    return "_".join(market)
def dash_indicate(row):
    if row['FIRST_OPERATING_CARRIER']=='--' or row['SECOND_OPERATING_CARRIER']=='--':
        return True
    else:
        return False
        '''
        truf= route_demands.apply(dash_indicate,1)
        route_demands[truf]['PASSENGERS'].sum()/route_demands['PASSENGERS'].sum()
         0.0081616088056469024 1% overall but may be locally important?
         '''

def discord_indicate(row):
    if row['NUM_FLIGHTS']==2 and row['SECOND_OPERATING_CARRIER']!=row['FIRST_OPERATING_CARRIER']:
        return True
    else:
        return False
        '''
        truf = route_demands.apply(discord_indicate, 1)
        route_demands[truf]['PASSENGERS'].sum()/route_demands['PASSENGERS'].sum()
         0.13201529306009127

        '''
    # carriers: count of total, count of passsengers, routes used for each carrier {'routes':{},'pax':0} 
records = {pair:{'mktPax':0,'mktPaxD':0,'segPax':0,'segPaxD':0,'nsPax':0,'nsPaxD':0,'cxnOut':0,'cxnOutD':0, 'carriers':{},'carriersD':{}} for pair in txtpairs }    
pax_discordant = 0
# GET NUMBER CONTAINING --, BOTHCONTAINNG -- NUMBER WITHOUT -- BUT DISCORDANT for lines if relevant
route_demands['BI_MARKET']=route_demands.apply(create_market,1)
#rd = route_demands.set_index('BI_MARKET').loc[txtpairs].reset_index()
for i,line in enumerate(route_demands.to_dict('records')):
    nonstop_flight = False
    discordant = False
    if i%1000 ==0:
        print(i)
    bimarket="_".join(sorted([line['ORIGIN'],line['DESTINATION']]))
    if line['NUM_FLIGHTS'] == 1:
        nonstop_flight=True
    leg1="_".join(sorted([line['ORIGIN'],str(line['CONNECTION'])]))
    leg2="_".join(sorted([str(line['CONNECTION']),line['DESTINATION']]))
    if bimarket in txtpairs:        
        if ( line['NUM_FLIGHTS']==2 and line['SECOND_OPERATING_CARRIER']!=line['FIRST_OPERATING_CARRIER']) or (line['FIRST_OPERATING_CARRIER']=='--' or line['SECOND_OPERATING_CARRIER']=='--'):
            discordant=True
            pax_discordant +=line['PASSENGERS']
        #if carriers discordant, only add to ...D keys
        if discordant:
            records[bimarket]['mktPaxD']+=line['PASSENGERS']
           
            
            if nonstop_flight:
                carrier_hybrid = line['FIRST_OPERATING_CARRIER']
                route = "_".join([line['ORIGIN'],line['DESTINATION']])
                #add to segment and nonstop passengers
                records[bimarket]['nsPaxD']+=line['PASSENGERS']
                records[bimarket]['segPaxD']+=line['PASSENGERS']
                #if carrier already noted for market
                if carrier_hybrid in records[bimarket]['carriersD']:
                    records[bimarket]['carriersD'][carrier_hybrid]['pax']+=line['PASSENGERS']
                    #if nonstop, and route carrier combo is new...
                    if route in records[bimarket]['carriersD'][carrier_hybrid]['routes']:
                        records[bimarket]['carriersD'][carrier_hybrid]['routes'][route]+=line['PASSENGERS']
                    else:
                        records[bimarket]['carriersD'][carrier_hybrid]['routes'][route]=line['PASSENGERS']
                #if carrier new for market...        
                else:
                    #create new carrier dict
                    records[bimarket]['carriersD'][carrier_hybrid] = {'pax':line['PASSENGERS'], 'routes':{route:line['PASSENGERS']}}
                  
            else: #if not non stop...
                carrier_hybrid = line['FIRST_OPERATING_CARRIER']+'_'+line['SECOND_OPERATING_CARRIER']
                #market new route name
                route = "_".join([line['ORIGIN'],line['CONNECTION'],line['DESTINATION']])
                #if out of network connection... add to count
                if line['CONNECTION'] not in carriers:
                    records[bimarket]['cxnOutD'] += line['PASSENGERS']                  
                #if carrier already noted for market
                if carrier_hybrid in records[bimarket]['carriersD']:
                    records[bimarket]['carriersD'][carrier_hybrid]['pax']+=line['PASSENGERS']
                    #if nonstop, and route carrier combo is new...
                    if route in records[bimarket]['carriersD'][carrier_hybrid]['routes']:
                        records[bimarket]['carriersD'][carrier_hybrid]['routes'][route]+=line['PASSENGERS']
                    else:
                        records[bimarket]['carriersD'][carrier_hybrid]['routes'][route]=line['PASSENGERS']
                #if carrier new for market...        
                else:
                    #create new carrier dict
                    records[bimarket]['carriersD'][carrier_hybrid] = {'pax':line['PASSENGERS'], 'routes':{route:line['PASSENGERS']}}
        
        #if not discordant, do same thing but also add to non D categories simulatneously          
        else:
            records[bimarket]['mktPaxD']+=line['PASSENGERS']
            records[bimarket]['mktPax']+=line['PASSENGERS']
            carrier_hybrid = line['FIRST_OPERATING_CARRIER']
            if carrier_hybrid=='--':
                print(carrier_hybrid)
            if nonstop_flight:
                route = "_".join([line['ORIGIN'],line['DESTINATION']])
                #add to segment and nonstop passengers
                records[bimarket]['nsPaxD']+=line['PASSENGERS']
                records[bimarket]['segPaxD']+=line['PASSENGERS']
                records[bimarket]['nsPax']+=line['PASSENGERS']
                records[bimarket]['segPax']+=line['PASSENGERS']
                #if carrier already noted for market
                if carrier_hybrid in records[bimarket]['carriersD']:
                    records[bimarket]['carriersD'][carrier_hybrid]['pax']+=line['PASSENGERS']
                    #if nonstop, and route carrier combo is new...
                    if route in records[bimarket]['carriersD'][carrier_hybrid]['routes']:
                        records[bimarket]['carriersD'][carrier_hybrid]['routes'][route]+=line['PASSENGERS']
                    else:
                        records[bimarket]['carriersD'][carrier_hybrid]['routes'][route]=line['PASSENGERS']
                #if carrier new for market...        
                else:
                    #create new carrier dict
                    records[bimarket]['carriersD'][carrier_hybrid] = {'pax':line['PASSENGERS'], 'routes':{route:line['PASSENGERS']}}
                #same for non D
                if carrier_hybrid in records[bimarket]['carriers']:
                    records[bimarket]['carriers'][carrier_hybrid]['pax']+=line['PASSENGERS']
                    #if nonstop, and route carrier combo is new...
                    if route in records[bimarket]['carriers'][carrier_hybrid]['routes']:
                        records[bimarket]['carriers'][carrier_hybrid]['routes'][route]+=line['PASSENGERS']
                    else:
                        records[bimarket]['carriers'][carrier_hybrid]['routes'][route]=line['PASSENGERS']
                #if carrier new for market...        
                else:
                    #create new carrier dict
                    records[bimarket]['carriers'][carrier_hybrid] = {'pax':line['PASSENGERS'], 'routes':{route:line['PASSENGERS']}}
                  
            else: #if not non stop...
                #market new route name
                route = "_".join([line['ORIGIN'],line['CONNECTION'],line['DESTINATION']])
                #if out of network connection... add to count
                if line['CONNECTION'] not in carriers:
                    records[bimarket]['cxnOutD'] += line['PASSENGERS']         
                    records[bimarket]['cxnOut'] += line['PASSENGERS']   
                #if carrier already noted for market
                if carrier_hybrid in records[bimarket]['carriersD']:
                    records[bimarket]['carriersD'][carrier_hybrid]['pax']+=line['PASSENGERS']
                    #if nonstop, and route carrier combo is new...
                    if route in records[bimarket]['carriersD'][carrier_hybrid]['routes']:
                        records[bimarket]['carriersD'][carrier_hybrid]['routes'][route]+=line['PASSENGERS']
                    else:
                        records[bimarket]['carriersD'][carrier_hybrid]['routes'][route]=line['PASSENGERS']
                #if carrier new for market...        
                else:
                    #create new carrier dict
                    records[bimarket]['carriersD'][carrier_hybrid] = {'pax':line['PASSENGERS'], 'routes':{route:line['PASSENGERS']}}
                
                #same for non D                
                if carrier_hybrid in records[bimarket]['carriers']:
                    records[bimarket]['carriers'][carrier_hybrid]['pax']+=line['PASSENGERS']
                    #if nonstop, and route carrier combo is new...
                    if route in records[bimarket]['carriers'][carrier_hybrid]['routes']:
                        records[bimarket]['carriers'][carrier_hybrid]['routes'][route]+=line['PASSENGERS']
                    else:
                        records[bimarket]['carriers'][carrier_hybrid]['routes'][route]=line['PASSENGERS']
                #if carrier new for market...        
                else:
                    #create new carrier dict
                    records[bimarket]['carriers'][carrier_hybrid] = {'pax':line['PASSENGERS'], 'routes':{route:line['PASSENGERS']}}
                
                
    #if pair match is just a single let of trip ...           
    elif not nonstop_flight and (leg1 in txtpairs or leg2 in txtpairs):
        bimarket = leg1 if leg1 in txtpairs else leg2
        if ( line['NUM_FLIGHTS']==2 and line['SECOND_OPERATING_CARRIER']!=line['FIRST_OPERATING_CARRIER']) or line['FIRST_OPERATING_CARRIER']=='--' or line['SECOND_OPERATING_CARRIER']=='--':
            discordant==True
        if discordant:
            records[bimarket]['segPaxD']+=line['PASSENGERS']
        else:
            records[bimarket]['segPax']+=line['PASSENGERS']
            records[bimarket]['segPaxD']+=line['PASSENGERS']
            
df = pd.DataFrame(records).transpose()

df = df.sort(columns=['mktPax'], ascending=False)
dfold=df
df=df[df['mktPax']>0]
df['ns/seg'] = df['nsPax']/df['segPax']

df['oon/mkt'] = df['cxnOut']/df['mktPax']

df['ns/mkt'] = df['nsPax']/df['mktPax']

def nice_carriers(row):
    carrier_dict = row['carriers']
    
    carrs = []
    paxes = []
    route_carrier = []
    route_paxes = []
    for key, value in carrier_dict.items():
        carrs.append(key)
        paxes.append(value['pax'])
        for route, count in value['routes'].items():
            route_carrier.append(key + '-' + route)
            route_paxes.append(count)
    percs = [pax/row['mktPax'] for pax in paxes]
    route_percs = [pax/row['mktPax'] for pax in route_paxes]
    num_competitors = len(carrs)
    compets = [(comp, count) for comp, count in zip(carrs,percs)]
    combos = [(comp, count) for comp, count in zip(route_carrier,route_percs)]
    return pd.Series([num_competitors,str(compets),str(combos)])
df[['numComps','listComps','routes']]=df.apply(nice_carriers,1)

df[['mktPax','segPax','nsPax','ns/seg','oon/mkt','ns/mkt','numComps','listComps','routes',]].to_csv('network_table1.csv',sep='\t')
#WHY -- showing up
networkdf = pd.read_csv('network_table1.csv',sep='\t')
networkdf = networkdf[:-4]

import ast

rows = []
for row in networkdf.to_dict('records'):
    listComps = row['listComps']
    seg = row['Unnamed: 0']
    percent_market = 0
    comp_count = 0
    airlines = []
    percs = []
    for airline in ast.literal_eval(listComps):
        
        if airline[1]>=.1:
            percent_market+=airline[1]
            comp_count+=1
            airlines.append(airline[0])
            percs.append(airline[1])
    
    for airline, perc in zip(airlines, percs):
        new_row={}
        new_row['bimarket'] = seg
        new_row['carrier'] = airline
        new_row['ms'] = round(perc,4)
        new_row['competitors'] = comp_count
        new_row['combined_ms'] = round(percent_market,4)
        rows.append(new_row)
relevant_markets = pd.DataFrame(rows)
relevant_markets.to_csv('network_combos.csv', sep='\t')

relevant_markets = pd.read_csv('network_combos.csv', sep='\t')
rm2 = relevant_markets[relevant_markets['competitors']==2]
def fullmarket_spec(gb):
    bimarket = gb['bimarket']
    print(bimarket)
    carriers = "_".join(sorted(gb['carrier'].tolist()))
    gb['spec']=bimarket + '_' + carriers
    return gb
rm2full = rm2.groupby('bimarket').apply(fullmarket_spec)
#COMPARE TO MVALID 2

        
            