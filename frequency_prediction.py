# -*- coding: utf-8 -*-
"""
Created on Sun May 17 23:07:54 2015

@author: Reed
"""
#WEDS JULY 8th MEETING

import os
import numpy as np
import pandas as pd
from collections import Counter
from itertools import product
from math import ceil

'''
function creates file consisting of nonstop market/carrier combinations for relevant markets and carriers, including empirical costs, frequencies and prices
qaurters is a list of yearly quarters to include in the analysis
t100_fn is T100 segments file from BTS
p52_fn is form P.52 file from BTS
ac_type_fn is aircraft type file from BTS
b43_fn is Schedule B4 file from BTS
airports is a list of airports in the network
freq_cuttoff is minimum daily frequency to consider
ms_cuttoff is cumulative market share to consider in each market
'''
def nonstop_market_profile(output_file = "nonstop_competitive_markets.csv",directory="C:/Users/Reed/Desktop/vaze_competition_paper", quarters=[1,2,3,4], \
    t100_fn="T100_2007.csv",p52_fn="P52_2007.csv", ac_type_fn ="AIRCRAFT_TYPE_LOOKUP.csv",b43_fn = "SCHEDULE_B43.csv", \
    freq_cuttoff = .5, ms_cuttoff=.9, airports = ['SEA','PDX','SFO','SAN','LAX','LAS','PHX','OAK','ONT','SMF','SJC']):
        
    #read in revelant bts files and supplementary data files 
    os.chdir("C:/Users/Reed/Desktop/vaze_competition_paper")
    t100 = pd.read_csv(t100_fn)
    p52 = pd.read_csv(p52_fn)
    type1 = pd.read_csv(ac_type_fn)
    b43 = pd.read_csv(b43_fn)
    
    #create bidrectional market pairs
    pairs =[sorted([pair[0],pair[1]]) for pair in product(airports,airports) if pair[0]!=pair[1] ]
    txtpairs = list(set(["_".join(pair) for pair in pairs]))
    
    #leave out fare finding for now, may add later
    #get relevant segments within network for all market pairs
    t100 = t100[t100['QUARTER'].isin(quarters)]
    t100['BI_MARKET']=t100.apply(create_market,1) #first, create bidriectional market indicator   
    relevant_t100= t100.set_index('BI_MARKET').loc[txtpairs].reset_index() #then, select markets
    
    #get relevant data from schedule P-5.2
    relevant_p52 = p52[p52['REGION']=='D'][p52['QUARTER'].isin(quarters)]
    
    #average quarterly costs if necessary 
    if len(quarters) > 1:
        expenses_by_type=relevant_p52[['AIRCRAFT_TYPE','UNIQUE_CARRIER','TOT_AIR_OP_EXPENSES', 'TOTAL_AIR_HOURS']].groupby(['AIRCRAFT_TYPE','UNIQUE_CARRIER']).aggregate(np.sum).reset_index().dropna()
    else:
        expenses_by_type = relevant_p52.dropna()   
    #calculate expenses per air hour for each type for each airline
    expenses_by_type['EXP_PER_HOUR'] = expenses_by_type['TOT_AIR_OP_EXPENSES'] / expenses_by_type['TOTAL_AIR_HOURS']

    #average relevant monthly frequencie to get daily freqencies
    t100fields =['BI_MARKET','UNIQUE_CARRIER','ORIGIN', 'DEST','AIRCRAFT_TYPE','DEPARTURES_SCHEDULED','SEATS','PASSENGERS','DISTANCE','AIR_TIME']
    t100_summed = relevant_t100[t100fields].groupby(['UNIQUE_CARRIER','BI_MARKET','ORIGIN','DEST','AIRCRAFT_TYPE']).aggregate({'DEPARTURES_SCHEDULED':lambda x: np.sum(x),'SEATS':lambda x: np.sum(x)/(365/(4/len(quarters))),'PASSENGERS':lambda x: np.sum(x)/(365/(4/len(quarters))),'DISTANCE':np.mean,'AIR_TIME': np.mean}).reset_index()
    #convert airtime to hours
    t100_summed['AIR_HOURS']=(t100_summed['AIR_TIME']/60)
    t100_summed['FLIGHT_TIME']=t100_summed['AIR_HOURS']/t100_summed['DEPARTURES_SCHEDULED']
    t100_summed['DAILY_FREQ']=t100_summed['DEPARTURES_SCHEDULED']/(365/(4/len(quarters)))
    t100_summed = t100_summed.drop('AIR_TIME',axis=1)
    #average values between segments sharing a bidirectional market 
    t100fields =['BI_MARKET','UNIQUE_CARRIER','AIRCRAFT_TYPE','DEPARTURES_SCHEDULED','SEATS','PASSENGERS','DISTANCE','AIR_HOURS', 'DAILY_FREQ']
    #merge t100 data with cost data
    t100_summed=pd.merge(t100_summed,expenses_by_type,on=['AIRCRAFT_TYPE','UNIQUE_CARRIER'])
    t100_summed['FLIGHT_COST'] = t100_summed['AIR_HOURS']*t100_summed['EXP_PER_HOUR']/t100_summed['DEPARTURES_SCHEDULED'] #get cost per flight type
    
    t100_summed = t100_summed[t100_summed['PASSENGERS']>0]
    t100_summed = t100_summed[t100_summed['DEPARTURES_SCHEDULED']>0]    
    t100_summed.to_csv('t100_summed.csv') #SEE DISTRIBUTION WITHIN MARKETS, IS AVERAGING REASONABLE -> weight by passengers: before averaging markets: plane types back and forth might not be same , passengers more likely to correlate, but now we can test
    #average flight cost between different types  [WEIGHTED BY DAILY FREQ -> CIRCULAR REASOINING HERE?]
    t100fields =['BI_MARKET','ORIGIN','DEST','UNIQUE_CARRIER','AIRCRAFT_TYPE','DEPARTURES_SCHEDULED','SEATS','PASSENGERS','DISTANCE', 'DAILY_FREQ','FLIGHT_COST','FLIGHT_TIME']
    t100_summed_avgs = t100_summed[t100fields].groupby(['UNIQUE_CARRIER','BI_MARKET']).apply(avg_costs)
    t100_craft_avg = t100_summed_avgs[t100fields].groupby(['UNIQUE_CARRIER','BI_MARKET','ORIGIN','DEST']).aggregate({'DEPARTURES_SCHEDULED':np.sum,'SEATS':np.sum,'PASSENGERS':np.sum,'DISTANCE':np.mean, 'DAILY_FREQ':np.sum,'FLIGHT_COST':np.mean,'FLIGHT_TIME':np.mean}).reset_index()
    #textfile to check passenger equivalence between market directions
    t100_craft_avg.to_csv("t100_craft_avg.csv")
    #average values between segments sharing a bidirectional market 
    t100fields =['BI_MARKET','UNIQUE_CARRIER','DEPARTURES_SCHEDULED','SEATS','PASSENGERS','DISTANCE', 'DAILY_FREQ','FLIGHT_COST','FLIGHT_TIME']
    t100_avgd = t100_craft_avg[t100fields].groupby(['UNIQUE_CARRIER','BI_MARKET']).aggregate({'DEPARTURES_SCHEDULED':np.mean,'DAILY_FREQ':np.mean,'SEATS':np.mean,'PASSENGERS':np.mean,'DISTANCE':np.mean,'FLIGHT_COST': np.mean,'FLIGHT_TIME':np.mean}).reset_index()
    #save data frame to csv: costs and frequencies by market, carrier, aircraft type
    t100_avgd.to_csv("t100_avgd.csv",sep="\t")  
    t100_avgd_clip = t100_avgd[t100_avgd['DAILY_FREQ']>=freq_cuttoff]
    #group and rank carriers within markets
    t100_grouped = t100_avgd_clip.groupby('BI_MARKET')
    grouplist = []
    for market in list(set(t100_avgd_clip['BI_MARKET'].tolist())):
        market_group = t100_grouped.get_group(market)
        new_group = market_rank(market_group)
        grouplist.append(new_group)
    t100ranked = pd.concat(grouplist,axis=0)
    ##t100_avgd_clip = t100_avgd_clip.groupby('BI_MARKET').apply(market_rank)
    t100ranked.to_csv(output_file)

 
    
    return output_file
'''
function to create a bidirectional market indicator (with airports sorted by text) for origin-destination pairs
'''    
def create_market(row):
    market = [row['ORIGIN'], row['DEST']]
    market.sort()
    return "_".join(market)
'''
get a weighed average costs and flight times across a directional market
'''
def avg_costs(gb):
    cost_weighted = np.average(gb['FLIGHT_COST'], weights=gb['DAILY_FREQ'])
    gb['FLIGHT_COST'] = np.repeat(cost_weighted,gb.shape[0])
    time_weighted = np.average(gb['FLIGHT_TIME'], weights=gb['DAILY_FREQ'])
    gb['FLIGHT_TIME'] = np.repeat(time_weighted,gb.shape[0])
            
    return gb
'''
function to average across aircraft types and rank carriers by passenger flow 
via pandas groupby function, recieves sub-dataframes, each one comprising a market
'''      
def market_rank(gb):
    Mtot = gb['PASSENGERS'].sum()
    gb['MARKET_TOT'] = np.repeat(Mtot,gb.shape[0] )    
    Mcount =gb.shape[0]
    gb['MARKET_COMPETITORS'] = np.repeat(Mcount,gb.shape[0] )
    rank = np.array(gb['PASSENGERS'].tolist()).argsort()[::-1].argsort() +1 
    gb['MARKET_RANK'] = rank         
    gb = gb.sort(columns=['MARKET_RANK'],ascending=True,axis =0)        
    gb['MS_TOT']=gb['PASSENGERS']/gb['MARKET_TOT']
    #cumulative market share upto and including that ranking
    gb['CUM_MS']=gb.apply(lambda x: gb['MS_TOT'][:x['MARKET_RANK']].sum(), axis=1)
    #cumulative market share upto that ranking
    gb['PREV_CUM_MS']=gb.apply(lambda x: gb['MS_TOT'][:x['MARKET_RANK']-1].sum(), axis=1)
    #remove those carriers that appear after cuttoff
    gb=gb[gb['PREV_CUM_MS']<=ms_cuttoff]
    #recalculate market shares
    Mtot = gb['PASSENGERS'].sum()
    gb['MARKET_TOT'] = np.repeat(Mtot,gb.shape[0] )    
    Mcount =gb.shape[0]
    gb['MARKET_COMPETITORS'] = np.repeat(Mcount,gb.shape[0] )
    gb['MS_TOT']=gb['PASSENGERS']/gb['MARKET_TOT']
    return gb    
    
    
    
#COEFFICIENT CALCULATION FOR 2, 3 4, and 1
    #use .1 cuttoff to remove 6
    
#COMPARATIVE STATICS, in context of game theory, COMPARATIVE DYNMACS
  
'''
function to find most common type of plane used on each segment
and to get a fleet composition for network for each carrier
also, calculate F
'''  
def fleet_assign(market_table_fn= "nonstop_competitive_markets.csv",ac_type_fn ="AIRCRAFT_TYPE_LOOKUP.csv",b43_fn = "SCHEDULE_B43.csv"):
    t100ranked = pd.read_csv(market_table_fn)
    markets =  t100ranked['BI_MARKET'].tolist()
    carriers = t100ranked['UNIQUE_CARRIER'].tolist()
    t100_summed= pd.read_csv('t100_summed.csv')
    t100gb = t100_summed.groupby(['UNIQUE_CARRIER','BI_MARKET'])
    t100gb_carrier = t100_summed.groupby(['UNIQUE_CARRIER'])
    type1 = pd.read_csv(ac_type_fn)
    b43 = pd.read_csv(b43_fn)
    type1ind = type1.set_index('SHORT_NAME')
    #loop through each carrier market combo found, catalogue the presence of aircraft types
    rows = []            
    i=0
    for market, carrier in zip(markets, carriers):
        i+=1
        row = {}
        group = t100gb.get_group((carrier, market))  
        group =group[['BI_MARKET','UNIQUE_CARRIER','PASSENGERS','FLIGHT_TIME','FLIGHT_COST','SEATS','DAILY_FREQ','AIRCRAFT_TYPE']].groupby(['BI_MARKET','UNIQUE_CARRIER','AIRCRAFT_TYPE']).aggregate({'PASSENGERS':np.sum,'SEATS':np.sum,'DAILY_FREQ':np.sum, 'FLIGHT_TIME':np.mean,'FLIGHT_COST':np.mean}).reset_index()
        num_types = group.shape[0]    
        #total passengers across types
        totpax = group['PASSENGERS'].sum()
        totfreq = group['DAILY_FREQ'].sum()
        group_sort = group.sort(columns=['DAILY_FREQ'], axis=0,ascending=False)
       
        group_sort['PPAX']=group_sort['PASSENGERS']/totpax
        group_sort['PFREQ']=group_sort['DAILY_FREQ']/totfreq
        max_perc = group_sort['PFREQ'].iloc[0]
        max_seats = group_sort['SEATS'].iloc[0]
        max_pax = group_sort['PASSENGERS'].iloc[0]
        max_type=group_sort['AIRCRAFT_TYPE'].iloc[0]
        
       
        group_sort['CRAFT_SEATS'] = group_sort.apply(find_seats, axis=1)       
        
        type_dict = {gs['AIRCRAFT_TYPE']:[gs['CRAFT_SEATS'],gs['FLIGHT_COST'], gs['DAILY_FREQ'],gs['PFREQ'],gs['FLIGHT_TIME'],gs['SEATS']] for gs in group_sort.to_dict('records')}
       #place into row
        row['bimarket'] = market
        row['carrier'] = carrier         
        row['max_type'] = max_type
        row['max_perc'] = max_perc
        row['max_pax'] = max_pax
        row['max_seats'] = max_seats
        row['type_dict'] = type_dict   
        #carrier distribution overall all segments
        craftlist=list(set(t100gb_carrier.get_group(carrier)['AIRCRAFT_TYPE'].tolist()))
        craft_seats = [(craft,find_seats({'AIRCRAFT_TYPE':craft})) for craft in craftlist]
        row['type_list'] = craft_seats
        rows.append(row)
    fleet_dist = pd.DataFrame(rows)    
    fleet_dist.to_csv("fleetdist1.csv", sep='\t')
    
    #ADDITIONAL CODE TO FIND NUMBER OF ASSIGNED SEATS FOR ASSIGNED AIRCRAFT IN FLEET DISTRIBUTION TABLE
    '''
    by_carrier=fleet_dist[['carrier','type_list']].groupby('carrier').aggregate(lambda x: x.iloc[0]).reset_index()
    seat_lookup = {}
    for row in by_carrier.to_dict('records'):        
        seat_lookup[row['carrier']] = {rec[0]:rec[1] for rec in row['type_list']}
    aug_fleet = pd.read_csv('fleet_dist_aug.csv')  
    def assigned_seats(row):
        carr = row['carrier']
        plane_list=str(row['assigned_type']).split('-')
        seatlist  = [seat_lookup[carr][int(plane)] for plane in plane_list]
        return round(np.mean(seatlist))
    aug_fleet['assigned_seats'] = aug_fleet.apply(assigned_seats, axis=1)
    '''
    #carrier distribution
    
    #MAKE SURE MARKET/BIMARKET times and Frequencies are consistent
        
    
    '''
function to create a bidirectional market indicator (with airports sorted by text) for origin-destination pairs
'''    
def create_market(row):
    market = [row['ORIGIN'], row['DEST']]
    market.sort()
    return "_".join(market)

   
def find_seats(row):
    model=type1[type1['AC_TYPEID']==row['AIRCRAFT_TYPE']]['SHORT_NAME'].iloc[0]
    
    try:
        seats =b43[b43['MODEL']==model]['NUMBER_OF_SEATS'].iloc[0]
    except IndexError:
        print(model)
        try:
            seats =b43[b43['MODEL']==model[:-2]]['NUMBER_OF_SEATS'].iloc[0]
        except IndexError:
            print(model)
            seats =b43[b43['MODEL']==(model[:-2] + '/A')]['NUMBER_OF_SEATS'].iloc[0]
    return seats
    
return fleet_dist

#estimate size of an aircraft's fleet of a certain type from the total airtime of that fleet type
#PASS quarters VARIABLE
def fleet_size_by_airtime(df):
    hours_pday = df['AIR_HOURS']/(365/(4/len(quarters))) + (45/60)*df['DAILY_FREQ'] #add in average turn around time
    total_time = hours_pday.sum()
    #return estimated number of aircraft assuming airlines use as mcuh of fleet as possible (a questionable assumption?)
    return total_time/18

#PASS seat_lookup VARIABLE
def construct_Ftable():
    aug_fleet = pd.read_csv('fleet_dist_aug.csv')  
    t100_summed = pd.read_csv('t100_summed.csv')
    t100_gb_carr_ac = t100_summed.grouby(['UNIQUE_CARRIER','AIRCRAFT_TYPE'])
    airlines = list(set(aug_fleet['carrier'].tolist()))
    for airline in airlines:
        ac_types= seat_lookup[airline].keys()
        for ac_type in ac_types:
            a
















   
t100_jan= t100[t100['MONTH']==1] 
t100_jan['BI_MARKET']=t100_jan.apply(create_market,1)
t100_gb = t100_jan.groupby(['UNIQUE_CARRIER','BI_MARKET'])
cost_lookup = merge_cost.groupby(['BI_MARKET','UNIQUE_CARRIER','AIRCRAFT_TYPE'])
b43ind = b43.set_index('TAIL_NUMBER')
type1ind = type1.set_index('SHORT_NAME')
rows = []            
i=0
for market, carrier in zip(markets, carriers):
    i+=1
    row = {}
    group = t100_gb.get_group((carrier, market))  
    group =group[['BI_MARKET','UNIQUE_CARRIER','PASSENGERS','AIR_TIME','SEATS','AIRCRAFT_TYPE']].groupby(['BI_MARKET','UNIQUE_CARRIER','AIRCRAFT_TYPE']).aggregate({'PASSENGERS':np.mean,'SEATS':np.mean,'AIR_TIME':np.sum}).reset_index()
    num_types = group.shape[0]    
    #total passengers across types
    totpax = group['PASSENGERS'].sum()
    
    group_sort = group.sort(columns=['PASSENGERS'], axis=0,ascending=False)
   
    group_sort['PPAX']=group_sort['PASSENGERS']/totpax
    max_perc = group_sort['PPAX'].iloc[0]
    max_seats = group_sort['SEATS'].iloc[0]
    max_pax = group_sort['PASSENGERS'].iloc[0]
    max_type=group_sort['AIRCRAFT_TYPE'].iloc[0]
    max_time=group_sort['AIR_TIME'].iloc[0]
    
    group_sort['CRAFT_SEATS'] = group_sort.apply(find_seats, axis=1)
    print(i, 'seats')
    group_sort['CRAFT_COST'] = group_sort.apply(find_cost, axis=1)
    print(i, 'costs')
    ##group_sort = group_sort[pd.notnull(group_sort['CRAFT_COST'])]
    type_dict = {gs['AIRCRAFT_TYPE']:[gs['CRAFT_SEATS'],gs['CRAFT_COST'], gs['PASSENGERS'],gs['PPAX'],gs['AIR_TIME'],gs['SEATS']] for gs in group_sort.to_dict('records')}
    
    
   
    #place into row
    row['bimarket'] = market
    row['carrier'] = carrier   
    
    row['max_type'] = max_type
    row['max_time'] = max_time
    row['max_perc'] = max_perc
    row['max_pax'] = max_pax
    row['max_seats'] = max_seats
    row['type_dict'] = type_dict
    
    rows.append(row)
fleet_dist = pd.DataFrame(rows)    
fleet_dist.to_csv("fleetdist.csv", sep='\t')
    
    
        
       
#compare individual vs bi markets
import math
#NEED TO AGGREGATE BETWEEN AIRCRAFT TYPES
def group_reg_form_2(gb):
    alpha=1.29
    beta=-.0045
    N=0
    gb=gb[:2]
    gb['p1'] = np.repeat(gb.iloc[0]['FARE'],2)
    gb['p2'] = np.repeat(gb.iloc[1]['FARE'],2)
    gb['p2Avg'] = np.repeat((gb.iloc[0]['FARE']+gb.iloc[1]['FARE'])/2,2)
    gb['f1'] = np.repeat(gb.iloc[0]['FREQ'],2)
    gb['f2'] = np.repeat(gb.iloc[1]['FREQ'],2)
    gb['f1*f2'] = gb['f1']*gb['f2']
    gb['f1^2'] = gb['f1']*gb['f1']
    gb['f2^2'] = gb['f2']*gb['f2']
    
    M=float(gb['MARKET_2'][:1])
    f1 = float(gb['f1'][:1])
    f2 = float(gb['f2'][:1])
    p1 = float(gb['p1'][:1])
    p2=float(gb['p2'][:1])
    c1 = float(gb['COSTS'].iloc[0])
    c2 = float(gb['COSTS'].iloc[1])
    pass1 = float(gb['PASSENGERS'].iloc[0])
    pass2 = float(gb['PASSENGERS'].iloc[1])
    #SEATING RESTRICTIONS FOR LATER
    #S1
    #S2
    
    #profit estimated
    numerator1 =  math.exp(alpha*math.log(f1)+beta*p1)
    denominator1 = math.exp(alpha*math.log(f1)+beta*p1)+math.exp(alpha*math.log(f2)+beta*p2)+N;
    ms1=numerator1/denominator1  
    profit1 = ms1*M*p1-c1*f1
    
    numerator2 =  math.exp(alpha*math.log(f2)+beta*p2)
    denominator2 = math.exp(alpha*math.log(f1)+beta*p1)+math.exp(alpha*math.log(f2)+beta*p2)+N;
    ms2=numerator2/denominator2  
    profit2 = ms2*(M/(365/4))*p2-c2*f2
    gb['Prof1'] = np.array([profit1,profit2])
    #perhaps correlate over alpha and beta to find max prof1 prof2 concordance
    prof21 = (pass1/(365/4))*p1-c1*f1 #AVERAGE COST SUM OF FREQUENCIS
    #EMPIRICAL PROFIT COMBINATION, EASY? WHAT ABOUT FREQUENCY USED IN CORRELATION - > SUM, THEN SUBTRACT OUT
    prof22 = (pass2/(365/4))*p2-c2*f2
    gb['Prof2']= np.array([prof21,prof22])
    
    return gb
    
def group_reduced_2CARRIER(gb):   
    gb=gb[:2]
    gb['p1'] = np.repeat(gb.iloc[0]['FARE'],2)
    gb['p2'] = np.repeat(gb.iloc[1]['FARE'],2)    
    gb['f1'] = np.repeat(gb.iloc[0]['FREQ'],2)
    gb['f2'] = np.repeat(gb.iloc[1]['FREQ'],2)
    gb['f1*f2'] = gb['f1']*gb['f2']
    gb['f1^2'] = gb['f1']*gb['f1']
    gb['f2^2'] = gb['f2']*gb['f2']
    gb['c1'] = np.repeat(gb.iloc[0]['COSTS'],2)
    gb['c2'] = np.repeat(gb.iloc[1]['COSTS'],2)  
    
    M=float(gb['MARKET_2'][:1])
  
    return gb

mreg = m.groupby(markettype).apply(group_reduced_2CARRIER)
mreg2 = mreg.drop('BI_MARKET',1).reset_index()
#mreg2_reduced = mreg[['MARKET','BI_MARKET','UNIQUE_CARRIER','COSTS','FREQ','MARKET_COMPT]]
mreg2.to_csv('verif_2car.csv',sep='\t')
mreg2  = pd.read_csv('verif_2car.csv',sep='\t')
#get markets with at least 3 carriers
m3 = m[m['MARKET_COMPETITORS']>2]

def group_reduced_3CARRIER(gb):   
    gb=gb[:3]
    
    gb['p1'] = np.repeat(gb.iloc[0]['FARE'],3)
    gb['p2'] = np.repeat(gb.iloc[1]['FARE'],3) 
    gb['p3'] = np.repeat(gb.iloc[2]['FARE'],3)
 
    gb['f1'] = np.repeat(gb.iloc[0]['FREQ'],3)
    gb['f2'] = np.repeat(gb.iloc[1]['FREQ'],3)
    gb['f3'] = np.repeat(gb.iloc[2]['FREQ'],3)
    gb['f1*f2'] = gb['f1']*gb['f2']
    gb['f1*f3'] = gb['f1']*gb['f3']
    gb['f2*f3'] = gb['f2']*gb['f3']
    gb['f1^2'] = gb['f1']*gb['f1']
    gb['f2^2'] = gb['f2']*gb['f2']
    gb['c1'] = np.repeat(gb.iloc[0]['COSTS'],3)
    gb['c2'] = np.repeat(gb.iloc[1]['COSTS'],3)  
    gb['c3'] = np.repeat(gb.iloc[2]['COSTS'],3) 
  
    return gb
    
    


mreg3 = m3.groupby(markettype).apply(group_reduced_3CARRIER)
mreg3 = mreg3.drop('BI_MARKET',1).reset_index()
#mreg2_reduced = mreg[['MARKET','BI_MARKET','UNIQUE_CARRIER','COSTS','FREQ','MARKET_COMPT]]
mreg3.to_csv('verif_3car.csv',sep='\t')

mreg3 = pd.read_csv('verif_3car.csv', sep='\t')


routes = pd.read_csv("route_demand_Q1.csv")
def create_market(row):
    market = [row['ORIGIN'], row['DESTINATION']]
    market.sort()
    return "_".join(market)
routes['BI_MARKET']=routes.apply(create_market,1)
routes = routes.set_index('BI_MARKET').loc[markets]
routes = routes.reset_index()
o_gs = list(set(routes['ORIGIN'].tolist() +routes['DESTINATION'].tolist()))

#MARKET SIZ PER DAY

#Intercept	f1	f2	f1^2	f2^2	f1*f2
coef=[-274960.0,-16470.0,	34936.0,	425.6,	-1300.0,	595.7]
with open('coefsN2.txt','w') as outfile:
    for i,row in enumerate(mreg2.to_dict('records')):
        Mold = 1000
        C1old =10000
        C2old = 10000
        C1new = float(row['c1'])
        C2new = float(row['c2'])
        Mnew = int(row['MARKET_2'])
        if i%2==0:
            transcoef1 = [-(Mnew/Mold)*coef[0],(Mnew/Mold)*(C1old-coef[1])-C1new] + [-(Mnew/Mold)*coef[i] for i in range(2,6)]
            transcoef2 = [-(Mnew/Mold)*coef[0],(Mnew/Mold)*(C2old-coef[1])-C2new] + [-(Mnew/Mold)*coef[i] for i in range(2,6)]
            #transcoef2 = [-(Mnew/Mold)*coef[0], -(Mnew/Mold)*coef[2],(Mnew/Mold)*(C2old-coef[1])-C2new] + [-(Mnew/Mold)*coef[i] for i in range(3,6)]
            line = transcoef1+transcoef2
            line = [str(i) for i in line]
            outfile.write("\t".join(line) + "\n")          
            
            
passen = 0        

#three player (need to solve for MS adjustment)
coef=[-274960.0,-16470.0,	34936.0,	425.6,	-1300.0,	595.7]
with open('coefsN3.txt','w') as outfile:
    for i,row in enumerate(mreg2.to_dict('records')):
        Mold = 1000
        C1old =10000
        C2old = 10000
        C3old = 10000
        C1new = float(row['c1'])
        C2new = float(row['c2'])
        C3new = float(row['c3'])
        Mnew = int(row['MARKET_2'])
        if i%2==0:
            transcoef1 = [-(Mnew/Mold)*coef[0],(Mnew/Mold)*(C1old-coef[1])-C1new] + [-(Mnew/Mold)*coef[i] for i in range(2,6)]
            transcoef2 = [-(Mnew/Mold)*coef[0],(Mnew/Mold)*(C2old-coef[1])-C2new] + [-(Mnew/Mold)*coef[i] for i in range(2,6)]
            transcoef2 = [-(Mnew/Mold)*coef[0],(Mnew/Mold)*(C2old-coef[1])-C2new] + [-(Mnew/Mold)*coef[i] for i in range(2,6)]
            #transcoef2 = [-(Mnew/Mold)*coef[0], -(Mnew/Mold)*coef[2],(Mnew/Mold)*(C2old-coef[1])-C2new] + [-(Mnew/Mold)*coef[i] for i in range(3,6)]
            line = transcoef1+transcoef2
            line = [str(i) for i in line]
            outfile.write("\t".join(line) + "\n")
            
            
        
'''
 checking of order 
'''        
        
for i in routes.to_dict('records'):
    if i['CONNECTION'] in k[1:]:
        passen += i['PASSENGERS']
        
        
     k=[p for p in list(set(routes['CONNECTION'].tolist())) if p not in list(set(list(W_markets[0].tolist() +W_markets[1].tolist())))]

mreg2['c1_high'] = mreg2.apply(lambda x: 1 if x['c1']>x['c2'] else False, 1)
c1high=[j for i,j in enumerate(mreg2['c1_high']) if i%2==0]


   freq1low= [0,
     0,
     0,
     0,
     1,
     1,
     1,
     0,
     1,
     0,
     1,
     1,
     1,
     0,
     0,
     0,
     0,
     0,
     1,
     0,
     0,
     0,
     0,
     0,
     1,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     1,
     1,
     0,
     0,
     1,
     0,
     0,
     0]
 test = []
 for freq, cost in zip(freq1low,c1high):
     if freq==cost:
         test.append(0)
     else:
         test.append(1)
#differential follows cost ranking in all cases: lower cost, higher freq
     
from scipy.stats import linregress
M2 = mreg2[mreg2.index%2==0]
linregress(M2['f1']-M2['f2'], M2['c1']-M2['c2'])
with open('mreg2_f1_f2.csv','w') as outfile:
    for row in M2.to_dict('records'):
        print(row)
        outfile.write(str(row['f1'])+'\t'+str(row['f2'])+'\n')
#3/4s cost and freqeuency discrepancy match (higher cost, lower frequency or vice versa
cost_freq_concord = []     
for row in M2.to_dict('records'):
    if row['f1'] > row['f2']:
        onef = 1
    else:
        onef = 0
    if row['c1'] > row['c2']:
        onec = 1
    else:
        onec = 0
    if onef == onec:
        cost_freq_concord.append(0)
    else:
        cost_freq_concord.append(1)
        
        
        
        
        
        
        
        
'''
end checking of order 
'''        
        
        
##mvalid2 = mreg2[mreg2['MARKET_COMPETITORS']==2]
mvalid1 = mreg2#[mreg2['MS_TOT']>=.1]
mv1gb=mvalid1.groupby('BI_MARKET')
only2 = []
for row in mvalid1.to_dict('records'):
    if mv1gb.get_group(row['BI_MARKET']).shape[0]==2:
        only2.append(True)
    else:
        only2.append(False)      
        
mvalid2 = mvalid1[only2]
def fullmarket_spec(gb):
    bimarket = gb['BI_MARKET']
    print(bimarket)
    carriers = "_".join(sorted(gb['UNIQUE_CARRIER'].tolist()))
    gb['SPEC']=bimarket + '_' + carriers
    return gb
rvalid2full = mvalid2.groupby('BI_MARKET').apply(fullmarket_spec)
rvalid2full.to_csv('rvalid2full.csv',sep='\t')
rvalid2full = pd.read_csv('rvalid2full.csv',sep='\t')

rmlist=list(set(rm2full['spec'].tolist()))

validlist=list(set(rvalid2full['SPEC'].tolist()))
intersect = [val for val in rmlist if val in validlist]
valid_freqs = rvalid2full.set_index('SPEC').loc[intersect].reset_index()
valid_freqs1 = valid_freqs.iloc[[i for i in range(0,valid_freqs.shape[0]) if i%2==0]][['f1','f2']]
for row in valid_freqs1.to_dict('records'):
    print(row['f1'],row['f2'])
coef=[-274960.0,-16470.0,	34936.0,	425.6,	-1300.0,	595.7]
with open('coefsN.txt','w') as outfile:
    for i,row in enumerate(valid_freqs.to_dict('records')):
        Mold = 1000
        C1old =10000
        C2old = 10000
        C1new = float(row['c1'])
        C2new = float(row['c2'])
        Mnew = int(row['MARKET_2'])
        if i%2==0:
            transcoef1 = [-(Mnew/Mold)*coef[0],(Mnew/Mold)*(C1old-coef[1])-C1new] + [-(Mnew/Mold)*coef[i] for i in range(2,6)]
            transcoef2 = [-(Mnew/Mold)*coef[0],(Mnew/Mold)*(C2old-coef[1])-C2new] + [-(Mnew/Mold)*coef[i] for i in range(2,6)]
            #transcoef2 = [-(Mnew/Mold)*coef[0], -(Mnew/Mold)*coef[2],(Mnew/Mold)*(C2old-coef[1])-C2new] + [-(Mnew/Mold)*coef[i] for i in range(3,6)]
            line = transcoef1+transcoef2
            line = [str(i) for i in line]
            outfile.write("\t".join(line) + "\n") 
            
            
            
            
'''
old fleet allocation script
'''

# find number of planes of each type used in each market by each airline
markets =  rvalid2full['BI_MARKET'].tolist()
carriers = rvalid2full['UNIQUE_CARRIER'].tolist()
aotp_jan = pd.read_csv("AOTP_JAN.csv")
aotp_jan['BI_MARKET']=aotp_jan.apply(create_market,1) 
aotp_jan_gb = aotp_jan.groupby(['UNIQUE_CARRIER','BI_MARKET'])
b43ind = b43.set_index('TAIL_NUMBER')
type1ind = type1.set_index('SHORT_NAME')
rows = []
for market, carrier in zip(markets, carriers):
    row = {}
    group = aotp_jan_gb.get_group((carrier, market))
    group = group[pd.notnull(group['TAIL_NUM'])]
    nflights = group.shape[0]
    nunique  = group['TAIL_NUM'].nunique()
    tails = list(set(group['TAIL_NUM'].tolist()))
    #model for each aircraft number
    models = b43ind.loc[tails]['MODEL'].tolist()
    #type number for each model
    types = type1ind.loc[models]['AC_TYPEID'].value_counts().to_dict()
    #seats for each model
    seats_lookup = pd.concat([type1ind.loc[models]['AC_TYPEID'], b43ind.loc[tails]['NUMBER_OF_SEATS']], axis=1).drop_duplicates().set_index('AC_TYPEID')
   
    #get count and percentage of each type 
    seat_sum = sum([count*seats_lookup[ac_type] for ac_type, count in types.items()])
    type_dict = {ac_type: [count, count/sum(types.values()), count*seats_lookup[ac_type], count*seats_lookup[ac_type]/seat_sum] for ac_type, count in types.items()}
    #get max type, count and percentage for this market
    max_perc = 0
    for ac_type, counts in type_dict.items():
        if counts[3]>=max_perc:
            max_perc_freq = counts[1]
            max_count_freq = counts[0]
            max_perc_seat = counts[3]
            max_count_seat = counts[2]
            max_type = ac_type

    #place into row
    row['bimarket'] = market
    row['carrier'] = carrier
    row['num_flights'] = nflights
    row['num_craft'] = nunique
    row['tails'] = tails
    row['type_dict'] = type_dict
    row['max_type'] = max_type
    row['max_perc_freq'] = max_perc_freq
    row['max_count_freq'] = max_count_freq
    row['max_perc_seat'] = max_perc_seat
    row['max_count_seat'] = max_count_seat
    
    rows.append(row)
fleet_dist = pd.DataFrame(rows)            
    
        

'''
end old fleet allocation script
'''



#get most common aircraft type on segment for each carrier
# find number of planes of each type used in each market by each airline
markets =  rvalid2full['BI_MARKET'].tolist()
carriers = rvalid2full['UNIQUE_CARRIER'].tolist()
##aotp_jan = pd.read_csv("AOTP_JAN.csv")
##aotp_jan['BI_MARKET']=aotp_jan.apply(create_market,1)
t100_jan= t100[t100['MONTH']==1] 
t100_jan['BI_MARKET']=t100_jan.apply(create_market,1)
t100_gb = t100_jan.groupby(['UNIQUE_CARRIER','BI_MARKET'])
cost_lookup = merge_cost.groupby(['BI_MARKET','UNIQUE_CARRIER','AIRCRAFT_TYPE'])
b43ind = b43.set_index('TAIL_NUMBER')
type1ind = type1.set_index('SHORT_NAME')
rows = []
       
def find_seats(row):
    model=type1[type1['AC_TYPEID']==row['AIRCRAFT_TYPE']]['SHORT_NAME'].iloc[0]
    
    try:
        seats =b43[b43['MODEL']==model]['NUMBER_OF_SEATS'].iloc[0]
    except IndexError:
        print(model)
        seats =b43[b43['MODEL']==model[:-2]]['NUMBER_OF_SEATS'].iloc[0]
    return seats
    
def find_cost(row):
    try:
        costgroup = cost_lookup.get_group((row['BI_MARKET'], row['UNIQUE_CARRIER'],row['AIRCRAFT_TYPE']))
        #print(costgroup)
        costs = costgroup['COSTS'].mean()
    except KeyError:
        costs = np.nan
   # print(costs)
    return costs
            
i=0
for market, carrier in zip(markets, carriers):
    i+=1
    row = {}
    group = t100_gb.get_group((carrier, market))  
    group =group[['BI_MARKET','UNIQUE_CARRIER','PASSENGERS','AIR_TIME','SEATS','AIRCRAFT_TYPE']].groupby(['BI_MARKET','UNIQUE_CARRIER','AIRCRAFT_TYPE']).aggregate({'PASSENGERS':np.mean,'SEATS':np.mean,'AIR_TIME':np.sum}).reset_index()
    num_types = group.shape[0]    
    #total passengers across types
    totpax = group['PASSENGERS'].sum()
    
    group_sort = group.sort(columns=['PASSENGERS'], axis=0,ascending=False)
   
    group_sort['PPAX']=group_sort['PASSENGERS']/totpax
    max_perc = group_sort['PPAX'].iloc[0]
    max_seats = group_sort['SEATS'].iloc[0]
    max_pax = group_sort['PASSENGERS'].iloc[0]
    max_type=group_sort['AIRCRAFT_TYPE'].iloc[0]
    max_time=group_sort['AIR_TIME'].iloc[0]
    
    group_sort['CRAFT_SEATS'] = group_sort.apply(find_seats, axis=1)
    print(i, 'seats')
    group_sort['CRAFT_COST'] = group_sort.apply(find_cost, axis=1)
    print(i, 'costs')
    ##group_sort = group_sort[pd.notnull(group_sort['CRAFT_COST'])]
    type_dict = {gs['AIRCRAFT_TYPE']:[gs['CRAFT_SEATS'],gs['CRAFT_COST'], gs['PASSENGERS'],gs['PPAX'],gs['AIR_TIME'],gs['SEATS']] for gs in group_sort.to_dict('records')}
    
    
   
    #place into row
    row['bimarket'] = market
    row['carrier'] = carrier   
    
    row['max_type'] = max_type
    row['max_time'] = max_time
    row['max_perc'] = max_perc
    row['max_pax'] = max_pax
    row['max_seats'] = max_seats
    row['type_dict'] = type_dict
    
    rows.append(row)
fleet_dist = pd.DataFrame(rows)    
fleet_dist.to_csv("fleetdist.csv", sep='\t')