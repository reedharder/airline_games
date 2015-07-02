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
#REMOVE SLC AND RENO
#HEURISTIC RULES FOR 2 CARRIER CUTTOFF: 90% of MARKET
#convert market to per day
#TUEsday 9th skype 11: 00am or thursdat
# fRI 12 th 11:00 am

os.chdir("C:/Users/Reed/Desktop/vaze_competition_paper")
t100 = pd.read_csv("T100_2007.csv")
p52 = pd.read_csv("P52_2007.csv")
type1 = pd.read_csv("AIRCRAFT_TYPE_LOOKUP.csv")
b43 = pd.read_csv("SCHEDULE_B43.csv")
collapsed_market = pd.read_csv("D1B1_AGGREGATED.csv")
W_markets = pd.read_csv("W_markets.txt",sep="\t",header=None)
#get fares of relevant markets

def create_market(row):
    market = [row[0], row[1]]
    market.sort()
    return "_".join(market)
W_markets['BI_MARKET'] = W_markets.apply(create_market,1)
markets = list(set(W_markets['BI_MARKET'].tolist()))

rows = []
cm = collapsed_market.to_dict('records')
for row in cm:
    if row['BI_MARKET'] in markets:
        rows.append(row)
fares = pd.DataFrame(rows)
fares['UNIQUE_CARRIER'] = fares['OPERATING_CARRIER']
fares = fares.drop('DISTANCE',1).drop('OPERATING_CARRIER',1).drop('Unnamed: 0',1)

#get t100 records of relevat western markets
def create_market(row):
    market = [row['ORIGIN'], row['DEST']]
    market.sort()
    return "_".join(market)
t100['BI_MARKET']=t100.apply(create_market,1) 
t101 = t100.set_index('BI_MARKET')
t101 = t101.loc[markets]
t101 =t101.reset_index()
t101 = t101[t101['QUARTER']==1]

# get cost per hour per airline/aircraft combo
p52 = p52[p52['QUARTER']==1]
p52 = p52[p52['REGION']=='D']
p52['EXP_PER_HOUR'] = p52['TOT_AIR_OP_EXPENSES'] / p52['TOTAL_AIR_HOURS']
#POTENTIALLY A GOOD OPTION, CHECK WITH VIKRANT. COSTS PER FLIGHT ARE STILL HIGH THOUGH
##p=p52[p52['TOTAL_AIR_HOURS']>20]
p=p52
by_type=p[['AIRCRAFT_TYPE','UNIQUE_CARRIER','EXP_PER_HOUR']].groupby(['AIRCRAFT_TYPE','UNIQUE_CARRIER']).aggregate(np.mean)
by_type_records = by_type.reset_index().dropna()

def create_ordered_market(row):
    market = [row['ORIGIN'], row['DEST']]
    return "_".join(market)
t101['MARKET'] = t101.apply(create_ordered_market,1)
#DIVIDE MARKETS BY MONTH
t102=t101[['BI_MARKET','MARKET','UNIQUE_CARRIER','AIRCRAFT_TYPE','SEATS','PASSENGERS','DEPARTURES_SCHEDULED','AIR_TIME']].groupby(['BI_MARKET','MARKET','UNIQUE_CARRIER','AIRCRAFT_TYPE']).aggregate({'SEATS':np.sum,'PASSENGERS':np.sum,'DEPARTURES_SCHEDULED':np.sum,'AIR_TIME':np.sum}).reset_index()
t102['FREQ']=t102['DEPARTURES_SCHEDULED']/(365/4)
##t102 = t102[t102['FREQ']>=1]
##t102 = t102[t102['FREQ']<=20] #SET TO QX TO 20

merge_cost = pd.merge(t102,by_type_records,on=['AIRCRAFT_TYPE','UNIQUE_CARRIER'])
#total expenses across all hours over departures = cost per departure
merge_cost['COSTS'] = merge_cost['AIR_TIME']*merge_cost['EXP_PER_HOUR']*(1/60)/merge_cost['DEPARTURES_SCHEDULED']
merge_full = pd.merge(merge_cost,fares,on=['BI_MARKET','UNIQUE_CARRIER'])
merge_full['PROFIT']=merge_full['PASSENGERS']*merge_full['FARE'] - merge_full['FREQ']*merge_full['COSTS']
#MERGE BI_MARKETS?
#FOR MARKET , DO NON STOP FIRST, THEN GET ALL
#HOW TO DEAL WITH DIFFERENT TYPES OF MARKETS

#


#CHECK RANK ORDER FOR 2SLS
#MARKET SHARE TO PROFITS BOTH WAYS? WHICH MARKET SIZE? DEPENDS ON N PRESUMABLY

#CHECK COSTS
#DO THREE WAY AS WELL
#DOUBLE CHECK ALL UNITS, MAKE SURE AGGREGATE WORKS

#for each artpicular N solve for  Market size

#
#tomorrow, run 2 player nd 3 player games
#modify N check, 
#order computer 
#run analysis for routes and markets
#begin to observe network
#2sls
def group_rank(gb):
    gb =gb[['MARKET','BI_MARKET','UNIQUE_CARRIER','PASSENGERS','FREQ','FARE','COSTS']].groupby(['MARKET','BI_MARKET','UNIQUE_CARRIER']).aggregate({'PASSENGERS':np.sum,'FREQ':np.sum,'FARE':np.mean, 'COSTS':np.mean}).reset_index()
    #average stats across ORDERED MARKETS OF SAME BYMARKE    
    gb=gb.groupby(['BI_MARKET','UNIQUE_CARRIER']).aggregate({'PASSENGERS':np.mean,'FREQ':np.mean,'FARE':np.mean, 'COSTS':np.mean})
    Mtot = gb['PASSENGERS'].sum()
    gb['MARKET_TOT'] = np.repeat(Mtot,gb.shape[0] )    
    Mcount =gb.shape[0]
    gb['MARKET_COMPETITORS'] = np.repeat(Mcount,gb.shape[0] )
    rank = gb['PASSENGERS'].argsort().argsort() +1 
    gb['MARKET_RANK'] = rank 
    gb = gb.sort(columns=['MARKET_RANK'],ascending=False,axis =0)
    gb['MS_TOT']=gb['PASSENGERS']/gb['MARKET_TOT']
    M2 = gb[:2]['PASSENGERS'].sum()
    gb['MARKET_2'] = np.repeat(M2,gb.shape[0] )
    gb['MS_2']=gb['PASSENGERS']/gb['MARKET_2']
    gb['TOP2_MS']=M2/Mtot
    
    return gb

#compare individual vs bi markets
markettype='BI_MARKET'
merge_market = merge_full.groupby(markettype).apply(group_rank)
merge_market.index = merge_market.index.droplevel(0)

m=merge_market.reset_index()
m.to_csv("validation_df.csv", sep='\t')
#remove non competitive markets
m = m[m['MARKET_COMPETITORS']>1]
##m = m[m['MARKET_COMPETITORS']==2]
#remove below certain 2 market dominance
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