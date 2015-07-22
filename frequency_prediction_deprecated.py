# -*- coding: utf-8 -*-
"""
Created on Sun May 17 23:07:54 2015

@author: Reed
"""


import os
import numpy as np
import pandas as pd
from itertools import product

'''
STEP ONE: CREATE NETWORK PROFILE TABLE, MAJOR CARRIERS IN MAJOR MARKETS IN SELECT AIRPORTS

'''
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
merge_HP: True if carrier HP is to be merged with UA
'''
def nonstop_market_profile(output_file = "nonstop_competitive_markets.csv",aotp_fn = 'aotp_march.csv',directory="C:/Users/Reed/Desktop/vaze_competition_paper", quarters=[1,2,3,4], \
    t100_fn="T100_2007.csv",p52_fn="P52_2007.csv", merge_HP=True, \
    freq_cuttoff = .5, ms_cuttoff=.1, airports = ['SEA','PDX','SFO','SAN','LAX','LAS','PHX','OAK','ONT','SMF','SJC']):
        
    #read in revelant bts files and supplementary data files 
    os.chdir("C:/Users/Reed/Desktop/vaze_competition_paper")
    t100 = pd.read_csv(t100_fn)
    p52 = pd.read_csv(p52_fn)

    #create bidrectional market pairs
    pairs =[sorted([pair[0],pair[1]]) for pair in product(airports,airports) if pair[0]!=pair[1] ]
    txtpairs = list(set(["_".join(pair) for pair in pairs]))
    
    #leave out fare finding for now, may add later
    #get relevant segments within network for all market pairs
    t100 = t100[t100['QUARTER'].isin(quarters)]
    print("creating markets...")
    t100['BI_MARKET']=t100.apply(create_market,1) #first, create bidriectional market indicator   
    print("done")
    relevant_t100= t100.set_index('BI_MARKET').loc[txtpairs].reset_index() #then, select markets
    #merge carrier HP under UA if this is called for.
    if merge_HP:
        relevant_t100['UNIQUE_CARRIER']=relevant_t100['UNIQUE_CARRIER'].replace('HP','US')

    #get relevant data from schedule P-5.2
    relevant_p52 = p52[p52['REGION']=='D'][p52['QUARTER'].isin(quarters)]
    
    #get aotp to get flight times
    aotp_mar = pd.read_csv(aotp_fn)
    aotp_mar['BI_MARKET']=aotp_mar.apply(create_market,1) 
    #DISSAGREGGATE BY AIRCRAFT TYPE LATER
    aotp_mar_times = aotp_mar[['UNIQUE_CARRIER','BI_MARKET','AIR_TIME']].groupby(['UNIQUE_CARRIER','BI_MARKET']).aggregate(lambda x: np.mean(x)/60)
    aotp_mar_times = aotp_mar_times.reset_index().groupby(['UNIQUE_CARRIER','BI_MARKET'])    
    
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
    #NOTE: FOR THIS HOURS AND FRACTION FOF TOTAL HOURS FOR FLIGHT FOR CARRIER MUST BE ADDED , I THINK IT CAN BE DONE HERE 
    t100_summed = t100_summed[t100_summed['PASSENGERS']>0]
    t100_summed = t100_summed[t100_summed['DEPARTURES_SCHEDULED']>0]    
    t100_summed.to_csv('t100_summed.csv') # NOTE: SEE DISTRIBUTION WITHIN MARKETS, IS AVERAGING REASONABLE -> weight by passengers: before averaging markets: plane types back and forth might not be same , passengers more likely to correlate, but now we can test
    #average flight cost between different types  
    t100fields =['BI_MARKET','ORIGIN','DEST','UNIQUE_CARRIER','AIRCRAFT_TYPE','DEPARTURES_SCHEDULED','SEATS','PASSENGERS','DISTANCE', 'DAILY_FREQ','FLIGHT_COST','FLIGHT_TIME','AIR_HOURS']
    t100_summed_avgs = t100_summed[t100fields].groupby(['UNIQUE_CARRIER','BI_MARKET']).apply(avg_costs)
    t100_craft_avg = t100_summed_avgs[t100fields].groupby(['UNIQUE_CARRIER','BI_MARKET','ORIGIN','DEST']).aggregate({'DEPARTURES_SCHEDULED':np.sum,'SEATS':np.sum,'PASSENGERS':np.sum,'DISTANCE':np.mean, 'DAILY_FREQ':np.sum,'FLIGHT_COST':np.mean,'FLIGHT_TIME':np.mean,'AIR_HOURS':np.mean}).reset_index()
    #textfile of t100 summed over months, to check passenger equivalence between market directions
    t100_craft_avg.to_csv("t100_craft_avg.csv")
    #average values between segments sharing a bidirectional market 
    t100fields =['BI_MARKET','UNIQUE_CARRIER','DEPARTURES_SCHEDULED','SEATS','PASSENGERS','DISTANCE', 'DAILY_FREQ','FLIGHT_COST','FLIGHT_TIME','AIR_HOURS']
    t100_avgd = t100_craft_avg[t100fields].groupby(['UNIQUE_CARRIER','BI_MARKET']).aggregate({'DEPARTURES_SCHEDULED':np.mean,'DAILY_FREQ':np.mean,'SEATS':np.mean,'PASSENGERS':np.mean,'DISTANCE':np.mean,'FLIGHT_COST': np.mean,'FLIGHT_TIME':np.mean,'AIR_HOURS':np.mean}).reset_index()
    #save data frame to csv: costs and frequencies by market, carrier, aircraft type
    t100_avgd.to_csv("t100_avgd.csv",sep="\t")  
    #remove entries below daily frequency cuttoff
    t100_avgd_clip = t100_avgd[t100_avgd['DAILY_FREQ']>=freq_cuttoff]
    #group and rank carriers within markets
    t100_grouped = t100_avgd_clip.groupby('BI_MARKET')
    grouplist = []
    for market in list(set(t100_avgd_clip['BI_MARKET'].tolist())):
        market_group = t100_grouped.get_group(market)
        new_group = market_rank(market_group, ms_cuttoff=ms_cuttoff)
        grouplist.append(new_group)
    t100ranked = pd.concat(grouplist,axis=0)
    t100ranked=t100ranked.sort(columns=['BI_MARKET','MARKET_RANK'])
    #Add aotp based average flight time, to replace flaulty flight time NOTE: REMOVE OLD FLIGHT TIME ABOVE
    def seg_block_hours(row):
        #idiosyncratic function for estimating block hours from AOTP 
        try: 
            block_hours=aotp_mar_times.get_group((row['UNIQUE_CARRIER'],row['BI_MARKET']))['AIR_TIME'].iloc[0]
        except KeyError:
            try:
                aotp_mar_times_avg =aotp_mar[['UNIQUE_CARRIER','BI_MARKET','AIR_TIME']].groupby(['UNIQUE_CARRIER','BI_MARKET']).aggregate(lambda x: np.mean(x)/60)
                aotp_mar_times_avg =aotp_mar_times_avg.reset_index().groupby(['BI_MARKET'])
                block_hours=aotp_mar_times_avg.get_group(row['BI_MARKET'])['AIR_TIME'].iloc[0]
            except KeyError:
                mkk=row['BI_MARKET'].replace('ONT','LAX')
                block_hours=aotp_mar_times_avg.get_group(mkk)['AIR_TIME'].iloc[0]   
        return block_hours
    t100ranked['AOTP_FLIGHT_TIME']=t100ranked.apply(seg_block_hours,1)
    
    ##t100_avgd_clip = t100_avgd_clip.groupby('BI_MARKET').apply(market_rank)
    t100ranked.to_csv(output_file)
    
    return t100ranked
'''
helper function to create a bidirectional market indicator (with airports sorted by text) for origin-destination pairs
'''    
def create_market(row):
    market = [row['ORIGIN'], row['DEST']]
    market.sort()
    return "_".join(market)
'''
helper function get a weighed average costs and flight times across a directional market
'''
def avg_costs(gb):
    cost_weighted = np.average(gb['FLIGHT_COST'], weights=gb['DAILY_FREQ'])
    gb['FLIGHT_COST'] = np.repeat(cost_weighted,gb.shape[0])
    time_weighted = np.average(gb['FLIGHT_TIME'], weights=gb['DAILY_FREQ'])
    gb['FLIGHT_TIME'] = np.repeat(time_weighted,gb.shape[0])            
    return gb
'''
helper function to average across aircraft types and rank carriers by passenger flow 
via pandas groupby function, recieves sub-dataframes, each one comprising a market
'''      
def market_rank(gb, ms_cuttoff):
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
    gb=gb[gb['MS_TOT']>=ms_cuttoff]
    #recalculate market shares
    Mtot = gb['PASSENGERS'].sum()
    #get total market size
    gb['MARKET_TOT'] = np.repeat(Mtot,gb.shape[0] )   
    #get total number of competitors in market and save as column 
    Mcount =gb.shape[0]
    gb['MARKET_COMPETITORS'] = np.repeat(Mcount,gb.shape[0] )
    #get market share as passengers for that carrier over total market size 
    gb['MS_TOT']=gb['PASSENGERS']/gb['MARKET_TOT']
    return gb    



'''
STEP TWO: ANALYZE NETWORK FLEET COMPOSITION

'''    

'''
function to get fleets available to each carrier by comparing time ratios in and out of network and comparing to full inventory size
'''
def Ftable_new(output_fn="fleet_lookup.csv", full_t00_fn="t100_seg_all.csv", ac_type_fn ="AIRCRAFT_TYPE_LOOKUP.csv",t100summed_fn="t100_summed.csv",market_table_fn= "nonstop_competitive_markets.csv",b43_fn = "SCHEDULE_B43.csv"):
    t100_all = pd.read_csv(full_t00_fn)
    b43 = pd.read_csv(b43_fn)
    type1 = pd.read_csv(ac_type_fn)
    t100ranked = pd.read_csv(market_table_fn)
    t100ranked_gb =t100ranked.groupby(['UNIQUE_CARRIER'])
    t100_summed =pd.read_csv(t100summed_fn)
    t100_gb = t100_summed.groupby(['UNIQUE_CARRIER'])
    markets_sorted = sorted(list(set(t100ranked['BI_MARKET'].tolist())))       
    carriers_sorted = sorted(list(set(t100ranked['UNIQUE_CARRIER'].tolist())))
    treduced=t100_all[t100_all['UNIQUE_CARRIER'].isin(carriers_sorted)]
    treduced['BI_MARKET']=treduced.apply(create_market,1)
    treduced['AIR_TIME']= treduced['AIR_TIME']/60
    treduced['IN_NETWORK'] = treduced.apply(lambda x: 1 if x['BI_MARKET'] in markets_sorted else 0, 1)
    treduced['TIME_IN_NETWORK'] = treduced['IN_NETWORK']*treduced['AIR_TIME']
    tr_gb = treduced.groupby(['UNIQUE_CARRIER','AIRCRAFT_TYPE'])
    #create ac numerical iD to short name converson table
    keys =sorted(list(set(t100_summed['AIRCRAFT_TYPE'].tolist())))
    reduced_type = type1.set_index('AC_TYPEID').loc[keys].reset_index()[['index','SHORT_NAME']]
    #apparently corresponding model 
    model = [['SF-340/A'],['EMB-120'],['DASH8-Q4'],['DASH8-1'],['DHC8-200'],['B737-7','B737-7/L'],['B737-8'],['B737-5'],['B737-4'],['B737-3'],['B757-2'],['B767-2'],['B767-3'],['B777-2'],['CRJ-2/4'],['RJ-700'],['B737-9'],['CRJ-900'],['A318'],['MD-80'],['MD-90'],['EMB-135'],['EMB-145'],['EMB-140'],['A320-1/2'],['A319'],['A321']]    
    reduced_type['model']=pd.DataFrame({'model':model})
    reduced_type=reduced_type.set_index('index')
    reduced_type.to_csv('model_lookup.csv')
    #average freqs for market carrier ac_Tye
    freqs_across_markets_carrier_ac_type = t100_summed[['UNIQUE_CARRIER','BI_MARKET','DAILY_FREQ','AIRCRAFT_TYPE']].groupby(['UNIQUE_CARRIER','BI_MARKET','AIRCRAFT_TYPE']).aggregate(np.mean).reset_index().groupby(['UNIQUE_CARRIER','AIRCRAFT_TYPE'])
    # construct F tabke
    rows=[]
    for airline in carriers_sorted:
        ac_types = list(set(t100_gb.get_group(airline)['AIRCRAFT_TYPE'].tolist()))
        for ac_type in ac_types:
            row={}            
            row['carrier'] = airline
            row['aircraft_type']=ac_type
            carrier_type = tr_gb.get_group((row['carrier'],row['aircraft_type']))
            rat=carrier_type['TIME_IN_NETWORK'].sum()/carrier_type['AIR_TIME'].sum()            
            #b43 model code
            model = reduced_type.loc[ac_type]['model']
            print(model) #should be a lists
            craft_data =b43[b43['CARRIER']==airline][b43['MODEL'].isin(model)]
            craft_data = craft_data[craft_data['OPERATING_STATUS']=='Y'] #get only operating craft
            craft_data = craft_data[craft_data['NUMBER_OF_SEATS']>0] #get only passenger  craft
            fleet_size = craft_data.shape[0]      
            craft_seats = craft_data['NUMBER_OF_SEATS'].mean()
            F=fleet_size*rat
            #get lower bound
            current_freqs_df=freqs_across_markets_carrier_ac_type.get_group((airline, ac_type))  
            freqs_by_market=[[record['BI_MARKET'],record['DAILY_FREQ']] for record in current_freqs_df.to_dict('records')]               
            carrier_df = t100ranked_gb.get_group(airline).set_index('BI_MARKET')            
            F_lower_bound = sum([2*f[1]*(45/60+carrier_df.loc[f[0]]['AOTP_FLIGHT_TIME']) for f in freqs_by_market if f[0] in carrier_df.index ])/18
            
            row['fleet_full'] = fleet_size
            row['craft_seats'] = craft_seats
            row['F_old'] = F
            row['F_lower_bound'] = F_lower_bound
            row['in_net_ratio'] =  rat
            rows.append(row)
            
    fleet_lookup =pd.DataFrame(rows)  
    fleet_lookup['below_bound'] = fleet_lookup.apply(lambda x: 1 if x['F_old'] < x['F_lower_bound'] else 0,1)
    fleet_lookup['F'] = fleet_lookup.apply(lambda x: max( x['F_old'], x['F_lower_bound']),1)
    fleet_lookup.to_csv(output_fn)
    return fleet_lookup     
'''
function to find most common type of plane used on each segment
and to get a fleet composition for network for each carrier

'''  
def fleet_assign(output_fn="fleetdist1.csv", fleet_lookup_fn = "fleet_lookup.csv",market_table_fn= "nonstop_competitive_markets.csv"):
    #load netowrk data file created by nonstop_market_profile function
    t100ranked = pd.read_csv(market_table_fn)
    #get carriers and markets under study
    markets =  t100ranked['BI_MARKET'].tolist()
    carriers = t100ranked['UNIQUE_CARRIER'].tolist()
    #load network data broken down by craft type
    t100_summed= pd.read_csv('t100_summed.csv')
    t100gb = t100_summed.groupby(['UNIQUE_CARRIER','BI_MARKET'])
    t100gb_carrier = t100_summed.groupby(['UNIQUE_CARRIER'])
    #load fleet data for relevant carriers on this network
    fleet_lookup= pd.read_csv(fleet_lookup_fn)
    fleet_lookup_gb = fleet_lookup.groupby(['carrier','aircraft_type'])   
    #loop through each carrier market combo found, catalogue the presence of aircraft types
    rows = []            
    i=0
    for market, carrier in zip(markets, carriers):
        i+=1
        row = {}
        group = t100gb.get_group((carrier, market))  
        #sum over market directions #NOTE: PERHAPS CHANGE SEATS TO OTHER METHOD IF LOOKING INCONGRUENT
        group =group[['BI_MARKET','UNIQUE_CARRIER','PASSENGERS','FLIGHT_TIME','FLIGHT_COST','SEATS','DAILY_FREQ','AIRCRAFT_TYPE']].groupby(['BI_MARKET','UNIQUE_CARRIER','AIRCRAFT_TYPE']).aggregate({'PASSENGERS':np.sum,'SEATS':np.sum,'DAILY_FREQ':np.sum, 'FLIGHT_TIME':np.mean,'FLIGHT_COST':np.mean}).reset_index()
        #total passengers across types
        totpax = group['PASSENGERS'].sum()
        #total frequency across types
        totfreq = group['DAILY_FREQ'].sum()
        #sort types according to frequency
        group_sort = group.sort(columns=['DAILY_FREQ'], axis=0,ascending=False)
        #get proportions of frequency and passengers each type accounts for 
        group_sort['PPAX']=group_sort['PASSENGERS']/totpax
        group_sort['PFREQ']=group_sort['DAILY_FREQ']/totfreq
        #get type with max freq and its associated data
        max_perc = group_sort['PFREQ'].iloc[0]
        max_seats = group_sort['SEATS'].iloc[0]
        max_pax = group_sort['PASSENGERS'].iloc[0]
        max_type=group_sort['AIRCRAFT_TYPE'].iloc[0]  
        #get the number of seats this aircraft has 
        group_sort['CRAFT_SEATS'] = group_sort.apply(lambda x: fleet_lookup_gb.get_group((carrier,x['AIRCRAFT_TYPE']))['craft_seats'] , axis=1)  
        group_sort['F']== group_sort.apply(lambda x: fleet_lookup_gb.get_group((carrier,x['AIRCRAFT_TYPE']))['F'] , axis=1)
        #create a dictionary of all types for this carrier/segment            
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
        #get seats for each craft        
        craft_seats = [(craft,fleet_lookup_gb.get_group((carrier,craft))['craft_seats']) for craft in craftlist]
        row['type_list'] = craft_seats
        rows.append(row)
        #construct data frame (MAY WISH TO PRESENT DATA IN A WAY THAT MAKES FOR EASIER ASsIGNEMENT  DECISIONS)
    fleet_dist = pd.DataFrame(rows)    
    fleet_dist.to_csv(output_fn, sep='\t')    
    return fleet_dist


'''
NOTE: now make segment craft assignments, by hand
'''   


'''
NOTE: BELOW ARE POSSIIBLY ALL DEPRECATED,  DELETE WHEN CONFIRMED 
'''
'''
  function to find number of seats in an assigned aircraft for a fleet distribution table. requires table returned by fleet_assign function
augmented with "assigned_type" column assigning an aircraft type to a segment-carrier combination
'''
'''
def seat_assign(fleet_dist_aug_fn='fleet_dist_aug.csv', b43_fn = "SCHEDULE_B43.csv"):
    #read augmented file
    aug_fleet = pd.read_csv(fleet_dist_aug_fn) 
    #get table of top craft for each carrier
    by_carrier=aug_fleet[['carrier','type_list']].groupby('carrier').aggregate(lambda x: x.iloc[0]).reset_index()
    #create dictionary that shows seats for a given carrier/craft
    seat_lookup = {}   
    
    #NOTE :PERHAPS CHANGE THIS METHOD GIVEN INTER CARRIER VARIATION
    
    for row in by_carrier.to_dict('records'):        
        seat_lookup[row['carrier']] = {rec[0]:rec[1] for rec in row['type_list']}
    #function to get number of seats for assigned type
    def assigned_seats(row):
        carr = row['carrier']
        plane_list=str(row['assigned_type']).split('-')
        seatlist  = [seat_lookup[carr][int(plane)] for plane in plane_list]
        return round(np.mean(seatlist))
    #create new column for this data
    aug_fleet['assigned_seats'] = aug_fleet.apply(assigned_seats, axis=1)


'''
'''
helper functio to find the number of seats for a particular aircraft 
THIS SHOULD BE REVISED IN LIGHT OF INTERCARRIER SEAT VARIATION 
AND ALSO IN LIGHT OF NEW SHORT NAME TO MODEL TABLE 
'''
'''
def find_seats(row, type1,b43):
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
    

#estimate size of an aircraft's fleet of a certain type from the total airtime of that fleet type
#PASS quarters VARIABLE

def fleet_size_by_airtime(df):
    hours_pday = df['AIR_HOURS']/(365/(4/len(quarters))) + (45/60)*df['DAILY_FREQ'] #add in average turn around time
    total_time = hours_pday.sum()
    #return estimated number of aircraft assuming airlines use as mcuh of fleet as possible (a questionable assumption?)
    return (total_time/18)

#PASS seat_lookup VARIABLE
def construct_Ftable():
    aug_fleet = pd.read_csv('fleet_dist_aug.csv')  
    t100_summed = pd.read_csv('t100_summed.csv')
    t100_gb = t100_summed.groupby(['UNIQUE_CARRIER','AIRCRAFT_TYPE'])
    airlines = list(set(aug_fleet['carrier'].tolist()))
    rows=[]
    for airline in airlines:
        ac_types= seat_lookup[airline].keys()
        for ac_type in ac_types:
            row={}
            select=(airline, ac_type)
            fleet=t100_gb.get_group(select)
            row['carrier'] = airline
            row['aircraft_type']=ac_type
            row['fleet_count'] = fleet_size_by_airtime(fleet)
            rows.append(row)
    fleet_lookup =pd.DataFrame(rows)      
    return fleet_lookup

#A FUNCTION TO CONVERT FROM SHORTNAME TO NUMBER PERHAPSS
def b43_get_type():
    pass 

#A FUNCTION TO GET TABLE OF "SHORT" NAMES, WHICH ARE APPARENTLY NOT INVENTORY MODEL NAMES, I SUPPOSE SUCH A CONVERSION TABLE WILL BE USEFUL LATER
def create_shortname_table():
    keys =sorted(list(set(fleet_lookup['aircraft_type'].tolist())))
    reduced_type = type1.set_index('AC_TYPEID').loc[keys].reset_index()[['index','SHORT_NAME']]
    #apparently corresponding model 
    model = [['SF-340/B'],['EMB-120'],['DASH8-Q4'],['DHC8-200'],['B737-7','B737-7/L'],['B737-8'],['B737-5'],['B737-4'],['B737-3'],['B757-2'],['B767-2'],['B767-3'],['B777-2'],['CRJ-2/4'],['RJ-700'],['B737-9'],['CRJ-900'],['MD-80'],['EMB-135'],['EMB-145'],['EMB-140'],['A320-1/2'],['A319'],['A321']]    
    reduced_type['model']=model
    #FIX
    


#TIME RATIO METHOD TO GET F, compare to above
def Ftable_new(output_fn="fleet_lookup.csv"):
    t100_all = pd.read_csv("t100_seg_all.csv")
    treduced=t100_all[t100_all['UNIQUE_CARRIER'].isin(carriers_sorted)]
    treduced['BI_MARKET']=treduced.apply(create_market,1)
    treduced['AIR_TIME']= treduced['AIR_TIME']/60
    treduced['IN_NETWORK'] = treduced.apply(lambda x: 1 if x['BI_MARKET'] in markets_sorted else 0, 1)
    treduced['TIME_IN_NETWORK'] = treduced['IN_NETWORK']*treduced['AIR_TIME']
    tr_gb = treduced.groupby(['UNIQUE_CARRIER','AIRCRAFT_TYPE'])
    in_net_ratio = []
    fleet_sizes =[]
    Fs=[]
    for row in fleet_lookup.to_dict('records'):
        try: #CHECK THISSS WHY ('XE', 674)
            carrier_type = tr_gb.get_group((row['carrier'],row['aircraft_type']))
            rat=carrier_type['TIME_IN_NETWORK'].sum()/carrier_type['AIR_TIME'].sum()
            in_net_ratio.append(rat)
            model=type1[type1['AC_TYPEID']==row['aircraft_type']]['SHORT_NAME'].iloc[0]    
            try:
                craft_data =b43[b43['CARRIER']==row['carrier']][b43['MODEL']==model]
            except IndexError:
                print(model)
                try:
                    craft_data=b43[b43['CARRIER']==row['carrier']][b43['MODEL']==model[:-2]]                    
                except IndexError:
                    print(model)
                    craft_data=b43[b43['CARRIER']==row['carrier']][b43['MODEL']==(model[:-2] + '/A')] 
            fleet_size = craft_data[craft_data['OPERATING_STATUS']=='Y'].shape[0]       
            fleet_sizes.append(fleet_size)
            Fs.append(fleet_size*rat)
        except KeyError:
            in_net_ratio.append(0)
            fleet_sizes.append(0)
            Fs.append(0)
    fleet_lookup['F']=Fs
    fleet_lookup['fleet_full']=fleet_sizes    
    fleet_lookup['in_net_ratio']=in_net_ratio 
    fleet_lookup.to_csv(output_fn)
'''









   

'''
function to create table of carrier and market data used by matlab myopic best response network game
#NOTE:CREATE AND OFFICIAL INDEX OF CARRIER MARKET COMBO FOR EASY MAPPING
ANOTHER NOTE:  COMMENT THS FUNCTION MORE EXTENSIVELY  
'''    
def create_network_game_datatable(t100ranked = "nonstop_competitive_markets.csv", fleet_lookup_fn = "fleet_lookup.csv",aotp_fn = 'aotp_mar.csv',fleet_dist_aug_fn='fleet_dist_aug.csv'):   
    fleet_lookup= pd.read_csv(fleet_lookup_fn)
    aug_fleet = pd.read_csv(fleet_dist_aug_fn) 
    #flgith times by airline market combo
    aotp_mar = pd.read_csv(aotp_fn)
    aotp_mar['BI_MARKET']=aotp_mar.apply(create_market,1) 
    #DISSAGREGGATE BY AIRCRAFT TYPE LATER
    aotp_mar_times = aotp_mar[['UNIQUE_CARRIER','BI_MARKET','AIR_TIME']].groupby(['UNIQUE_CARRIER','BI_MARKET']).aggregate(lambda x: np.mean(x)/60)
    aotp_mar_times = aotp_mar_times.reset_index().groupby(['UNIQUE_CARRIER','BI_MARKET'])
    
    with open('carrier_data.txt','w') as outfile:
        t100_gb_market = t100ranked.groupby('BI_MARKET')
        markets_sorted = sorted(list(set(t100ranked['BI_MARKET'].tolist())))
        num_mkts = len(markets_sorted)
        carriers_sorted = sorted(list(set(t100ranked['UNIQUE_CARRIER'].tolist())))
        num_carriers = len(carriers_sorted)
        outfile.write(str(num_carriers) + "\t" + str(num_mkts) + "\n")
        mkt_sizes = [str(t100_gb_market.get_group(mkt)['MARKET_COMPETITORS'].iloc[0]) for mkt in markets_sorted]
        mkt_sizes_str = "["+",".join(mkt_sizes)+"]"
        outfile.write(mkt_sizes_str + "\n")
        aug_fleet_gb_carrier = aug_fleet.groupby('carrier')
        for i, carrier in enumerate(carriers_sorted):
            carrier_data = t100ranked[t100ranked['UNIQUE_CARRIER']==carrier]
            carrier_markets_str = carrier_data['BI_MARKET'].tolist()
            fleet_assign=aug_fleet_gb_carrier.get_group(carrier).set_index('bimarket').loc[carrier_markets_str].reset_index()
            fleet_assign['bimarket']=fleet_assign['index']
            fleet_assign=fleet_assign.sort(columns=['bimarket'])
            ac_types = sorted(list(set(fleet_assign['assigned_type'].tolist())))
            fleet_assign_gb_type = fleet_assign.groupby('assigned_type')
            #build A matrix and b matrix
            A_rows = []
            b_rows = []
            for ac_type in ac_types:
                mkts_for_craft_df = fleet_assign_gb_type.get_group(ac_type)
                mkts_for_craft = mkts_for_craft_df['bimarket'].tolist()
                a_row = []
                #for each column of A matrix
                for mk in carrier_markets_str:
                    if mk in mkts_for_craft:
                        try:
                            block_hours=aotp_mar_times.get_group((carrier,mk))['AIR_TIME'].iloc[0]
                        except KeyError:
                            try:
                                aotp_mar_times_avg =aotp_mar[['UNIQUE_CARRIER','BI_MARKET','AIR_TIME']].groupby(['UNIQUE_CARRIER','BI_MARKET']).aggregate(lambda x: np.mean(x)/60)
                                aotp_mar_times_avg =aotp_mar_times_avg.reset_index().groupby(['BI_MARKET'])
                                block_hours=aotp_mar_times_avg.get_group(mk)['AIR_TIME'].iloc[0]
                            except KeyError:
                                mkk=mk.replace('ONT','LAX')
                                block_hours=aotp_mar_times_avg.get_group(mkk)['AIR_TIME'].iloc[0]    
                        a_row.append(2*(block_hours +45/60))
                    else:
                        a_row.append(0)
                A_rows.append(a_row)
                F = sum([fleet_lookup.groupby(['carrier','aircraft_type']).get_group((carrier,int(subtype)))['F'].iloc[0] for subtype in ac_type.split('-') ])
                b_rows.append(18*F)
            carrier_Markets = [markets_sorted.index(mk)+1 for mk in carrier_markets_str]
            carrier_freq_ind = carrier_data['MARKET_RANK'].tolist()
            #get coefficients, stacked in order of markets
            carrier_coef = []
            for record in carrier_data.to_dict('records'):
                Cold = 10000
                Cnew = record['FLIGHT_COST']
                Mold = 1000
                Mnew = record['MARKET_TOT']
                freq_ind = record['MARKET_RANK']
                if record['MARKET_COMPETITORS']==1:
                    base = [-95164.0447,-36238.3083,1148.0305]
                    transcoef = [-(Mnew/Mold)*base[0],(Mnew/Mold)*(Cold-base[1])-Cnew,-(Mnew/Mold)*base[2] ]
                elif record['MARKET_COMPETITORS']==2:
                    base = [-274960.0,-16470.0,	34936.0,	425.6,	-1300.0,	595.7]
                    transcoef = [-(Mnew/Mold)*base[0]] + [(Mnew/Mold)*(Cold-base[j])-Cnew if (i+1)==freq_ind else -(Mnew/Mold)*base[j] for i,j in enumerate(range(1,3))  ] + [-(Mnew/Mold)*base[i] for i in range(3,6)]
                elif record['MARKET_COMPETITORS']==3:
                    base=[-150395.5496,-10106.6470,13135.9798,13136.1506,264.4822,-376.1793,-376.1781,270.2080,270.1927,-260.0113]
                    transcoef = [-(Mnew/Mold)*base[0]] + [(Mnew/Mold)*(Cold-base[j])-Cnew if (i+1)==freq_ind else -(Mnew/Mold)*base[j] for i,j in enumerate(range(1,4))  ] + [-(Mnew/Mold)*base[i] for i in range(4,10)]
                else:
                    base=[-101456.3779,-5039.0076,6450.0318,6450.0511,6450.0624,134.9756,-137.7129,-137.7135,-137.7157,169.9196,169.9198,169.9212,-126.7018,-126.7025,-126.7034]    
                    transcoef = [-(Mnew/Mold)*base[0]] + [(Mnew/Mold)*(Cold-base[j])-Cnew if (i+1)==freq_ind else -(Mnew/Mold)*base[j] for i,j in enumerate(range(1,5))  ] + [-(Mnew/Mold)*base[i] for i in range(5,15)]
                carrier_coef += transcoef 
            #construct rowstring
            row_string = '['
            for a_row in A_rows:
                row_string+=",".join([str(num) for num in a_row])
                row_string+=";"
            row_string+=']'+'\t'+'['
            row_string+=",".join([str(num) for num in b_rows])
            row_string+=']'+'\t'+'['
            row_string+=",".join([str(num) for num in carrier_Markets])
            row_string+=']'+'\t'+'['
            row_string+=",".join([str(num) for num in carrier_freq_ind])
            row_string+=']'+'\t'+'['
            row_string+=",".join([str(num) for num in carrier_coef])
            row_string+=']'+'\n'
            outfile.write(row_string)


#NOTE:  MAKE THIS INTOA FUNCTION

#FUNCTION TO BUILD TABLE FROM NEWORK GAME RESULTS FROM MATLAB
def create_results_table():
    network_results_raw = pd.read_csv("matlab_2stagegames/network_results_revisedF.csv",header=None)
    network_results = t100ranked[['UNIQUE_CARRIER','BI_MARKET','MARKET_RANK','MARKET_COMPETITORS','DAILY_FREQ']]
    network_results['EST_FREQ'] = network_results_raw[2].tolist()
    results_market_grouped =network_results.groupby('BI_MARKET')
    MAPES=[]
    for mkt in markets_sorted:
        mkt_gb = results_market_grouped.get_group(mkt)
        fs = mkt_gb['DAILY_FREQ'].tolist()
        f_hats = mkt_gb['EST_FREQ'].tolist()
        mape = sum([abs(f_hat-f) for f_hat,f in zip(f_hats,fs)])/sum(fs)
        MAPES.append(mape)
    mape_column = []
    for competitors, mape in zip(mkt_sizes, MAPES):
        mape_column += np.repeat(mape,int(competitors)).tolist()
    network_results['MAPE'] = mape_column
    network_results.to_csv('network_MAPE_revisedF.csv',sep='\t')




