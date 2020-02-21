import pandas as pd, numpy as np, tensorflow as tf 

#D = pd.read_csv('data/hscls_explore_n_exploit_combined_v3.csv') 
#D = pd.read_csv('data/test_data_v3.csv') 

D = pd.read_csv('../data/hs_allcohort_maytoaug_11wks_cap8to9_data.csv') 

#hs_allcohort_maytoaug_2mil_data.csv') 

#hscls_explore_only_data_mar_to_aug_433krows.csv') 

#explore_only_more_buckets_1mmsequences_marchtoaugust.csv') 
#before_grad_clip_download_all_cohorts_maytoaug.csv') 

## drop trip_most_freq_city_id 

D = D.drop(columns=['cohort', 'targeting_model', 'randbucket', 'trip_most_freq_city_id', 'net_billings_usd', 'trip_most_freq_city_id', 'strategy_name']) 

### deal with null features, normalize data 
D.num_trips[pd.isnull(D.num_trips)] = 0.0 
D.gross_bookings_usd[pd.isnull(D.gross_bookings_usd)] = 0.0 
D.variable_contribution_usd[pd.isnull(D.variable_contribution_usd)] = 0.0 

li = ['churns_hard_lifetime', 
      'days_active_lifetime', 
      'days_since_trip_first_lifetime', 
      'fare_lifetime', 
      'days_active_84d', 
      'trip_pool_matched_avg_84d', 
      'fare_promo_total_avg_84d', 
      'fare_total_avg_84d', 
      'ata_trip_max_avg_84d', 
      'eta_trip_max_avg_84d', 
      'rating_2rider_total_avg_84d', 
      'surge_trip_avg_84d', 
      'fare_total_win7d_potential_84d', 
      'trip_complete_win7d_potential_84d', 
      'trip_total_win7d_potential_84d', 
      'fare_total_win28d_potential_84d', 
      'trip_complete_win28d_potential_84d', 
      'trip_total_win28d_potential_84d'] 

for l in li: 
    print('number of nans: ' + str(sum(pd.isnull(D[l]))))
    D[l] = pd.to_numeric(D[l], errors='coerce') 
    D[l] = D[l] - D[l].mean() 
    D[l] = D[l] / D[l].std() 
    D[l][pd.isnull(D[l])] = 0.0 ## at zero mean due to standard normalization 

label_list = [ 
    'num_trips', 
    'gross_bookings_usd', 
    'variable_contribution_usd' 
] 

for l in label_list: 
    D[l] = pd.to_numeric(D[l], errors='coerce') 

#print(D['is_treatment'] == True) 
#print(D['datestr'] > '2019-07-15') 
### compute cpit approximately for 3 weeks 
treated_entries = D[D['is_treatment'] == True] # & D['datestr'] > '2019-07-15'] 
untreated_entries = D[D['is_treatment'] == False] # & D['datestr'] > '2019-07-15'] 

rpu_treated = treated_entries['num_trips'].sum() / len(treated_entries) 
nipu_treated = treated_entries['variable_contribution_usd'].sum() / len(treated_entries) 

rpu_untreated = untreated_entries['num_trips'].sum() / len(untreated_entries) 
nipu_untreated = untreated_entries['variable_contribution_usd'].sum() / len(untreated_entries) 

cpit = -1.0 * (nipu_treated - nipu_untreated) / (rpu_treated - rpu_untreated) 

print('rpu_treated : ' + str(rpu_treated)) 
print('nipu_treated : ' + str(nipu_treated)) 
print('rpu_untreated : ' + str(rpu_untreated)) 
print('nipu_treated : ' + str(nipu_untreated)) 
print('cpit : ' + str(cpit)) 

### reshape them into 3-dimensional tensors with each user as a sequence of features 
dgrouped = D.groupby(['rider_uuid']) 

datedict = { 
    '2019-05-27':0, 
    '2019-06-03':1, 
    '2019-06-10':2, 
    '2019-06-17':3, 
    '2019-06-24':4, 
    '2019-07-01':5, 
    '2019-07-08':6, 
    '2019-07-15':7, 
    '2019-07-22':8, 
    '2019-07-29':9, 
    '2019-08-05':10 
} 

assert(len(datedict) == 11) 

"""
"""
    #j.is_treatment, 
    #j.churns_hard_lifetime, 
    #j.days_active_lifetime, 
    #j.days_since_trip_first_lifetime, 
    #j.fare_lifetime, 
    #j.days_active_84d, 
    #j.trip_pool_matched_avg_84d, 
    #j.fare_promo_total_avg_84d, 
    #j.fare_total_avg_84d, 
    #j.ata_trip_max_avg_84d, 
    #j.eta_trip_max_avg_84d, 
    #j.rating_2rider_total_avg_84d, 
    #j.surge_trip_avg_84d, 
    #j.fare_total_win7d_potential_84d, 
    #j.trip_complete_win7d_potential_84d, 
    #j.trip_total_win7d_potential_84d, 
    #j.fare_total_win28d_potential_84d, 
    #j.trip_complete_win28d_potential_84d, 
    #j.trip_total_win28d_potential_84d, 
    
    #l.num_trips, 
    #l.gross_bookings_usd, 
    #l.variable_contribution_usd, 

def user_sequence_process_func(dataframe): 
    ## this function should process all data into tensor format 
    
    ##### dataframe column names: 
    
    #j.datestr as datestr, [index this] 
    #j.rider_uuid as rider_uuid, 
    
    #j.churns_hard_lifetime, 
    #j.days_active_lifetime, 
    #j.days_since_trip_first_lifetime, 
    #j.fare_lifetime, 
    #j.days_active_84d, 
    #j.trip_pool_matched_avg_84d, 
    #j.fare_promo_total_avg_84d, 
    #j.fare_total_avg_84d, 
    #j.ata_trip_max_avg_84d, 
    #j.eta_trip_max_avg_84d, 
    #j.rating_2rider_total_avg_84d, 
    #j.surge_trip_avg_84d, 
    #j.fare_total_win7d_potential_84d, 
    #j.trip_complete_win7d_potential_84d, 
    #j.trip_total_win7d_potential_84d, 
    #j.fare_total_win28d_potential_84d, 
    #j.trip_complete_win28d_potential_84d, 
    #j.trip_total_win28d_potential_84d, 
    
    #j.is_treatment, 
    
    #l.num_trips, 
    #l.gross_bookings_usd, 
    #l.variable_contribution_usd, 
    
    ## create a tensor for this user 11 (wks) x 28 dimensions 
    ## in 28 dimensions 18 are features, 4 are past week labels, 4 are this week labels 
    
    num_weeks = 11 
    num_dimensions = 26 + 2 
    ## added to prev time step and cur time step, labels for missing data 
    
    tens = np.zeros((num_weeks, num_dimensions)) 
    
    ## replace treatment with 1s and 0s 
    dataframe['is_treatment'] = dataframe['is_treatment'].apply(lambda x: 1.0 if x == True else 0.0) 
    
    feature_list = [ 
        'churns_hard_lifetime', 
        'days_active_lifetime', 
        'days_since_trip_first_lifetime', 
        'fare_lifetime', 
        'days_active_84d', 
        'trip_pool_matched_avg_84d', 
        'fare_promo_total_avg_84d', 
        'fare_total_avg_84d', 
        'ata_trip_max_avg_84d', 
        'eta_trip_max_avg_84d', 
        'rating_2rider_total_avg_84d', 
        'surge_trip_avg_84d', 
        'fare_total_win7d_potential_84d', 
        'trip_complete_win7d_potential_84d', 
        'trip_total_win7d_potential_84d', 
        'fare_total_win28d_potential_84d', 
        'trip_complete_win28d_potential_84d', 
        'trip_total_win28d_potential_84d' 
    ] 
    
    label_list = [ 
        'is_treatment', 
        'num_trips', 
        'gross_bookings_usd', 
        'variable_contribution_usd'        
    ] 
    
    num_features = len(feature_list) 
    num_labels = len(label_list) + 1 ## one for existence of data 
    
    ## sort by the dates 
    ## fill in missing dates and fill missing data with zeros, (features and labels) 
    ## second point fulfilled by initializing with np.zeros 
    
    minidx = num_weeks - 1 
    maxidx = 0 
    for indx, d in dataframe.iterrows(): 
        if d['datestr'] not in datedict: 
            break 
        idx = datedict[d['datestr']] 
        if idx < minidx: 
            minidx = idx 
        if idx > maxidx: 
            maxidx = idx 
        tens[idx, 0:num_features] = d[feature_list].as_matrix() 
        tens[idx, num_features + num_labels:-1] = d[label_list].as_matrix() 
        tens[idx, -1] = 1.0 # existence of data 
    
    ## fill in the treatment label of previous iteration in next time step 
    tens[1:, num_features:num_features + num_labels] = tens[0:-1, num_features + num_labels:] 
    
    ## lens: first dimension is start of sequence, second is the length of sequence 
    lens = [minidx, maxidx + 1]     
    return tens, lens 

dataset_size = len(dgrouped) 

num_weeks = 11 
num_tensor_dims = 26 + 2

datastore = np.zeros((dataset_size, num_weeks, num_tensor_dims)) 
seqlenstore = np.zeros((dataset_size, 2)) 

i = 0 ## index 
for g in dgrouped: 
    #print(g) 
    #print(type(g)) 
    df = g[1] 
    
    trsarray, lens = user_sequence_process_func(df) 
    #print('------------- after processing -------------------') 
    #print(trsarray) 
    #print(lens) 
    datastore[i, :, :]  = np.reshape(trsarray, (1, num_weeks, num_tensor_dims)) 
    seqlenstore[i, :] = np.reshape(lens, (1, 2)) 
    
    i = i + 1 
    #print(g['datestr']) 
    
    if i % 100 == 0: 
        print('processed: ' + str(i) + '/' + str(dataset_size) + ' records')
    
    if i == dataset_size: 
        break 
    #print(user_sequence_process_func(g)) 

print(datastore) 
print(seqlenstore) 

## the entire code takes 8 hours to process 957702 user records 

import pickle as pkl 

datasavefilename = '../data/hscls_seqdata_size' + str(dataset_size) + '_allcohorts11weekscap8to9_save.pkl' 
saveD = {} 
saveD['datastore'] = datastore 
saveD['seqlenstore'] = seqlenstore 
pkl.dump(saveD, open(datasavefilename, 'w')) 

