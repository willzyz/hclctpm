import pandas as pd, numpy as np, tensorflow as tf 
import pickle as pkl 

D = pd.read_csv('../data/subs_upsell_short_v2.csv') 

Dt = D[D['cohort'] == 'treatment'] 
Dc = D[D['cohort'] == 'control'] 

import ipdb; ipdb.set_trace() 

Dt = Dt.sample(frac= len(Dc) * 2.0 / len(Dt)) 
D = pd.concat([Dt, Dc]) 
D = D.sample(frac=1.0) 

feature_list = [ 
        'trip_complete_84d',
        'trip_complete_per_days_active_84d',
        'promo_used_84d',
        'trip_x_prc_84d',
        'trip_pool_prc_84d',
        'trip_pool_per_x_84d',
        'session_per_days_active_84d',
        'session_request_prc_84d',
        'session_background_pre_request_prc_84d',
        'has_session_request_84d',
        'duration_session_outside_total_prc_84d',
        'has_session_without_request_84d',
        'payment_cash_trip_prc_84d',
        'surge_trip_prc_84d',
        'ufp_trip_not_honored_prc_84d',
        'ufp_trip_total_prc_84d',
        'trip_promo_prc_84d',
        'trip_complete_prc_84d',
        'trip_rider_cancelled_prc_84d',
        'trip_driver_cancelled_prc_84d',
        'request_to_trip_prc_84d',
        'days_session_request_prc_84d',
        'trips_lifetime',
        'trip_complete_win7d_potential_84d',
        'days_since_trip_first_lifetime',
        'trip_complete_win28d_potential_84d',
        'fare_total_win7d_potential_84d',
        'trip_total_total_84d',
        'fare_total_win28d_potential_84d',
        'days_since_last_soft_churn_lifetime',
        'days_active_84d',
        'days_since_last_hard_churn_lifetime', 
        'session_lt_1m_prc_84d', 
        'fare_max_p50_84d', 
        'uber_preferred_score' 
] 

label_list = [ 
    'buy_pass_15d', 
    #'buy_pass_15d_all', 
    'total_trip_dropoff' 
] 

for l in feature_list: 
    print('number of nans: ' + str(sum(D[l] == '\N'))) 
    D[l] = pd.to_numeric(D[l], errors='coerce') 
    D[l] = D[l] - D[l].mean() 
    D[l] = D[l] / D[l].std() 
    D[l][pd.isnull(D[l])] = 0.0 ## at zero mean due to standard normalization 

for l in label_list: 
    D[l] = pd.to_numeric(D[l], errors='coerce') 
    D[l][pd.isnull(D[l])] = 0.0 ## at zero mean due to standard normalization 

### compute cpit approximately for 3 weeks 
treated_entries = D[D['cohort'] == 'treatment'] 
untreated_entries = D[D['cohort'] == 'control'] 

rpu_treated = float(treated_entries[label_list[0]].sum()) / len(treated_entries) 
nipu_treated = float(treated_entries[label_list[1]].sum()) / len(treated_entries) 

rpu_untreated = float(untreated_entries[label_list[0]].sum()) / len(untreated_entries) 
nipu_untreated = float(untreated_entries[label_list[1]].sum()) / len(untreated_entries) 

cpit = (nipu_treated - nipu_untreated) / (rpu_treated - rpu_untreated) 

print('rpu_treated : ' + str(rpu_treated)) 
print('nipu_treated : ' + str(nipu_treated)) 
print('rpu_untreated : ' + str(rpu_untreated)) 
print('nipu_untreated : ' + str(nipu_untreated)) 
print('cpit : ' + str(cpit)) 

### save the data to disk 
len_tr = len(D) / 5 * 3 
len_va = len(D) / 5 

nX = D[feature_list].as_matrix() 
w = D['cohort'].apply(lambda x: 1.0 if x == 'treatment' else 0.0) 
w = w.as_matrix() 
values = D[label_list[0]] 
values = values.as_matrix() 
negcost = D[label_list[1]] 
negcost = negcost.as_matrix() * -1.0 

## split train/val/test sets 

nX_tr = nX[0:len_tr, :] 
nX_va = nX[len_tr:len_tr + len_va, :] 
nX_te = nX[len_tr + len_va:, :] 

w_tr = w[0:len_tr]
w_va = w[len_tr:len_tr + len_va] 
w_te = w[len_tr + len_va:] 

values_tr = values[0:len_tr] 
values_va = values[len_tr:len_tr + len_va] 
values_te = values[len_tr + len_va:] 

#avg_ni_usd_tr = avg_ni_usd[0:len_tr] 
negcost_tr = negcost[0:len_tr] 

#avg_ni_usd_va = avg_ni_usd[len_tr:len_tr + len_va] 
negcost_va = negcost[len_tr:len_tr + len_va] 

#avg_ni_usd_te = avg_ni_usd[len_tr + len_va:] 
negcost_te = negcost[len_tr + len_va:] 

## saving data using cPickel and naming the dictionaries 
saveD = {'nX_tr':nX_tr, 
         'w_tr':w_tr, 
         'values_tr':values_tr, 
         'nX_va':nX_va, 
         'w_va':w_va, 
         'values_va':values_va, 
         'nX_te':nX_te, 
         'w_te':w_te, 
         'values_te':values_te, 
         'feature_list':feature_list, 
         #'avg_ni_usd_tr':avg_ni_usd_tr, 
         'negcost_tr': negcost_tr, 
         #'avg_ni_usd_va':avg_ni_usd_va, 
         'negcost_va': negcost_va, 
         #'avg_ni_usd_te':avg_ni_usd_te, 
         'negcost_te': negcost_te 
         } 
pkl.dump(saveD, open('subs_upsell_ma_training_data_v2', 'w')) 

""" 
## the entire code takes 8 hours to process 957702 user records 

import pickle as pkl 

datasavefilename = '../data/hscls_seqdata_size' + str(dataset_size) + '_allcohorts2mm11weeks_save.pkl' 
saveD = {} 
saveD['datastore'] = datastore 
saveD['seqlenstore'] = seqlenstore 
pkl.dump(saveD, open(datasavefilename, 'w')) 
""" 
