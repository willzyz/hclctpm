import pandas as pd, numpy as np, tensorflow as tf 
import pickle as pkl 

### ---------- US census public dataset settings: ---------- 
prefix = 'uscensus_pub' 
D = pd.read_csv('../data/USCensus1990.data.txt') 

#import ipdb; ipdb.set_trace() 
## caseid,dAge,dAncstry1,dAncstry2,iAvail,iCitizen,iClass,dDepart,iDisabl1,iDisabl2,iEnglish,iFeb55,iFertil,dHispanic,dHour89,dHours,iImmigr,dIncome1,dIncome2,dIncome3,dIncome4,dIncome5,dIncome6,dIncome7,dIncome8,dIndustry,iKorean,iLang1,iLooking,iMarital,iMay75880,iMeans,iMilitary,iMobility,iMobillim,dOccup,iOthrserv,iPerscare,dPOB,dPoverty,dPwgt1,iRagechld,dRearning,iRelat1,iRelat2,iRemplpar,iRiders,iRlabor,iRownchld,dRpincome,iRPOB,iRrelchld,iRspouse,iRvetserv,iSchool,iSept80,iSex,iSubfam1,iSubfam2,iTmpabsnt,dTravtime,iVietnam,dWeek89,iWork89,iWorklwk,iWWII,iYearsch,iYearwrk,dYrsserv 

#Dt = Dt.sample(frac= len(Dc) * 2.0 / len(Dt)) 
#D = pd.concat([D1, D2]) 
#e_median = np.median(D[0].values) 
#D = D[D[0] > e_median] 
#D = D.sample(frac=1.0) 

cohort_column_name = 'dHour89'
treatment_indicator_value = 1.0 
control_indicator_value = 0.0 

#import ipdb; ipdb.set_trace() 
#w_mean = np.mean(D[cohort_column_name].values) 
D = D[D['iFertil'] > 1.5] ## screen out n/a items 
#D = D[D['iFertil'] < 4] ## screen out n/a items 
D = D[D['dAge'] < 5] ## choose people less than 50 years old 
D = D[D['iCitizen'] == 0] 

w_median = np.median(D[cohort_column_name].values) 
D[cohort_column_name] = D[cohort_column_name].apply(lambda x: 1.0 if x > w_median else 0.0) 

## feature take out 4 due to vert. hydro 
feature_list_a = [ 
    #'dAge',
    #'dAncstry1',
    #'dAncstry2',
    'iAvail',
    'iCitizen',
    'iClass',
    'dDepart',
    'iDisabl1',
    'iDisabl2',
    'iEnglish',
    'iFeb55',
    #'dHispanic',
    #'iImmigr',    
    'iKorean',
    'iLang1',
    'iLooking',
    #'iMarital',
    'iMay75880',
    'iMeans',
    'iMobility',
    'iMobillim',
    'iOthrserv',
    'iPerscare',
    'dPOB',
    'dPwgt1',
    #'iRagechld',
    'dRearning',
    'iRelat1',
    'iRelat2',
    'iRemplpar',
    'iRiders',
    'iRlabor',
    #'iRownchld',
    'iRPOB',
    #'iRrelchld',
    #'iRspouse',
    'iRvetserv',
    'iSchool',
    'iSept80',
    'iSex',
    'iSubfam1',
    'iSubfam2',
    'iTmpabsnt',
    'dTravtime',
    'dWeek89',
    'iYearsch'
] 

feature_list_b = [
    'dIndustry', 
    'iMilitary',
    'dOccup',
    'dPoverty',
    'iVietnam',
    'iWork89',
    'iWorklwk',
    'iWWII',
    'iYearwrk',
    'dYrsserv' 
] 

#iFertil, 
#dHour89,dHours, 
#dIncome1,dIncome2,dIncome3,dIncome4,dIncome5,dIncome6,dIncome7,dIncome8,dRpincome, 

label_list = [ 
    'dIncome1', 
    'iFertil', 
    'dHour89', 
    'dAge' 
] 

o_median = np.median(D[label_list[0]].values) 
D[label_list[0]] = D[label_list[0]].apply(lambda x: float(x)) #(lambda x: -1.0 * x / 100.0) # the reward is near wild fire starting points in hundres of meters 
D[label_list[1]] = D[label_list[1]].apply(lambda x: -1.0 * (x - 2.0)) #(lambda x: 100.0 if x == 1 else 0.0) ## the cost is pine vs fir 


"""
### ---------- covertype public dataset settings: ---------- 
prefix = 'covtype_pub' 
D = pd.read_csv('../data/covtype.data', header=None) 

### note integer column names for no-header direct csv reads 
D1 = D[D[54] == 1] ## Spruce-Fir 
D2 = D[D[54] == 2] ## Lodgepole Pine 

#Dt = Dt.sample(frac= len(Dc) * 2.0 / len(Dt)) 
D = pd.concat([D1, D2]) 
e_median = np.median(D[0].values) 
D = D[D[0] > e_median] 
D = D.sample(frac=1.0) 

cohort_column_name = 3 ## column name for distance to hydrology (meters) 
treatment_indicator_value = 1.0 
control_indicator_value = 0.0 

w_median = np.median(D[cohort_column_name].values) 
D[cohort_column_name] = D[cohort_column_name].apply(lambda x: 1.0 if x < w_median else 0.0) 

## feature take out 4 due to vert. hydro 
feature_list = [ i for i in range(3)] + [i for i in range(5, 9)] + [i for i in range(10, 54)] 

label_list = [ 
    9, ## distance to wild fire ignition points 
    54 ## Pine (2) vs Fir (1) 
    #3 ## distance to hydrology (meters) 
] 
import ipdb; ipdb.set_trace()
o_median = np.median(D[label_list[0]].values)
D[label_list[0]] = D[label_list[0]].apply(lambda x: 1.0 if x < o_median else 0.0) #(lambda x: -1.0 * x / 100.0) # the reward is near wild fire starting points in hundres of meters 
D[label_list[1]] = D[label_list[1]].apply(lambda x: 1.0 if x == 1 else 0.0) #(lambda x: 100.0 if x == 1 else 0.0) ## the cost is pine vs fir 

print(feature_list) 
"""
""" 
### ---------- subscriptions upsell settings: ---------- 
D = pd.read_csv('../data/subs_upsell_short_v2.csv') 

Dt = D[D['cohort'] == 'treatment'] 
Dc = D[D['cohort'] == 'control'] 

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
""" 

# ----------- the code below are generic to all use cases --------------- 

print(' --- compute simple statistics and cpit --- ') 
### compute cpit 
treated_entries = D[D[cohort_column_name] == treatment_indicator_value] 
untreated_entries = D[D[cohort_column_name] == control_indicator_value] 

rpu_treated = float(treated_entries[label_list[0]].sum()) / len(treated_entries) 
cipu_treated = float(treated_entries[label_list[1]].sum()) / len(treated_entries) 

rpu_untreated = float(untreated_entries[label_list[0]].sum()) / len(untreated_entries) 
cipu_untreated = float(untreated_entries[label_list[1]].sum()) / len(untreated_entries) 

cpit = (cipu_treated - cipu_untreated) / (rpu_treated - rpu_untreated) 

for l in feature_list_a: 
    print('number of nans: ' + str(sum(D[l] == '\N'))) 
    D[l] = pd.to_numeric(D[l], errors='coerce') 
    D[l] = D[l] - D[l].mean() 
    D[l] = D[l] / D[l].std() 
    D[l][pd.isnull(D[l])] = 0.0 ## at zero mean due to standard normalization 

for l in feature_list_b: 
    print('number of nans: ' + str(sum(D[l] == '\N'))) 
    D[l] = pd.to_numeric(D[l], errors='coerce') 
    D[l] = D[l] - D[l].mean() 
    D[l] = D[l] / D[l].std() 
    D[l][pd.isnull(D[l])] = 0.0 ## at zero mean due to standard normalization 

for l in label_list: 
    D[l] = pd.to_numeric(D[l], errors='coerce') 
    D[l][pd.isnull(D[l])] = 0.0 ## at zero mean due to standard normalization 

print('rpu_treated : ' + str(rpu_treated)) 
print('cipu_treated : ' + str(cipu_treated)) 
print('rpu_untreated : ' + str(rpu_untreated)) 
print('cipu_untreated : ' + str(cipu_untreated)) 
print('cpit : ' + str(cpit)) 

### save the data to disk 
len_tr = len(D) / 5 * 3 
len_va = len(D) / 5 

nX_a = D[feature_list_a].as_matrix() 
nX_b = D[feature_list_b].as_matrix() 
w = D[cohort_column_name].apply(lambda x: 1.0 if x == treatment_indicator_value else 0.0) 
w = w.as_matrix() 
values = D[label_list[0]] 
values = values.as_matrix() 
negcost = D[label_list[1]] 
negcost = negcost.as_matrix() * -1.0 
treatintensity = D[label_list[2]].as_matrix() * 1.0 
d2dist = D[label_list[3]].as_matrix() * 1.0 

## split train/val/test sets 

nX_a_tr = nX_a[0:len_tr, :] 
nX_a_va = nX_a[len_tr:len_tr + len_va, :] 
nX_a_te = nX_a[len_tr + len_va:, :] 

nX_b_tr = nX_b[0:len_tr, :] 
nX_b_va = nX_b[len_tr:len_tr + len_va, :] 
nX_b_te = nX_b[len_tr + len_va:, :] 

w_tr = w[0:len_tr]
w_va = w[len_tr:len_tr + len_va] 
w_te = w[len_tr + len_va:] 

values_tr = values[0:len_tr] 
values_va = values[len_tr:len_tr + len_va] 
values_te = values[len_tr + len_va:] 

#avg_ni_usd_tr = avg_ni_usd[0:len_tr] 
negcost_tr = negcost[0:len_tr] 
treatintensity_tr = treatintensity[0:len_tr] 
d2dist_tr = d2dist[0:len_tr] 

#avg_ni_usd_va = avg_ni_usd[len_tr:len_tr + len_va] 
negcost_va = negcost[len_tr:len_tr + len_va] 
treatintensity_va = treatintensity[len_tr:len_tr + len_va] 
d2dist_va = d2dist[len_tr:len_tr + len_va] 

#avg_ni_usd_te = avg_ni_usd[len_tr + len_va:] 
negcost_te = negcost[len_tr + len_va:] 
treatintensity_te = treatintensity[len_tr + len_va:] 
d2dist_te = d2dist[len_tr + len_va:] 

## saving data using cPickel and naming the dictionaries 
saveD = {'nX_a_tr':nX_a_tr, 
         'nX_b_tr':nX_b_tr, 
         'w_tr':w_tr, 
         'values_tr':values_tr, 
         'nX_a_va':nX_a_va, 
         'nX_b_va':nX_b_va, 
         'w_va':w_va, 
         'values_va':values_va, 
         'nX_a_te':nX_a_te, 
         'nX_b_te':nX_b_te, 
         'w_te':w_te, 
         'values_te':values_te, 
         'feature_list_a':feature_list_a, 
         'feature_list_b':feature_list_b, 
         'negcost_tr': negcost_tr, 
         'negcost_va': negcost_va, 
         'negcost_te': negcost_te, 
         'treatintensity_tr': treatintensity_tr, 
         'treatintensity_va': treatintensity_va, 
         'treatintensity_te': treatintensity_te, 
         'd2dist_tr': d2dist_tr, 
         'd2dist_va': d2dist_va, 
         'd2dist_te': d2dist_te
         } 
pkl.dump(saveD, open('../data/' + str(prefix) + '_causal_data_intensity_matching_d2dist', 'w')) 
