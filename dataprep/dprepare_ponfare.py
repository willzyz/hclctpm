import pandas as pd, numpy as np, tensorflow as tf 
import pickle as pkl 

### ---------- covertype public dataset settings: ---------- 
prefix = 'ponfare_v3' 
D = pd.read_csv('../data/ponfare_kaggle_data_v3_young.csv') 

## ensure spend only occurs when conversion happens for coupon 
#D['EST_COST'] = D['EST_COST'] *  D['y'] 

### note integer column names for no-header direct csv reads 
#D0 = D[D['y'] == 0] ## 
#D1 = D[D['y'] == 1] ## 

### sample approx. 0.04 * 5 mil = 200k negative cases 
#D0 = D0.sample(frac = 0.04) 
#D = pd.concat([D0, D1]) 

D = D.sample(frac=1.0) 

### use treatment simulation first for TR > 54%, median 
D['TREATMENT'] = D['TREATMENT_RATE'] #D['DISCOUNT_AMOUNT'] #
cohort_column_name = 'TREATMENT' ## column name for distance to hydrology (meters) 
treatment_indicator_value = 1.0 
control_indicator_value = 0.0 

w_median = np.median(D[cohort_column_name].values) 
D[cohort_column_name] = D[cohort_column_name].apply(lambda x: 1.0 if x > w_median else 0.0) 

label_list = ['y', 'EST_COST', 'TREATMENT_RATE', 'd2'] 

feature_list = D.keys()[6:-6].tolist() 

## 6:141 is user feature, 141:-6 can be seen as coupon features 

print('len before remove: '+str(len(feature_list))) 
feature_list.remove('pb_same_genreprice') 
feature_list.remove('pb_same_v_genreprice') 
feature_list.remove('zprice') 
print('len after remove: '+str(len(feature_list))) 
print(feature_list) 

# ----------- the code below are generic to all use cases --------------- 

print(' --- compute simple statistics and cpit --- ') 
### compute cpit 
treated_entries = D[D[cohort_column_name] == treatment_indicator_value] 
untreated_entries = D[D[cohort_column_name] == control_indicator_value] 

rpu_treated = float(treated_entries[label_list[0]].sum()) / len(treated_entries) # len(treated_entries['COUPON_ID'].unique()) #
cipu_treated = float(treated_entries[label_list[1]].sum()) / len(treated_entries) #len(treated_entries['COUPON_ID'].unique()) 

rpu_untreated = float(untreated_entries[label_list[0]].sum()) / len(untreated_entries) #len(untreated_entries['COUPON_ID'].unique()) # 
cipu_untreated = float(untreated_entries[label_list[1]].sum()) / len(untreated_entries) #len(untreated_entries['COUPON_ID'].unique()) # 

cpit = (cipu_treated - cipu_untreated) / (rpu_treated - rpu_untreated) 

print('rpu_treated : ' + str(rpu_treated)) 
print('cipu_treated : ' + str(cipu_treated)) 
print('rpu_untreated : ' + str(rpu_untreated)) 
print('cipu_untreated : ' + str(cipu_untreated)) 
print('cpit : ' + str(cpit)) 

for l in feature_list: 
    print('number of nans: ' + str(sum(D[l].isnull()))) 
    D[l] = pd.to_numeric(D[l], errors='coerce') 
    D[l] = D[l] - D[l].mean() 
    D[l] = D[l] / D[l].std() 
    D[l][pd.isnull(D[l])] = 0.0 ## at zero mean due to standard normalization 

for l in label_list: 
    D[l] = pd.to_numeric(D[l], errors='coerce') 
    D[l][pd.isnull(D[l])] = 0.0 ## at zero mean due to standard normalization 

### save the data to disk 
len_tr = int(len(D) / 5 * 3) 
len_va = int(len(D) / 5) 

nX = D[feature_list].as_matrix() 
w = D[cohort_column_name].apply(lambda x: 1.0 if x == treatment_indicator_value else 0.0) 
w = w.as_matrix() 
values = D[label_list[0]] 
values = values.as_matrix() 
negcost = D[label_list[1]] 
negcost = negcost.as_matrix() * -1.0 
treatintensity = D[label_list[2]].as_matrix() * 1.0 
d2dist = D[label_list[3]].as_matrix() * 1.0 

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
pkl.dump(saveD, open('../data/' + str(prefix) + '_causal_data_with_intensity_d2dist', 'w')) 
