import os, sys, datetime, numpy as np, pandas as pd, pickle as pkl 

D = pd.read_csv('/Users/will.zou/code/deeplearning_hscls/ponfare_3rd_place/model/write_xgb_train_df_data.csv')

Daug = pd.read_csv('/Users/will.zou/code/deeplearning_hscls/ponfare_3rd_place/model/augment_coupons_clabels.csv') 
col_list = ['COUPON_ID', 'EST_COST', 'TREATMENT_RATE'] 
Daug = Daug[col_list] 

print('starting to merge left... ') 
D = D.merge(Daug, on=['COUPON_ID'], how='left') 

## ensure spend only occurs when conversion happens for coupon 
D['ORIG_PRICE'] = D['EST_COST'] / D['TREATMENT_RATE'] 
D['DISCOUNT_AMOUNT'] = D['EST_COST'] 
D['EST_COST'] = D['EST_COST'] *  D['y'] 
D['DISCOUNT_AMOUNT'] = D['ORIG_PRICE'] * D['TREATMENT_RATE']

print('number of rows in dataset before subsample: ' + str(len(D))) 

### note integer column names for no-header direct csv reads 
D0 = D[D['y'] == 0] ##
D1 = D[D['y'] == 1] ## 

### sample approx. 0.04 * 5 mil = 200k negative cases 
D0 = D0.sample(frac = 0.04)
import ipdb; ipdb.set_trace() 

D = pd.concat([D0, D1])

D = D.sample(frac=1.0) 

print('number of rows after subsample: ' + str(len(D))) 

print(D) 
D.to_csv('/Users/will.zou/code/deeplearning_hscls/data/ponfare_kaggle_data_v2.csv') 


