#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd, numpy as np, pickle as pkl 
import sys, os 
sys.path.append('../dataprep/') 
from QueryFunctions import * 
import tensorflow as tf 

# In[2]:

use_python3 = True 
use_query = 0 ### turn this switch on (to 1.0/True vs 0.0/False) to run query using QueryRunner 

if use_query: 
    from queryrunner_client import Client 
    qr = Client(user_email='will.zou@uber.com') 
    label_dates_weekly = "'2019-07-14', '2019-07-21', '2019-07-28', '2019-08-04'" 
    city_ids = '1,5,6,8,10,12,20,23,198' 
    feature_dates="'2019-06-30'" 
    proposal_start_date = '2019-07-08' 

    predFrame2 = qr.execute('presto', rxgy_data_sapphire_presto_featuremod2(label_dates_weekly, feature_dates,  city_ids)) 
    predFrame2 = pd.DataFrame(predFrame2.load_data()) 
else: 
    predFrame2 = pd.read_csv('../data/rxgy_query_result.csv') 

""" 
label_dates_weekly = "'2019-07-07','2019-06-30','2019-06-23','2019-06-16'"
city_ids = '1,5,6,8,10,12,20,23,198' 
feature_dates="'2019-06-02'" 
proposal_start_date = '2019-06-10' 

predFrame = qr.execute('presto', rxgy_data_sapphire_presto_featuremod2(label_dates_weekly, feature_dates,  city_ids)) 
predFrame = pd.DataFrame(predFrame.load_data()) 
""" 

""" 
label_dates_weekly = "'2019-08-11', '2019-08-18', '2019-08-25', '2019-09-01'" 
city_ids = '1,5,6,8,10,12,20,23,198' 
feature_dates="'2019-07-28'" 
proposal_start_date = '2019-08-05' 

predFrame3 = qr.execute('presto', rxgy_data_sapphire_presto_featuremod2(label_dates_weekly, feature_dates,  city_ids)) 
predFrame3 = pd.DataFrame(predFrame3.load_data()) 
""" 

"""
label_dates_weekly = "'2019-10-06', '2019-10-13', '2019-10-20', '2019-10-27'" 
city_ids = '1,5,6,8,10,12,20,23,198' 
feature_dates="'2019-09-22'" 
proposal_start_date = '2019-09-30' 

predFrame4 = qr.execute('presto', rxgy_data_sapphire_presto_featuremod2(label_dates_weekly, feature_dates,  city_ids)) 
predFrame4 = pd.DataFrame(predFrame4.load_data()) 
""" 


# In[3]:


cohort_column_name = 'cohort' 
treatment_indicator_value = 'treatment' 
control_indicator_value = 'control' 

print(len(predFrame2))
print(sum(predFrame2[cohort_column_name] == control_indicator_value))


# In[4]:


### preprocess the data 
### -- sample treatment to match control cohort 
### -- eliminate nulls, standard normalization 

D = predFrame2 
D = D.sample(frac=1.0) 

feature_list = [ 
    'rating_2driver_min_avg_84d'
    , 'trip_incomplete_total_84d'
    , 'days_active_84d'
    , 'days_since_trip_first_lifetime'
    , 'days_since_last_hard_churn_lifetime'
    , 'days_since_last_soft_churn_lifetime'
    , 'fare_max_sd_84d'
    , 'churns_hard_lifetime'
    , 'trips_lifetime'
    , 'fare_max_p50_84d'
    , 'duration_session_pre_request_max_p50_84d'
    , 'trip_pool_per_x_84d'
    , 'fare_total_win7d_sd_84d'
    , 'trip_complete_win7d_sd_84d'
    , 'session_per_days_active_84d'
    , 'churns_soft_lifetime'
    , 'trip_complete_per_days_active_84d'
    , 'trip_pool_prc_84d'
    , 'session_background_pre_request_prc_84d'
    , 'session_lt_1m_prc_84d'
    , 'session_request_prc_84d'
    , 'duration_session_outside_total_prc_84d'
    , 'trip_x_prc_84d'
    , 'days_since_trip_last_lifetime'
    , 'promo_used_84d'
    , 'has_session_request_84d'
    , 'has_session_without_request_84d' 
    , 'fare_promo_total_avg_84d', 
    'fare_total_avg_84d', 
    'surge_trip_avg_84d', 
    'fare_total_win7d_potential_84d', 
    'fare_total_win28d_potential_84d', 
    'fare_lifetime', 
    'time_to_first_message_minutes_mean_lifetime', 
    'ata_trip_max_avg_84d', 
    'eta_trip_max_avg_84d', 
    'trip_pool_matched_avg_84d', 
    'payment_cash_trip_total_84d', 
    'duration_trip_total_p50_84d'
] 

label_list = [ 
    'label_trip_28d', 
    'label_vc_28d' 
] 

for l in feature_list: 
    print('number of nans: ' + str(sum(D[l] == '\\N'))) 
    D[l] = pd.to_numeric(D[l], errors='coerce') 
    D[l] = D[l] - D[l].mean() 
    D[l] = D[l] / D[l].std() 
    D[l][pd.isnull(D[l])] = 0.0 ## at zero mean due to standard normalization 

for l in label_list: 
    D[l] = pd.to_numeric(D[l], errors='coerce') 
    D[l][pd.isnull(D[l])] = 0.0 ## at zero mean due to standard normalization 

### -- compute simple statistics 
### compute cpit 
treated_entries = D[D[cohort_column_name] == treatment_indicator_value] 
untreated_entries = D[D[cohort_column_name] == control_indicator_value] 

rpu_treated = float(treated_entries[label_list[0]].sum()) / len(treated_entries) 
nipu_treated = float(treated_entries[label_list[1]].sum()) / len(treated_entries) 

rpu_untreated = float(untreated_entries[label_list[0]].sum()) / len(untreated_entries) 
nipu_untreated = float(untreated_entries[label_list[1]].sum()) / len(untreated_entries) 

cpit = -1.0 * (nipu_treated - nipu_untreated) / (rpu_treated - rpu_untreated) 

print('rpu_treated : ' + str(rpu_treated)) 
print('nipu_treated : ' + str(nipu_treated)) 
print('rpu_untreated : ' + str(rpu_untreated)) 
print('nipu_untreated : ' + str(nipu_untreated)) 
print('cpit : ' + str(cpit)) 

### split the data into 3/1/1 train/val/test 
len_tr = int(len(D) / 5 * 3) 
len_va = int(len(D) / 5) 

nX = D[feature_list].as_matrix() 
w = D[cohort_column_name].apply(lambda x: 1.0 if x == treatment_indicator_value else 0.0) 
w = w.as_matrix() 
values = D[label_list[0]] 
values = values.as_matrix() 
negcost = D[label_list[1]] 
negcost = negcost.as_matrix() * 1.0 

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

negcost_tr = negcost[0:len_tr] 

negcost_va = negcost[len_tr:len_tr + len_va] 

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

if use_python3 == True: 
    pkl.dump(saveD, open('../data/rxgy_ma_training_data_v5_2019_07_08_vc_tr_featuremod3', 'wb')) 
else:
    pkl.dump(saveD, open('../data/rxgy_ma_training_data_v5_2019_07_08_vc_tr_featuremod3', 'w')) 

# In[ ]:


### code implements ranking model for treatment effect 
### for optimizing with respect to direct marketplace objectives 

### using tensorflow 

import numpy as np, pandas as pd, pickle as pkl, tensorflow as tf 
sys.path.append('../')  
from ModelDefinitions import * 
from DataProcFunctions import * 

### RxGy TQR setting: 
p_quantile = 0.4 ## percentage of quantile to aim for 
num_optimize_iterations = 300 ## number of optimization iterations 
num_modeling_inits = 2 ## number of random initializations 
num_hidden = 0 ## number of hidden units in DNN 
use_schedule = True ## option to use a constraint annealing schedule 
temp = 0.5 ## initial temperature for constraints 
inc_temp = 0.1 ## increment of temperature per 100 iterations 
save_cf_data = False ### whether to save data for causal forest training 

## set a random seed to reproduce results 
seed = 1234; np.random.seed(seed) #tf.compat.v2.random.set_seed(seed); 

sample_frac = 1.0 ## option to sample data by a fraction \in (0, 1) 
data_filename =  '../data/rxgy_ma_training_data_v5_2019_07_08_vc_tr_featuremod3' 
prefix = 'rxgy_v5_07_08_featuremod3_tr_iter100_run4' 

D_tre, D_unt, Dv_tre, Dv_unt, Dt_tre, Dt_unt, o_tre, o_unt, ov_tre, ov_unt, ot_tre, ot_unt, c_tre, c_unt, cv_tre, cv_unt, ct_tre, ct_unt, D, w, o, c, Dv, wv, ov, cv, Dt, wt, ot, ct = LoadDataFromPkl(data_filename, frac = sample_frac, use_python3=use_python3, save_cf_data=save_cf_data) 


print('### ----- start the training of deep learning models ------ ') 
gs_tqr = [] 
gs_drm = [] 
for i in range(num_modeling_inits): 
    gs_tqr.append(tf.Graph()) 
for i in range(num_modeling_inits): 
    gs_drm.append(tf.Graph()) 

print('------> Training TQR ranking model .... ') 
val_results = [] 
sess_list = [] 
for i in range(num_modeling_inits): 
    print('---> running cross validation, iteration: ' + str(i)) 
    obj, opt, dumh, dumhu, vtemp, p_quantile = TunableTQRankingModelDNN(gs_tqr[i], D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, 'train-first', temp, p_quantile, num_hidden, use_schedule) 
    ### session definitions and variable initialization 
    sess = tf.Session(graph = gs_tqr[i]) 
    sess_list.append(sess) 
    
    ### initialize variables and run optimization 
    with gs_tqr[i].as_default() as g: 
        init = tf.global_variables_initializer() 
    sess.run(init) 
    cur_temp = temp 
    for step in range(num_optimize_iterations): 
        _, objres = sess.run([opt, obj]) 
        if step % 100 == 0: 
            cur_temp = cur_temp + inc_temp 
            print('opt. step : ' + str(step) + ' obj: ' + str(objres)) 
            if use_schedule: 
                sess.run(vtemp.assign(cur_temp))
                print('setting temperature to :' + str(sess.run(vtemp))) 
    
    print('---> optimization finished ... ') 
    tempvalue = sess.run(vtemp)
    p_quantilevalue = p_quantile
    print('temp:') 
    print(tempvalue)
    print('p_quantile:')
    print(p_quantilevalue) 
    
    ### evaluate CPIT metric on validation set 
    objv, dumo, dumh, dumhu, dvtemp, dp_quantile = TunableTQRankingModelDNN(gs_tqr[i], Dv_tre, Dv_unt, ov_tre, ov_unt, cv_tre, cv_unt, 'eval', temp, p_quantile, num_hidden, use_schedule) 
    
    val_result = sess.run(objv) 
    print('validation CPIT:') 
    print(val_result) 
    val_results.append(val_result) 

from operator import itemgetter 
best_index = min(enumerate(val_results), key=itemgetter(1))[0] 

print('best performing model: iteration ' + str(best_index)) 

### run scoring on whole test set 
with gs_tqr[best_index].as_default() as g: 
    if num_hidden > 0: 
        with tf.variable_scope("tqrhidden") as scope: 
            h1_test = tf.contrib.layers.fully_connected(Dt, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
        with tf.variable_scope("tqranker") as scope: 
            h_test = tf.contrib.layers.fully_connected(h1_test, 1, activation_fn=None, reuse=tf.AUTO_REUSE, scope=scope) 
    else: 
        with tf.variable_scope("tqranker") as scope: 
            h_test = tf.contrib.layers.fully_connected(Dt, 1, activation_fn=None, reuse=tf.AUTO_REUSE, scope=scope) 
    tqrscore = sess_list[best_index].run(h_test) 

print('------> Training DRM ranking model .... ') 
sess_list = [] 
val_results = [] 
for i in range(num_modeling_inits): 
    print('---> running cross validation, iteration: ' + str(i)) 
    ### ---- train cpit ranking model for comparison --- 
    dobjc, doptc, ddumh, ddumu = DirectRankingModelDNN(gs_drm[i], D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, 'train-first-drm', num_hidden) 
    
    dsess = tf.Session(graph = gs_drm[i]) 
    sess_list.append(dsess) 
    
    ### initialize variables and run optimization 
    with gs_drm[i].as_default() as g: 
        dinit = tf.global_variables_initializer() 
    dsess.run(dinit) 
    for step in range(num_optimize_iterations): 
        _, dobjres = dsess.run([doptc, dobjc]) 
        if step % 100 == 0: 
            print('opt. step : ' + str(step) + ' obj: ' + str(dobjres)) 
    
    print('---> optimization finished ... ') 
    
    ### evaluate CPIT metric on validation set 
    dobjv, ddumo, dumh, dumhu = DirectRankingModelDNN(gs_drm[i], Dv_tre, Dv_unt, ov_tre, ov_unt, cv_tre, cv_unt, 'eval', num_hidden)
    val_result = dsess.run(dobjv) 
    print('validation CPIT:') 
    print(val_result) 
    val_results.append(val_result) 

best_index = min(enumerate(val_results), key=itemgetter(1))[0] 

print('best performing model: iteration ' + str(best_index)) 

### run scoring on whole test set 
with gs_drm[best_index].as_default() as g: 
    if num_hidden > 0: 
        with tf.variable_scope("drmhidden") as scope: 
            h1_test = tf.contrib.layers.fully_connected(Dt, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
        with tf.variable_scope("drmranker") as scope: 
            h_test = tf.contrib.layers.fully_connected(h1_test, 1, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope) 
    else: 
        with tf.variable_scope("drmranker") as scope: 
            h_test = tf.contrib.layers.fully_connected(Dt, 1, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope) 
    drmscore = sess_list[best_index].run(h_test) 

### ---- train hte model for comparison ---- 
### we could utimize the original HTE functions 
from LinearHTEModels import * 
from PromotionModels import PromotionModels 

pmodels = PromotionModels() 

## set-up RLearner 
rl_ridge_model_O, rl_ridge_model_C = pmodels.fit_rlearner(D, o, c, w) 

## one model for order lift and one model for cost drop 
pred_values_va_rlearner_O = rl_ridge_model_O.predict(Dt) 
pred_values_va_rlearner_C = rl_ridge_model_C.predict(Dt) 

#if ranking_model == 'effectiveness-ratio': ## if we use the effectiveness ratio model, compute effectiveness ratio 
pred_values_va_rlearner = np.divide(np.maximum(pred_values_va_rlearner_O, 0), pred_values_va_rlearner_C + 1e-7) 

lhmodels = LinearHTEModels() 
lambds = [0.1] 
rlearnerscores = [] 
rl_ridge_model_L_list = [] 
## set-up lagrangian rlearner 
for i in range(len(lambds)): 
    lambd = lambds[i] 
    rl_ridge_model_L = lhmodels.fit_rlearner_lagrangian(D, o, c, w, lambd) 
    rl_ridge_model_L_list.append(rl_ridge_model_L) 
    rlearnerscores.append(rl_ridge_model_L.predict(Dt)) 

""" 
### this section is to load the results trained by grf R code 
### 

ot_cf = pd.read_csv('../results/causal_forest_grf_test_set_results_O_finalsize_numtrees1002.csv') 
ct_cf = pd.read_csv('../results/causal_forest_grf_test_set_results_C_finalsize_numtrees1002.csv') 

ot_cf = ot_cf.as_matrix() 
Ocfscores = ot_cf[0] 

ct_cf = ct_cf.as_matrix() 
Ccfscores = ct_cf[0] 

cfscore = np.divide(Ocfscores, Ccfscores) 
""" 

### ---- experimentation and plotting cost-curves ----- 
from experimentation import * 
exp = Experimentation() 
ranscore = np.random.rand(ot.shape[0], ) 
colors = ['b', 'c', 'g', 'y'] 
plt.figure() 
rlearnerauccs = [] 
ranaucc = exp.AUC_cpit_cost_curve_deciles_cohort(ranscore, ot, wt, -1.0 * ct, 'k', plot_random=True) 
quasiaucc = exp.AUC_cpit_cost_curve_deciles_cohort(pred_values_va_rlearner_O, ot, wt, -1.0 * ct, 'c') 
for i in range(len(lambds)): 
    rlearnerauccs.append(exp.AUC_cpit_cost_curve_deciles_cohort(rlearnerscores[i], ot, wt, -1.0 * ct, colors[i] )) 
#cfaucc = exp.AUC_cpit_cost_curve_deciles_cohort(cfscore, ot, wt, -1.0 * ct, 'g') # causal forest aucc and plotting 
tqraucc = exp.AUC_cpit_cost_curve_deciles_cohort(tqrscore, ot, wt, -1.0 * ct, 'r' ) 
drmaucc = exp.AUC_cpit_cost_curve_deciles_cohort(drmscore, ot, wt, -1.0 * ct, 'm' ) 
plt.title('Causal learning cost curves using targeting models') 

"""
print('temp:') 
print(tempvalue) 
print('p_quantile:') 
print(p_quantilevalue) 
"""

### --- add legeneds to plot ---- 
leg_str = ['Random'] 
leg_str.append('R-learner on Incremental Gain') 
for i in range(len(lambds)): 
    leg_str.append('Duality R-learner') 
#leg_str.append('Causal Forest') # causal forest result 
leg_str.append('Top Quantile Ranking at ' + str(p_quantile*100) + '%') 
leg_str.append('Direct Ranking Model') 
plt.legend(leg_str) 

### --- print out aucc results for different models --- 
print('AUCC results: ') 
print('random: ' + str(ranaucc)) 
print('rlearner: ' + str(quasiaucc)) 
i = 0
for rlearneraucc in rlearnerauccs: 
    print('duality rlearner ' + str(i + 1) + ' with lambda = ' + str(lambds[i]) + ':' + str(rlearneraucc)) 
    i = i + 1
#print('cf: ' + str(cfaucc)) 
print('drm: ' + str(drmaucc)) 
print('tqr: ' + str(tqraucc)) 

plt.show() 

### --- saving data to results folder ---- 
save_filename = '../results/benchmarkwithcv_tqr_drm_hte_'+prefix+'_main_results.pkl' 
saveD = {'quasiscore':pred_values_va_rlearner, 'quasiscore_O':pred_values_va_rlearner_O, 'rlearnerscore':rlearnerscores, 'ot':ot, 'wt':wt, 'ct':ct, 'rlearnerauccs':rlearnerauccs, 'ranaucc':ranaucc, 'quasiaucc':quasiaucc, 'tqrscore':tqrscore, 'drmscore':drmscore, 'tempvalue':tempvalue, 'p_quantilevalue':p_quantilevalue, 'tqraucc':tqraucc, 'drmaucc':drmaucc} 
#'cfscore':cfscore, causal forest scores 

pkl.dump(saveD, open(save_filename, 'wb')) 

# In[ ]:
