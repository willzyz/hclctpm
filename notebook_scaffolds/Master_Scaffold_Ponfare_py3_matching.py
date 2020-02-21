#!/usr/bin/env python 
# coding: utf-8 

# In[1]:

import pandas as pd, numpy as np, pickle as pkl 
import sys, os 
sys.path.append('../dataprep/') 
sys.path.append('../models/') 
from QueryFunctions import * 
import tensorflow as tf 

# In[2]:

use_python3 = True 
use_query = 0 ### turn this switch on (to 1.0/True vs 0.0/False) to run query using QueryRunner 

### code implements ranking model for treatment effect 
### for optimizing with respect to direct marketplace objectives 

### using tensorflow 

import numpy as np, pandas as pd, pickle as pkl, tensorflow as tf 
sys.path.append('../') 
from ModelDefinitions import * 
from DataProcFunctions import * 

stop_opt_obj_value = 3000.0 
### RxGy CTPM setting: 
p_quantile = 0.4 ## percentage of quantile to aim for 
num_optimize_iterations = 1500 #2500 ## number of optimization iterations 
num_modeling_inits = 8 ## number of random initializations 
num_hidden = 15 ## number of hidden units in DNN 
use_schedule = True ## option to use a constraint annealing schedule 
temp = 0.5 ## initial temperature for constraints 
inc_temp = 0.1 ## increment of temperature per 100 iterations 
save_cf_data = False ### whether to save data for causal forest training 
with_intensity = True 

## set a random seed to reproduce results 
seed = 1234; np.random.seed(seed) #tf.compat.v2.random.set_seed(seed); 

sample_frac = 1.0 ## option to sample data by a fraction \in (0, 1) 
data_filename =  '../data/ponfare_v3_causal_data_with_intensity_matching_new_run' 
prefix = 'ponfare_run4' 

Da_tre, Da_unt, Db_tre, Db_unt, Dva_tre, Dva_unt, Dvb_tre, Dvb_unt, Dta_tre, Dta_unt, Dtb_tre, Dtb_unt, o_tre, o_unt, ov_tre, ov_unt, ot_tre, ot_unt, c_tre, c_unt, cv_tre, cv_unt, ct_tre, ct_unt, Da, Db, w, o, c, Dva, Dvb, wv, ov, cv, Dta, Dtb, wt, ot, ct, int_tre, int_unt, intv_tre, intv_unt, intt_tre, intt_unt, int, intv, intt = LoadDataFromPklMatching(data_filename, frac = sample_frac, use_python3=use_python3, save_cf_data=save_cf_data, with_intensity=with_intensity) 
Dt = np.concatenate((Dta, Dtb), axis=1) 
D_tre = np.concatenate((Da_tre, Db_tre), axis=1) 
D_unt = np.concatenate((Da_unt, Db_unt), axis=1) 
Dv_tre = np.concatenate((Dva_tre, Dvb_tre), axis=1) 
Dv_unt = np.concatenate((Dva_unt, Dvb_unt), axis=1) 
D = np.concatenate((Da, Db), axis=1) 

print('### ----- start the training of deep learning models ------ ') 
gs_tqr = [] 
gs_drm = [] 
for i in range(num_modeling_inits): 
    gs_tqr.append(tf.Graph()) 
for i in range(num_modeling_inits): 
    gs_drm.append(tf.Graph()) 

print('------> Training CTPM ranking model .... ') 
val_results = [] 
sess_list = [] 
for i in range(num_modeling_inits): 
    print('---> running cross validation, iteration: ' + str(i)) 
    #obj, opt, dumh, dumhu, vtemp, p_quantile = TunableTQRankingModelDNN(gs_tqr[i], D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, 'train-first', temp, p_quantile, num_hidden, use_schedule) 
    obj, opt, dumh, dumhu = CTPMMatcherDNN(gs_tqr[i], Da_tre, Da_unt, Db_tre, Db_unt, o_tre, o_unt, c_tre, c_unt, int_tre, int_unt, 'train-CTPM-first', num_hidden)
    
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
        if objres < stop_opt_obj_value: 
            print('objres : ' + str(objres) + '... breaking ... ') 
            break 
        
        if step % 100 == 0: 
            #cur_temp = cur_temp + inc_temp 
            print('opt. step : ' + str(step) + ' obj: ' + str(objres)) 
            #if use_schedule: 
                #sess.run(vtemp.assign(cur_temp))
                #print('setting temperature to :' + str(sess.run(vtemp))) 
    
    print('---> optimization finished ... ') 
    #tempvalue = sess.run(vtemp)
    #p_quantilevalue = p_quantile
    #print('temp:') 
    #print(tempvalue)
    #print('p_quantile:')
    #print(p_quantilevalue) 
    
    ### evaluate CPIT metric on validation set 
    objv, dumo, dumh, dumhu = CTPMMatcherDNN(gs_tqr[i], Dva_tre, Dva_unt, Dvb_tre, Dvb_unt, ov_tre, ov_unt, cv_tre, cv_unt, intv_tre, intv_unt, 'eval', num_hidden)
    #objv, dumo, dumh, dumhu, dvtemp, dp_quantile = TunableTQRankingModelDNN(gs_tqr[i], Dv_tre, Dv_unt, ov_tre, ov_unt, cv_tre, cv_unt, 'eval', temp, p_quantile, num_hidden, use_schedule) 
    
    val_result = sess.run(objv) 
    print('validation CPIT:') 
    print(val_result) 
    if val_result > 0: 
        val_results.append(val_result) 
    else: 
        val_results.append(1e10) 
    
from operator import itemgetter 
best_index = min(enumerate(val_results), key=itemgetter(1))[0] 

print('best performing model: iteration ' + str(best_index)) 

### run scoring on whole test set 
with gs_tqr[best_index].as_default() as g: 
    if num_hidden > 0: 
        with tf.variable_scope("ctpmmatcherhidden_a") as scope: 
            h_tre_matchhidden = tf.contrib.layers.fully_connected(Dta, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
        with tf.variable_scope("ctpmmatcher") as scope: 
            h_tre_matchscore = tf.contrib.layers.fully_connected(h_tre_matchhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
    else: 
        with tf.variable_scope("ctpmmatcher") as scope: 
            h_tre_matchscore = tf.contrib.layers.fully_connected(Dta, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
        
    with tf.variable_scope("ctpmmatcherhidden_b") as scope: 
        h_tre_matchhidden_b = tf.contrib.layers.fully_connected(Dtb, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
        
    if num_hidden > 0: 
        with tf.variable_scope("ctpmpolicysighidden") as scope: 
            h_tre_policyhidden = tf.contrib.layers.fully_connected(Dt, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
        with tf.variable_scope("ctpmpolicysig") as scope: 
            h_tre_policyscore = tf.contrib.layers.fully_connected(h_tre_policyhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
    else: 
        with tf.variable_scope("ctpmpolicysig") as scope: 
            h_tre_policyscore = tf.contrib.layers.fully_connected(Dt, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
    
    ## use the bell-shape cost function in treatment intensity 
    diff_tre = np.reshape(intt, (-1, 1)) - h_tre_policyscore 
        
    lh_tre_policyscore = tf.math.multiply(tf.math.sigmoid(diff_tre), (1 - tf.math.sigmoid(diff_tre))) 
        
    ## this is the un-normalized bayesian weighting score 
    s_tre_unnorm = tf.math.multiply(h_tre_matchscore, lh_tre_policyscore)     
    
    tre_normalize_a = tf.nn.l2_normalize(h_tre_matchhidden,1)
    tre_normalize_b = tf.nn.l2_normalize(h_tre_matchhidden_b,1)
    tre_cos_similarity=tf.reduce_sum(tf.multiply(tre_normalize_a,tre_normalize_b), axis = 1)
    tre_matchscore = 1 + tf.reshape(tre_cos_similarity, [-1, 1]) 
    
    ## this is the un-normalized bayesian weighting score with matching 
    s_tre_unnorm = tf.math.multiply(s_tre_unnorm, tre_matchscore) 
    
    ## this is the un-normalized bayesian weighting score 
    h_test = s_tre_unnorm 
    ctpmscore = sess_list[best_index].run(h_test) 
    
    """
    if num_hidden > 0: 
        with tf.variable_scope("tqrhidden") as scope: 
            h1_test = tf.contrib.layers.fully_connected(Dt, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
        with tf.variable_scope("tqranker") as scope: 
            h_test = tf.contrib.layers.fully_connected(h1_test, 1, activation_fn=None, reuse=tf.AUTO_REUSE, scope=scope) 
    else: 
        with tf.variable_scope("tqranker") as scope: 
            h_test = tf.contrib.layers.fully_connected(Dt, 1, activation_fn=None, reuse=tf.AUTO_REUSE, scope=scope) 
    tqrscore = sess_list[best_index].run(h_test) 
    """ 
    
print('------> Training DRM ranking model .... ') 
sess_list = [] 
val_results = [] 
for i in range(num_modeling_inits): 
    print('---> running cross validation, iteration: ' + str(i)) 
    ### ---- train cpit ranking model for comparison --- 
    dobjc, doptc, ddumh, ddumu = SimpleTCModelDNN(gs_drm[i], D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, 'train-first-drm', num_hidden) 
    
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
    dobjv, ddumo, dumh, dumhu = SimpleTCModelDNN(gs_drm[i], Dv_tre, Dv_unt, ov_tre, ov_unt, cv_tre, cv_unt, 'eval', num_hidden)
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

ctpmscore = np.reshape(ctpmscore, (-1,)) 
drmscore = np.reshape(drmscore, (-1,)) 

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
pred_values_va_rlearner = pred_values_va_rlearner_O #np.divide(np.maximum(pred_values_va_rlearner_O, 0), pred_values_va_rlearner_C + 1e-7) 

lhmodels = LinearHTEModels() 
lambds = [0.1, 0.01, 0.001] 
rlearnerscores = [] 
rl_ridge_model_L_list = [] 
## set-up lagrangian rlearner 
for i in range(len(lambds)): 
    lambd = lambds[i] 
    rl_ridge_model_L = lhmodels.fit_rlearner_lagrangian(D, o, c, w, lambd) 
    rl_ridge_model_L_list.append(rl_ridge_model_L) 
    rlearnerscores.append(rl_ridge_model_L.predict(Dt)) 

#import ipdb; ipdb.set_trace()
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
colors = ['b', 'g', 'y'] 
plt.figure() 
rlearnerauccs = [] 
ranaucc = exp.AUC_cpit_cost_curve_deciles_cohort(ranscore, ot, wt, -1.0 * ct, 'k', plot_random=True) 
quasiaucc = exp.AUC_cpit_cost_curve_deciles_cohort(pred_values_va_rlearner_O, ot, wt, -1.0 * ct, 'c') 
for i in range(len(lambds)): 
    rlearnerauccs.append(exp.AUC_cpit_cost_curve_deciles_cohort(rlearnerscores[i], ot, wt, -1.0 * ct, colors[i] )) 
#cfaucc = exp.AUC_cpit_cost_curve_deciles_cohort(cfscore, ot, wt, -1.0 * ct, 'g') # causal forest aucc and plotting 
ctpmaucc = exp.AUC_cpit_cost_curve_deciles_cohort(ctpmscore, ot, wt, -1.0 * ct, 'r' ) 
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
leg_str.append('CTPM') 
leg_str.append('Simple CT Model') 
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
print('ctpm: ' + str(ctpmaucc)) 

plt.show() 

### --- saving data to results folder ---- 
save_filename = '../results/benchmarkwithcv_ctpm_drm_hte_'+prefix+'_main_results.pkl' 
saveD = {'quasiscore':pred_values_va_rlearner, 'quasiscore_O':pred_values_va_rlearner_O, 'rlearnerscore':rlearnerscores, 'ot':ot, 'wt':wt, 'ct':ct, 'rlearnerauccs':rlearnerauccs, 'ranaucc':ranaucc, 'quasiaucc':quasiaucc, 'ctpmscore':ctpmscore, 'drmscore':drmscore, 'tempvalue':tempvalue, 'p_quantilevalue':p_quantilevalue, 'ctpmaucc':ctpmaucc, 'drmaucc':drmaucc} 
#'cfscore':cfscore, causal forest scores 

pkl.dump(saveD, open(save_filename, 'wb')) 

# In[ ]:
