#!/usr/bin/env python 
# coding: utf-8 

# In[1]: 

#global_iter = 1

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

reload = 0 
reload_pathfile = '../models/save/ctpm_model_ponfare_run4_for_analysis_step1500_init_2.ckpt-1500' 
#ctpm_model_ponfare_run4_step100_init_0.ckpt-100' 
d2dlamb = 0.1 
stop_opt_obj_value = 3000.0 
### RxGy CTPM setting: 
p_quantile = 0.4 ## percentage of quantile to aim for 
num_optimize_iterations = 2500 ## number of optimization iterations 
num_modeling_inits = 1 ## number of random initializations 
num_hidden = 15 ## number of hidden units in DNN 
use_schedule = True ## option to use a constraint annealing schedule 
temp = 0.5 ## initial temperature for constraints 
inc_temp = 0.1 ## increment of temperature per 100 iterations 
save_cf_data = False ### whether to save data for causal forest training 
with_intensity = True 

## set a random seed to reproduce results 
seed = 1234; np.random.seed(seed) #tf.compat.v2.random.set_seed(seed); 

sample_frac = 1.0 ## option to sample data by a fraction \in (0, 1) 
data_filename =  '../data/ponfare_v3_causal_data_with_intensity_matching_d2dist' 
prefix = 'ponfare_run5_std_bars' 

Da_tre, Da_unt, Db_tre, Db_unt, Dva_tre, Dva_unt, Dvb_tre, Dvb_unt, Dta_tre, Dta_unt, Dtb_tre, Dtb_unt, o_tre, o_unt, ov_tre, ov_unt, ot_tre, ot_unt, c_tre, c_unt, cv_tre, cv_unt, ct_tre, ct_unt, Da, Db, w, o, c, Dva, Dvb, wv, ov, cv, Dta, Dtb, wt, ot, ct, int_tre, int_unt, intv_tre, intv_unt, intt_tre, intt_unt, int, intv, intt, d2d_tre, d2d_unt, d2dv_tre, d2dv_unt, d2dt_tre, d2dt_unt, d2d, d2dv, d2dt = LoadDataFromPklMatchingD2Dist(data_filename, frac = sample_frac, use_python3=use_python3, save_cf_data=save_cf_data, with_intensity=with_intensity) 

Dt = np.concatenate((Dta, Dtb), axis=1) 
D_tre = np.concatenate((Da_tre, Db_tre), axis=1) 
D_unt = np.concatenate((Da_unt, Db_unt), axis=1) 
Dv_tre = np.concatenate((Dva_tre, Dvb_tre), axis=1) 
Dv_unt = np.concatenate((Dva_unt, Dvb_unt), axis=1) 
D = np.concatenate((Da, Db), axis=1) 
Dt_tre = np.concatenate((Dta_tre, Dtb_tre), axis=1)
Dt_unt = np.concatenate((Dta_unt, Dtb_unt), axis=1)

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
    obj, opt, dumh, dumhu, saver, opt_price_tre, opt_price_unt, s_emb_tre, s_emb_unt, c_emb_tre, c_emb_unt = CTPMMatcherD2DistDNN(gs_tqr[i], Da_tre, Da_unt, Db_tre, Db_unt, o_tre, o_unt, c_tre, c_unt, int_tre, int_unt, d2d_tre, d2d_unt, d2dlamb, 'train-CTPM-first', num_hidden) 
    #obj, opt, dumh, dumhu = CTPMDNN(gs_tqr[i], D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, int_tre, int_unt, 'train-CTPM-first', num_hidden) 
    
    ### session definitions and variable initialization 
    sess = tf.Session(graph = gs_tqr[i]) 
    sess_list.append(sess) 
    
    if reload: 
        saver.restore(sess, reload_pathfile) 
    else: 
        ### initialize variables and run optimization 
        with gs_tqr[i].as_default() as g: 
            init = tf.global_variables_initializer() 
        sess.run(init)     
        cur_temp = temp 
        for step in range(num_optimize_iterations): 
            _, objres = sess.run([opt, obj]) 
            if objres < stop_opt_obj_value and step != 0 and step != 1: 
                print('objres : ' + str(objres) + '... breaking ... ') 
                break 
        
            if step % 100 == 0: 
                print('opt. step : ' + str(step) + ' obj: ' + str(objres)) 
                if step != 0: 
                    ckpt_pathfile = '../models/save/ctpm_model_'+prefix+'_step' + str(step) +'_init_'+str(i)+'.ckpt' 
                    saver.save(sess, ckpt_pathfile, step) 
                    
        print('---> optimization finished ... ') 
    
    ### evaluate CPIT metric on validation set 
    #objv, dumo, dumh, dumhu = CTPMDNN(gs_tqr[i], Dv_tre, Dv_unt, ov_tre, ov_unt, cv_tre, cv_unt, intv_tre, intv_unt, 'eval', num_hidden)
    objv, dumo, dumh, dumhu, dsaver, vopt_price_tre, vopt_price_unt, vs_emb_tre, vs_emb_unt, vc_emb_tre, vc_emb_unt = CTPMMatcherD2DistDNN(gs_tqr[i], Dva_tre, Dva_unt, Dvb_tre, Dvb_unt, ov_tre, ov_unt, cv_tre, cv_unt, intv_tre, intv_unt, d2dv_tre, d2dv_unt, d2dlamb, 'eval', num_hidden)
    #objv, dumo, dumh, dumhu, dvtemp, dp_quantile = TunableTQRankingModelDNN(gs_tqr[i], Dv_tre, Dv_unt, ov_tre, ov_unt, cv_tre, cv_unt, 'eval', temp, p_quantile, num_hidden, use_schedule) 
    
    ctpm_val_result = sess.run(objv) 
    print('CTPM validation CPIT:') 
    print(ctpm_val_result) 
    if ctpm_val_result > 0: 
        val_results.append(ctpm_val_result) 
    else: 
        val_results.append(1e10) 
    
    objt, dumo, dumh, dumhu, dsaver, topt_price_tre, topt_price_unt, ts_emb_tre, ts_emb_unt, tc_emb_tre, tc_emb_unt = CTPMMatcherD2DistDNN(gs_tqr[i], Dta_tre, Dta_unt, Dtb_tre, Dtb_unt, ot_tre, ot_unt, ct_tre, ct_unt, intt_tre, intt_unt, d2dt_tre, d2dt_unt, d2dlamb, 'test', num_hidden)
    
    ctpm_test_result = sess.run(objt) 
    print('test CPIT:') 
    print(ctpm_test_result) 
    if reload: 
        [r_topt_price_tre, r_topt_price_unt, r_ts_emb_tre, r_ts_emb_unt, r_tc_emb_tre, r_tc_emb_unt] = sess.run([topt_price_tre, topt_price_unt, ts_emb_tre, ts_emb_unt, tc_emb_tre, tc_emb_unt]) 
        Dt_tre_assemble = np.concatenate((Dta_tre, Dtb_tre), axis = 1) 
        Dt_unt_assemble = np.concatenate((Dta_unt, Dtb_unt), axis = 1) 
        Dt_assemble = np.concatenate((Dt_tre_assemble, Dt_unt_assemble), axis = 0) 
        r_topt_price = np.concatenate((r_topt_price_tre, r_topt_price_unt), axis = 0) 
        r_ts_emb = np.concatenate((r_ts_emb_tre, r_ts_emb_unt), axis = 0) 
        r_tc_emb = np.concatenate((r_tc_emb_tre, r_tc_emb_unt), axis = 0) 
        i_price = np.concatenate((intt_tre, intt_unt), axis = 0) 
        
        D_raw = pkl.load(open(data_filename, 'rb'), encoding="latin1") 
        feature_list_a = D_raw['feature_list_a'] 
        feature_list_b = D_raw['feature_list_b'] 
        feature_list = feature_list_a + feature_list_b 
        ct_assemble = np.concatenate((ct_tre, ct_unt), axis = 0) 
        savedict = dict() 
        savedict = {'r_topt_price':r_topt_price, 'r_ts_emb':r_ts_emb, 'r_tc_emb':r_tc_emb, 'Dt_assemble':Dt_assemble, 'i_price':i_price, 'feature_list':feature_list, 'ct_assemble':ct_assemble} 
        pkl.dump(savedict, open('../data/save_icml_analysis_data.pkl', 'wb')) 
        

from operator import itemgetter 
best_index = min(enumerate(val_results), key=itemgetter(1))[0] 

print('best performing model: iteration ' + str(best_index)) 

### run scoring on whole test set 
with gs_tqr[best_index].as_default() as g: 
    h_test = forwardCTPMMatcherD2DistDNN(Dta, Dtb, intt, num_hidden) 
    #h_test = forwardCTPMMatcherFeatureD2DistDNN(Dta, Dt, num_hidden) 
    ## this is the un-normalized bayesian weighting score 
    ctpmscore = sess_list[best_index].run(h_test) 

print('------> Training simple TC model .... ') 
num_hidden = 0 
sess_list = [] 
val_results = [] 
for i in range(num_modeling_inits): 
    print('---> running cross validation, iteration: ' + str(i)) 
    ### ---- train cpit ranking model for comparison --- 
    dobjc, doptc, ddumh, ddumu, saver = SimpleTCModelDNN(gs_drm[i], D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, 'train-first-drm', num_hidden) 
    
    dsess = tf.Session(graph = gs_drm[i]) 
    sess_list.append(dsess) 
    
    if reload: 
        saver.restore(sess, reload_pathfile) 
    else:     
        ### initialize variables and run optimization 
        with gs_drm[i].as_default() as g: 
            dinit = tf.global_variables_initializer() 
        dsess.run(dinit) 
        for step in range(num_optimize_iterations): 
            _, dobjres = dsess.run([doptc, dobjc]) 
            if step % 100 == 0: 
                print('opt. step : ' + str(step) + ' obj: ' + str(dobjres)) 
                if step != 0: 
                    ckpt_pathfile = '../models/save/simple_TC_model_'+prefix+'_step' + str(step) +'_init_'+str(i)+'.ckpt' 
                    saver.save(sess, ckpt_pathfile, step) 
    
    print('---> optimization finished ... ') 
    
    ### evaluate CPIT metric on validation set 
    dobjv, ddumo, dumh, dumhu, dsaver = SimpleTCModelDNN(gs_drm[i], Dv_tre, Dv_unt, ov_tre, ov_unt, cv_tre, cv_unt, 'eval', num_hidden) 
    drm_val_result = dsess.run(dobjv) 
    print('validation CPIT:') 
    print(drm_val_result) 
    val_results.append(drm_val_result) 
    
    #dobjt, ddumo, dumh, dumhu, dsaver = SimpleTCModelDNN(gs_drm[i], Dt_tre, Dt_unt, ot_tre, ot_unt, ct_tre, ct_unt, 'test', num_hidden) 
    #drm_test_result = sess.run(dobjt) 
    #print('test CPIT:') 
    #print(drm_test_result) 

best_index = min(enumerate(val_results), key=itemgetter(1))[0] 

print('best performing model: iteration ' + str(best_index)) 

### run scoring on whole test set 
with gs_drm[best_index].as_default() as g: 
    h_test = forwardSimpleTCModelDNN(Dt, num_hidden) 
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
rl_ridge_model_O, rl_ridge_model_D = pmodels.fit_rlearner(D, o, d2d, w) 

## one model for order lift and one model for cost drop 
pred_values_va_rlearner_O = rl_ridge_model_O.predict(Dt) 
pred_values_va_rlearner_C = rl_ridge_model_C.predict(Dt) 
pred_values_va_rlearner_D = rl_ridge_model_D.predict(Dt) 

#if ranking_model == 'effectiveness-ratio': ## if we use the effectiveness ratio model, compute effectiveness ratio 
pred_values_va_rlearner = np.divide(np.maximum(pred_values_va_rlearner_O, 0), pred_values_va_rlearner_C + 1e-7) + d2dlamb * pred_values_va_rlearner_D #pred_values_va_rlearner_O 

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
#plt.figure() 
rlearnerauccs = [] 

ranaucc, rancs, ranos = exp.AUC_cpit_cost_curve_deciles_cohort(ranscore, ot, wt, -1.0 * ct, 'k', plot_random=True) 
quasiaucc,  quasics, quasios = exp.AUC_cpit_cost_curve_deciles_cohort(pred_values_va_rlearner_O, ot, wt, -1.0 * ct, 'c') 
#for i in range(len(lambds)): 
#    rlearnerauccs.append(exp.AUC_cpit_cost_curve_deciles_cohort_d2dist(rlearnerscores[i], ot, wt, -1.0 * ct, d2dt, d2dlamb, colors[i] )) 
#cfaucc = exp.AUC_cpit_cost_curve_deciles_cohort(cfscore, ot, wt, -1.0 * ct, 'g') # causal forest aucc and plotting 
ctpmaucc, ctpmcs, ctpmos = exp.AUC_cpit_cost_curve_deciles_cohort(ctpmscore, ot, wt, -1.0 * ct, 'r' ) 
drmaucc, drmcs, drmos = exp.AUC_cpit_cost_curve_deciles_cohort(drmscore, ot, wt, -1.0 * ct, 'm' ) 

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
#for i in range(len(lambds)): 
#    leg_str.append('Duality R-learner') 
#leg_str.append('Causal Forest') # causal forest result 
leg_str.append('CTPM') 
leg_str.append('Simple CT Model') 
plt.legend(leg_str) 

### --- print out aucc results for different models --- 
print('AUCC results: ') 
print('random: ' + str(ranaucc)) 
print('rlearner: ' + str(quasiaucc)) 
#i = 0
#for rlearneraucc in rlearnerauccs: 
#    print('duality rlearner ' + str(i + 1) + ' with lambda = ' + str(lambds[i]) + ':' + str(rlearneraucc)) 
#    i = i + 1
#print('cf: ' + str(cfaucc)) 
print('drm: ' + str(drmaucc)) 
print('ctpm: ' + str(ctpmaucc)) 

print('showing cost-curve figure for iteration: ' + str(global_iter) ) 
#plt.show() 

#plt.figure() 

ranaumc, ranps, ranms = exp.AUC_cpit_cost_curve_deciles_cohort_d2dist(ranscore, ot, wt, -1.0 * ct, d2dt, d2dlamb, 'k', plot_random=True) 
quasiaumc, quasips, quasims = exp.AUC_cpit_cost_curve_deciles_cohort_d2dist(pred_values_va_rlearner_O, ot, wt, -1.0 * ct, d2dt, d2dlamb, 'c') 
#for i in range(len(lambds)): 
#    rlearnerauccs.append(exp.AUC_cpit_cost_curve_deciles_cohort_d2dist(rlearnerscores[i], ot, wt, -1.0 * ct, d2dt, d2dlamb, colors[i] )) 
#cfaucc = exp.AUC_cpit_cost_curve_deciles_cohort(cfscore, ot, wt, -1.0 * ct, 'g') # causal forest aucc and plotting 
ctpmaumc, ctpmps, ctpmms = exp.AUC_cpit_cost_curve_deciles_cohort_d2dist(ctpmscore, ot, wt, -1.0 * ct, d2dt, d2dlamb, 'r' ) 
drmaumc, drmps, drmms = exp.AUC_cpit_cost_curve_deciles_cohort_d2dist(drmscore, ot, wt, -1.0 * ct, d2dt, d2dlamb, 'm' ) 


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
#for i in range(len(lambds)): 
#    leg_str.append('Duality R-learner') 
#leg_str.append('Causal Forest') # causal forest result 
leg_str.append('CTPM') 
leg_str.append('Simple CT Model') 
plt.legend(leg_str) 

### --- print out aucc results for different models --- 
print('AUMC results: ') 
print('random: ' + str(ranaumc)) 
print('rlearner: ' + str(quasiaumc)) 
#i = 0
#for rlearneraumc in rlearneraumcs: 
#    print('duality rlearner ' + str(i + 1) + ' with lambda = ' + str(lambds[i]) + ':' + str(rlearneraumc)) 
#    i = i + 1
#print('cf: ' + str(cfaumc)) 
print('drm: ' + str(drmaumc)) 
print('ctpm: ' + str(ctpmaumc)) 

print('showing metric-curve figure for iteration: ' + str(global_iter) ) 
#plt.show() 

### --- saving data to results folder ---- 
save_filename = '../results/benchmarkwithcv_ctpm_drm_hte_'+prefix+'_main_results_global_iter_' + str(global_iter) + '.pkl' 
saveD = {
    'quasiscore':pred_values_va_rlearner, 
    'quasiscore_O':pred_values_va_rlearner_O, 
    'rlearnerscore':rlearnerscores, 
    'ot':ot, 
    'wt':wt, 
    'ct':ct, 
    'rlearnerauccs':rlearnerauccs, 
    'ranaucc':ranaucc, 
    'rancs':rancs, 
    'ranos':ranos, 
    'ranaumc':ranaumc, 
    'ranps':ranps,
    'ranms':ranms, 
    'quasiaucc':quasiaucc, 
    'quasics':quasics, 
    'quasios':quasios, 
    'quasiaumc':quasiaumc, 
    'quasips':quasips, 
    'quasims':quasims, 
    'ctpmscore':ctpmscore, 
    'drmscore':drmscore, 
    'ctpmaucc':ctpmaucc, 
    'ctpmcs':ctpmcs,
    'ctpmos':ctpmos, 
    'ctpmaumc':ctpmaumc, 
    'ctpmps':ctpmps, 
    'ctpmms':ctpmms, 
    'drmaucc':drmaucc, 
    'drmcs':drmcs, 
    'drmos':drmos, 
    'drmaumc':drmaumc, 
    'drmps':drmps, 
    'drmms':drmms, 
    'ctpm_val_result':ctpm_val_result, 
    'ctpm_test_result':ctpm_test_result, 
    'drm_val_result':drm_val_result}
    #'drm_test_result':drm_test_result} 
#'cfscore':cfscore, causal forest scores 

pkl.dump(saveD, open(save_filename, 'wb')) 

# In[ ]:
