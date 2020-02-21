import pandas as pd 

import numpy as np, tensorflow as tf, pandas as pd, pickle as pkl 
from ModelDefinitions import * 
from DataProcFunctions import * 

#### experiment results, all plots 

plot_num = 1 

if plot_num == 1: 
    loadD = pkl.load(open('results/benchmarkwithcv_tqr_drm_hte_rxgy_v5_07_08_featuremod3_tr_iter2000_run7_goodrun-rerun_lessiters_main_result_run3.pkl')) 
    D_tre, D_unt, Dv_tre, Dv_unt, Dt_tre, Dt_unt, o_tre, o_unt, ov_tre, ov_unt, ot_tre, ot_unt, c_tre, c_unt, cv_tre, cv_unt, ct_tre, ct_unt, D, w, o, c, Dv, wv, ov, cv, Dt, wt, ot, ct = LoadDataFromPkl('data/rxgy_ma_training_data_v5_2019_07_08_vc_tr_featuremod3') 
elif plot_num == 2: 
    loadD = pkl.load(open('results/benchmarkwithcv_tqr_drm_hte_uscensus_pub_run_near_final_iter600_schedule_run2_main_results.pkl')) 
    D_tre, D_unt, Dv_tre, Dv_unt, Dt_tre, Dt_unt, o_tre, o_unt, ov_tre, ov_unt, ot_tre, ot_unt, c_tre, c_unt, cv_tre, cv_unt, ct_tre, ct_unt, D, w, o, c, Dv, wv, ov, cv, Dt, wt, ot, ct = LoadDataFromPkl('data/uscensus_pub_causal_data') 
else: 
    loadD = pkl.load(open('results/benchmarkwithcv_tqr_drm_hte_covtype_pub_rerun_main_result_run2_main_results.pkl')) 
    D_tre, D_unt, Dv_tre, Dv_unt, Dt_tre, Dt_unt, o_tre, o_unt, ov_tre, ov_unt, ot_tre, ot_unt, c_tre, c_unt, cv_tre, cv_unt, ct_tre, ct_unt, D, w, o, c, Dv, wv, ov, cv, Dt, wt, ot, ct = LoadDataFromPkl('data/covtype_pub_causal_data') 
tqrscore = loadD['tqrscore'] 
drmscore = loadD['drmscore'] 
quasiscore = loadD['quasiscore'] 
quasiscore_O = loadD['quasiscore_O'] 
rlearnerscores = loadD['rlearnerscore'] 
p_quantile = 0.40 
lambds = [0.1] 

if plot_num == 1: 
    ot_cf = pd.read_csv('results/causal_forest_grf_test_set_results_O_rxgy_ma_training_data_v5_2019_07_08_vc_tr_featuremod3_numtrees50_alpha0.2_min_node_size3_sample_fraction0.5.csv') 
    ct_cf = pd.read_csv('results/causal_forest_grf_test_set_results_C_rxgy_ma_training_data_v5_2019_07_08_vc_tr_featuremod3_numtrees50_alpha0.2_min_node_size3_sample_fraction0.5.csv') 
elif plot_num == 2: 
    ot_cf = pd.read_csv('results/causal_forest_grf_test_set_results_O_uscensus_pub_causal_data_numtrees100_alpha0.2_min_node_size3_sample_fraction0.5.csv') 
    ct_cf = pd.read_csv('results/causal_forest_grf_test_set_results_C_uscensus_pub_causal_data_numtrees100_alpha0.2_min_node_size3_sample_fraction0.5.csv') 
else: 
    ot_cf = pd.read_csv('results/causal_forest_grf_test_set_results_O_covtype_pub_causal_data_numtrees100_alpha0.2_min_node_size3_sample_fraction0.5.csv') 
    ct_cf = pd.read_csv('results/causal_forest_grf_test_set_results_C_covtype_pub_causal_data_numtrees100_alpha0.2_min_node_size3_sample_fraction0.5.csv')

ot_cf = ot_cf.as_matrix() 
Ocfscore = ot_cf[0][1:]

ct_cf = ct_cf.as_matrix() 
Ccfscore = ct_cf[0][1:]

cfscore = np.divide(Ocfscore, Ccfscore) 

### ---- experimentation and plotting cost-curves ----- 
from experimentation import * 
exp = Experimentation() 
ranscore = np.random.rand(ot.shape[0], ) 
colors = ['b', 'c', 'g', 'y'] 
plt.figure() 
rlearnerauccs = [] 
ranaucc = exp.AUC_cpit_cost_curve_deciles_cohort(ranscore, ot, wt, -1.0 * ct, 'k', plot_random=True) 
quasiaucc = exp.AUC_cpit_cost_curve_deciles_cohort(quasiscore_O, ot, wt, -1.0 * ct, 'c') 
for i in range(len(lambds)): 
    rlearnerauccs.append(exp.AUC_cpit_cost_curve_deciles_cohort(rlearnerscores[i], ot, wt, -1.0 * ct, colors[i] )) 
cfaucc = exp.AUC_cpit_cost_curve_deciles_cohort(cfscore, ot, wt, -1.0 * ct, 'g') 
tqraucc = exp.AUC_cpit_cost_curve_deciles_cohort(tqrscore, ot, wt, -1.0 * ct, 'r' ) 
drmaucc = exp.AUC_cpit_cost_curve_deciles_cohort(drmscore, ot, wt, -1.0 * ct, 'm' ) 
plt.title('Causal learning cost curves using targeting models') 

### --- add legeneds to plot ---- 
leg_str = ['Random'] 
leg_str.append('R-learner on Incremental Gain') 
for i in range(len(lambds)): 
    leg_str.append('Duality R-learner') 
leg_str.append('Causal Forest') 
leg_str.append('Top Quantile Ranking')# + str(p_quantile*100) + '%') 
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
print('cf: ' + str(cfaucc)) 
print('drm: ' + str(drmaucc)) 
print('tqr: ' + str(tqraucc)) 

plt.show() 

### - visualization: lift curves: 
plt.figure() 
ranscore = np.random.rand(rlearnerscores[0].shape[0], ) 
colors = ['b', 'c', 'g'] 
plt.figure()  
exp.AUC_ivpu(ranscore, ot, wt, np.arange(-5, 5, 0.1), 'k', -1.0 *ct)
for i in range(len(lambds)): 
    exp.AUC_ivpu(rlearnerscores[i], ot, wt, np.arange(-5, 5, 0.1), colors[i], -1.0 *ct)
plt.title('Causal learning lift curves using targeting models') 

leg_str = ['random'] 
for i in range(len(lambds)): 
    leg_str.append('Quasi Oracle Ridge-reg (lamb=' +str(lambds[i])+ ', Duality-rlearner)') 
leg_str.append('Top Quantile Ranking at ' + str(p_quantile*100) + '%') 
leg_str.append('Direct Ranking Model') 
plt.legend(leg_str) 
plt.show() 
