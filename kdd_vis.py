### code for utilizing linear HTE models (R-learner) and 
### Duality R-learner to form an example for visualization team 
### and other partner teams 

import numpy as np, pandas as pd, pickle as pkl 
#from ModelDefinitions import * 
#from DataProcFunctions import * 


lambds = [0.75] 
D = pkl.load(open('results/benchmarkwithcv_tqr_drm_hte_r2e_v5_07_08_featuremod3_tr_iter100_run7_main_results.pkl', 'rb'))
tqrscore = D['tqrscore']
drmscore = D['drmscore']
pred_values_va_rlearner = D['quasiscore'] 
pred_values_va_rlearner_O = D['quasiscore_O'] 
rlearnerscores = D['rlearnerscore']
ot = D['ot']
wt = D['wt']
ct = D['ct']
tempvalue = D['tempvalue']
p_quantilevalue = D['p_quantilevalue'] 
tqraucc = D['tqraucc']
drmaucc = D['drmaucc']
rlearnerauccs = D['rlearnerauccs']
ranaucc = D['ranaucc']
quasiaucc = D['quasiaucc']

### this section is to load the results trained by grf R code 
### 

ot_cf = pd.read_csv('results/causal_forest_grf_test_set_results_O_data_numtrees50_alpha_min_node_size_sample_fraction.csv') 
ct_cf = pd.read_csv('results/causal_forest_grf_test_set_results_C_data_numtrees50_alpha_min_node_size_sample_fraction.csv')

ot_cf = ot_cf.as_matrix() 
Ocfscores = ot_cf[0][1:]

ct_cf = ct_cf.as_matrix() 
Ccfscores = ct_cf[0][1:]

cfscore = np.divide(Ocfscores, Ccfscores) 

### ---- experimentation and plotting cost-curves ----- 
from experimentation import * 
exp = Experimentation() 
ranscore = np.random.rand(ot.shape[0], ) 
colors = ['b', 'c', 'g', 'y', 'b', 'c', 'g', 'y', 'b', 'c', 'g', 'y', 'b', 'c', 'g', 'y'] 
plt.figure() 
rlearnerauccs = [] 
ranaucc = exp.AUC_cpit_cost_curve_deciles_cohort(ranscore, ot, wt, -1.0 * ct, 'k', plot_random=True) 
quasiaucc = exp.AUC_cpit_cost_curve_deciles_cohort(pred_values_va_rlearner_O, ot, wt, -1.0 * ct, 'c') 
for i in range(len(lambds)): 
    rlearnerauccs.append(exp.AUC_cpit_cost_curve_deciles_cohort(rlearnerscores[i], ot, wt, -1.0 * ct, colors[i] )) 
cfaucc = exp.AUC_cpit_cost_curve_deciles_cohort(cfscore, ot, wt, -1.0 * ct, 'g') # causal forest aucc and plotting 
tqraucc = exp.AUC_cpit_cost_curve_deciles_cohort(tqrscore, ot, wt, -1.0 * ct, 'r' ) 
drmaucc = exp.AUC_cpit_cost_curve_deciles_cohort(drmscore, ot, wt, -1.0 * ct, 'm' ) 
plt.title('Causal learning cost curves using targeting models') 

### --- saving data to results folder ---- 
#save_filename = '../results/benchmarkwithcv_tqr_drm_hte_'+prefix+'_main_results.pkl' 
#saveD = {'tqrscore':tqrscore, 'drmscore':drmscore, 'quasiscore':pred_values_va_rlearner, 'quasiscore_O':pred_values_va_rlearner_O, 'rlearnerscore':rlearnerscores, 'ot':ot, 'wt':wt, 'ct':ct, 'tempvalue':tempvalue, 'p_quantilevalue':p_quantilevalue, 
#         'tqraucc':tqraucc, 'drmaucc':drmaucc, 'rlearnerauccs':rlearnerauccs, 'ranaucc':ranaucc, 'quasiaucc':quasiaucc} 
#'cfscore':cfscore, causal forest scores 
#pkl.dump(saveD, open(save_filename, 'wb')) 

### --- add legeneds to plot ---- 
leg_str = ['Random'] 
leg_str.append('R-learner on Incremental Gain') 
for i in range(len(lambds)): 
    leg_str.append('Duality R-learner') # lambda='+str(lambds[i])) 
leg_str.append('Causal Forest') # causal forest result 
leg_str.append('Top Quantile Ranking') # at ' + str(p_quantile*100) + '%') 
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
