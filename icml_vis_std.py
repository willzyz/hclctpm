### code for utilizing linear HTE models (R-learner) and 
### Duality R-learner to form an example for visualization team 
### and other partner teams 

import numpy as np, pandas as pd, pickle as pkl 
from experimentation import * 

exp = Experimentation() 

d2dlamb = 0.1 
D = pkl.load(open('results/benchmarkwithcv_ctpm_drm_hte_ponfare_run5_std_bars_main_results_global_iter_1.pkl', 'rb'))
ot = D['ot']
wt = D['wt']
ct = D['ct']

filelist = ['results/benchmarkwithcv_ctpm_drm_hte_ponfare_run5_std_bars_main_results_global_iter_'+str(i)+'.pkl' for i in range(7, 13)]
best_ran_aucc, best_ran_cc_series_cs, best_ran_cc_series_os, best_ran_aumc, best_ran_cc_series_ps, best_ran_cc_series_ms, best_quasi_aucc, best_quasi_cc_series_cs, best_quasi_cc_series_os, best_quasi_aumc, best_quasi_cc_series_ps, best_quasi_cc_series_ms, best_ctpm_aucc, best_ctpm_cc_series_cs, best_ctpm_cc_series_os, best_ctpm_aumc, best_ctpm_cc_series_ps, best_ctpm_cc_series_ms, best_drm_aucc, best_drm_cc_series_cs, best_drm_cc_series_os, best_drm_aumc, best_drm_cc_series_ps, best_drm_cc_series_ms, ranos_std, ranms_std, quasios_std, quasims_std, ctpmos_std, ctpmms_std, drmos_std, drmms_std = exp.Aggregate_Curves_Best_std(filelist) 

min_cf_runs = 3
max_cf_runs = 8
cf_o_score_list = ['results/causal_forest_grf_test_set_results_O_ponfare_v3_causal_data_with_intensity_matching_d2dist_numtrees50_alpha0.2_min_node_size3_sample_fraction0.5_global_iter_'+str(i)+'.csv' for i in range(min_cf_runs, max_cf_runs)] 
cf_c_score_list = ['results/causal_forest_grf_test_set_results_C_ponfare_v3_causal_data_with_intensity_matching_d2dist_numtrees50_alpha0.2_min_node_size3_sample_fraction0.5_global_iter_'+str(i)+'.csv' for i in range(min_cf_runs, max_cf_runs)] 
cf_d_score_list = ['results/causal_forest_grf_test_set_results_D_ponfare_v3_causal_data_with_intensity_matching_d2dist_numtrees50_alpha0.2_min_node_size3_sample_fraction0.5_global_iter_'+str(i)+'.csv' for i in range(min_cf_runs, max_cf_runs)] 

best_cf_aucc, best_cf_cc_series_cs, best_cf_cc_series_os, best_cf_aumc, best_cf_cc_series_ps, best_cf_cc_series_ms, cfos_std, cfms_std = exp.Aggregate_CF_Curves_Best_std(cf_o_score_list, cf_c_score_list, cf_d_score_list, 0.1, wt, ot, ct) 

### ---- experimentation and plotting ----- 
#### AUMC ATETP plots #### 
from experimentation import * 
exp = Experimentation() 
ranscore = np.random.rand(ot.shape[0], ) 
colors = ['b', 'c', 'g', 'y', 'b', 'c', 'g', 'y', 'b', 'c', 'g', 'y', 'b', 'c', 'g', 'y'] 
plt.figure() 
plt.plot(best_ran_cc_series_ps, best_ran_cc_series_ms, '-o'+'k', markersize=12, linewidth=3) 
plt.xlim(1.0, 0)
yerr=quasims_std
plt.errorbar(best_quasi_cc_series_ps, best_quasi_cc_series_ms, yerr=yerr, ecolor='c', fmt='-oc', linewidth=3, markersize=12, elinewidth=1, capsize=5) 
plt.xlim(1.0, 0)
filter=np.arange(0, 20, 1) 
yerr=cfms_std[filter] 
yerr[0] = 0.0 
best_cf_cc_series_ms[0] = 0.0 
plt.errorbar(np.asarray(best_cf_cc_series_ps)[filter], np.asarray(best_cf_cc_series_ms)[filter], yerr=yerr, ecolor='g', fmt="-og", linewidth=3, markersize=12, elinewidth=1, capsize=5) 
plt.xlim(1.0, 0)
yerr=drmms_std
plt.errorbar(best_drm_cc_series_ps, best_drm_cc_series_ms, yerr=yerr, ecolor='m', fmt="-om", linewidth=3, markersize=12, elinewidth=1, capsize=5)
plt.xlim(1.0, 0)
yerr=ctpmms_std
plt.errorbar(best_ctpm_cc_series_ps, best_ctpm_cc_series_ms, yerr=yerr, ecolor='r', fmt="-or", linewidth=3, markersize=12, elinewidth=1, capsize=5)
plt.xlim(1.0, 0)
### --- add legeneds to plot ---- 
leg_str = ['Random'] 
leg_str.append('Quasi-oracle estimation (R-learner)') 
leg_str.append('Causal Forest') # causal forest result 
leg_str.append('Simple CT Model') 
leg_str.append('CTPM') 
plt.legend(leg_str) 
ax = plt.gca()
vals = ax.get_xticks()
ax.set_xticklabels(['{:,.1%}'.format(x) for x in vals])
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
plt.grid(True) 
plt.show() 

#### AUCC cost-curve plots #### 
plt.figure() 
plt.plot(best_ran_cc_series_cs, best_ran_cc_series_os, '-o'+'k', markersize=12, linewidth=3) 
yerr=np.concatenate((np.reshape(quasios_std, (1, -1)), np.reshape(np.minimum(np.reshape(quasios_std, (-1,1)), 1.0 - best_quasi_cc_series_os), (1, -1))), axis = 0) 
plt.errorbar(best_quasi_cc_series_cs, best_quasi_cc_series_os, yerr=yerr, ecolor='c', fmt='-oc', linewidth=3, markersize=12, elinewidth=1, capsize=5) 
yerr=np.concatenate((np.reshape(cfos_std, (1, -1)), np.reshape(np.minimum(np.reshape(cfos_std, (-1,1)), 1.0 - best_cf_cc_series_os), (1, -1))), axis = 0) 
plt.errorbar(best_cf_cc_series_cs, best_cf_cc_series_os, yerr=yerr, ecolor='g', fmt="-og", linewidth=3, markersize=12, elinewidth=1, capsize=5) 
yerr=np.concatenate((np.reshape(drmos_std, (1, -1)), np.reshape(np.minimum(np.reshape(drmos_std, (-1,1)), 1.0 - best_drm_cc_series_os), (1, -1))), axis = 0) 
plt.errorbar(best_drm_cc_series_cs, best_drm_cc_series_os, yerr=yerr, ecolor='m', fmt="-om", linewidth=3, markersize=12, elinewidth=1, capsize=5)
yerr=np.concatenate((np.reshape(ctpmos_std, (1, -1)), np.reshape(np.minimum(np.reshape(ctpmos_std, (-1,1)), 1.0 - best_ctpm_cc_series_os), (1, -1))), axis = 0) 
plt.errorbar(best_ctpm_cc_series_cs, best_ctpm_cc_series_os, yerr=yerr, ecolor='r', fmt="-or", linewidth=3, markersize=12, elinewidth=1, capsize=5) 

### --- add legeneds to plot ---- 
leg_str = ['Random'] 
leg_str.append('Quasi-oracle estimation (R-learner)') 
leg_str.append('Causal Forest') # causal forest result 
leg_str.append('Simple CT Model') 
leg_str.append('CTPM') 
plt.legend(leg_str) 
ax = plt.gca()
vals = ax.get_xticks()
ax.set_xticklabels(['{:,.1%}'.format(x) for x in vals])
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
plt.grid(True) 
plt.show() 

### --- print out aucc results for different models --- 
print('AUCC results: ') 
print('random: ' + str(best_ran_aucc)) 
print('rlearner: ' + str(best_quasi_aucc)) 
print('cf: ' + str(best_cf_aucc)) 
print('drm: ' + str(best_drm_aucc)) 
print('ctpm: ' + str(best_ctpm_aucc)) 

### --- print out aucc results for different models --- 
print('AUMC results: ') 
print('random: ' + str(best_ran_aumc)) 
print('rlearner: ' + str(best_quasi_aumc)) 
print('cf: ' + str(best_cf_aumc)) 
print('drm: ' + str(best_drm_aumc)) 
print('ctpm: ' + str(best_ctpm_aumc)) 

