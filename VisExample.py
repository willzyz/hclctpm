### code for utilizing linear HTE models (R-learner) and 
### Duality R-learner to form an example for visualization team 
### and other partner teams 

import numpy as np, pandas as pd, pickle as pkl 
#from ModelDefinitions import * 
from DataProcFunctions import * 

use_python3 = True 
temp = 0.8 
p_quantile = 0.15 ## percentage of users to target 
num_optimize_iterations = 3000 
num_modeling_inits = 3 
num_hidden = 0 

### -- load and segment out dataset 
### -- treated / untreated 
D_tre, D_unt, Dv_tre, Dv_unt, Dt_tre, Dt_unt, o_tre, o_unt, ov_tre, ov_unt, ot_tre, ot_unt, c_tre, c_unt, cv_tre, cv_unt, ct_tre, ct_unt, D, w, o, c, Dv, wv, ov, cv, Dt, wt, ot, ct = LoadDataFromPkl('data/rxgy_ma_training_data_v5_2019_07_08_vc_tr_featuremod3', use_python3=use_python3, frac=0.2) 
#rxgy_ma_training_data_v5_2019_07_08_vc_featuremod3

### D, w, c, o: training data [features], [treatment], [cost outcome], [gain outcome] 
### Dv, wv, cv, ov: validation data [features], [treatment], [cost outcome], [gain outcome] 
### Dt, wt, ct, ot: test data [features], [treatment], [cost outcome], [gain outcome] 

### ---- train hte model for comparison ---- 
### we could utimize the original HTE functions 
from LinearHTEModels import * 
lhmodels = LinearHTEModels() 
lambds = [0.005, 0.05] 
rlearnerscores = [] 
rl_ridge_model_L_list = [] 
## set-up lagrangian rlearner 
for i in range(len(lambds)): 
    lambd = lambds[i] 
    rl_ridge_model_L = lhmodels.fit_rlearner_lagrangian(D, o, c, w, lambd) 
    rl_ridge_model_L_list.append(rl_ridge_model_L) 
    rlearnerscores.append(rl_ridge_model_L.predict(Dt)) 

### ---- experimentation and plotting cost-curves ----- 
from experimentation import * 
### - visualization: lift curves: 
exp = Experimentation() 
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

### - visualization: cost curves: 
plt.figure() 
exp.AUC_cpit_cost_curve_deciles_cohort(ranscore, ot, wt, -1.0 * ct, 'k' ) 
for i in range(len(lambds)): 
    exp.AUC_cpit_cost_curve_deciles_cohort(rlearnerscores[i], ot, wt, -1.0 * ct, colors[i] ) 
plt.title('Causal learning cost curves using targeting models') 

leg_str = ['random'] 
for i in range(len(lambds)): 
    leg_str.append('Quasi Oracle Ridge-reg (lamb=' +str(lambds[i])+ ', Duality-rlearner)') 
leg_str.append('Top Quantile Ranking at ' + str(p_quantile*100) + '%') 
leg_str.append('Direct Ranking Model') 
plt.legend(leg_str) 
plt.show() 
