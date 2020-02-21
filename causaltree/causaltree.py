import numpy as np, pandas as pd, pickle as pkl #tensorflow as tf, 
#from ModelDefinitions import * 
from DataProcFunctions import * 

#Dd = pkl.load(open('data/rxgy_training_data_v4_samp0.3', 'r')) 
#Dd = pkl.load(open('data/rxgy_training_data_v4_samp0.3', 'rb'), encoding="latin1") 

from causalml.dataset import make_uplift_classification 
from causalml.inference.tree import UpliftRandomForestClassifier 
from causalml.inference.tree import UpliftTreeClassifier 
from causalml.metrics import plot_gain 

D_tre, D_unt, Dv_tre, Dv_unt, Dt_tre, Dt_unt, o_tre, o_unt, ov_tre, ov_unt, ot_tre, ot_unt, c_tre, c_unt, cv_tre, cv_unt, ct_tre, ct_unt, D, w, o, c, Dv, wv, ov, cv, Dt, wt, ot, ct = LoadDataFromCsvP3('data/rxgy_training_data_v4_samp0.3') 

#data = pd.DataFrame(np.matrix()) 
#data.to_csv('D.csv') 
#exit()

uplift_model_o = UpliftTreeClassifier(max_depth=5, min_samples_leaf=200, min_samples_treatment=50,
                                    n_reg=100, evaluationFunction='KL', control_name='control')

def text_map_w(w_i):
    if w_i == 0:
        val = 'control'
    else:
        val = 'treatment'
    return val

wtext = [text_map_w(w_i) for w_i in w]
data = {'treatment_group_key':wtext, 'trips':o}
dframe = pd.DataFrame(data)
print(dframe)
uplift_model_o.fit(D, treatment=dframe['treatment_group_key'].values, y=dframe['trips'].values)
