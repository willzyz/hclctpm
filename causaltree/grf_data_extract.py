import numpy as np 
import pandas as pd 

#from rpy2.robjects.packages import importr 
#from rpy2.robjects import pandas2ri, Formula, numpy2ri, IntVector, FloatVector 
#import rpy2.robjects as robjects 

from DataProcFunctions import * 

prefix = '' 
data_filename =  'data/rxgy_ma_training_data_v5_2019_07_08_vc_tr_featuremod2' 

D_tre, D_unt, Dv_tre, Dv_unt, Dt_tre, Dt_unt, o_tre, o_unt, ov_tre, ov_unt, ot_tre, ot_unt, c_tre, c_unt, cv_tre, cv_unt, ct_tre, ct_unt, D, w, o, c, Dv, wv, ov, cv, Dt, wt, ot, ct = LoadDataFromPkl(data_filename) 

## make dataframe as input to causal forest 
import ipdb; ipdb.set_trace() 

def _temp_rdf(X, y=None):
    """
    Create temporary pandas data frame for R model training and predict
    :param X:
    :param y:
    :return: temp data frame in R
    """
    
    # must use float, otherwise will fail during numpy2ri.py2ri
    mat = X.astype(float)
    
    # insert y to the last column of the R dataframe
    if y is not None:
        mat = np.concatenate([X, y.reshape([-1, 1])], axis=1).astype(float)
    
    df = pd.DataFrame(mat, columns=['V{}'.format(i+1) for i in range(mat.shape[1])]) 
    
    #df = df.sample(frac = 1.0) 
    
    #pandas2ri.activate()
    #rdf = robjects.DataFrame(df)
    #pandas2ri.deactivate()
    #import ipdb; ipdb.set_trace() 
    rdf = df 
    
    return rdf 

print('D features: '  + str(D.shape[1]))

rdf = _temp_rdf(D, o) 
rdf.insert(D.shape[1] + 1, 'cost', c) 
rdf.insert(D.shape[1] + 2, 'treatment', w) 

rdf.to_csv('causal_forest_r_data_' + prefix + '_train.csv') 

rdf = _temp_rdf(Dv, ov) 
rdf.insert(Dv.shape[1] + 1, 'cost', cv) 
rdf.insert(Dv.shape[1] + 2, 'treatment', wv) 

rdf.to_csv('causal_forest_r_data_' + prefix + '_valid.csv') 

rdf = _temp_rdf(Dt, ot) 
rdf.insert(Dt.shape[1] + 1, 'cost', ct) 
rdf.insert(Dt.shape[1] + 2, 'treatment', wt) 

rdf.to_csv('causal_forest_r_data_' + prefix + '_test.csv') 

import os, sys 

#os.system('head -n 100000 causal_forest_r_data_rxgyfixvc_train.csv > causal_forest_r_data_rxgyfixvc_train_tiny.csv') 
#os.system('head -n 300000 causal_forest_r_data_rxgyfixvc_train.csv > causal_forest_r_data_rxgyfixvc_train_small.csv') 
