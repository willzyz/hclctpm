import numpy as np
import pandas as pd

from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, Formula, numpy2ri, IntVector, FloatVector 
import rpy2.robjects as robjects

from DataProcFunctions import * 

D_tre, D_unt, Dv_tre, Dv_unt, Dt_tre, Dt_unt, o_tre, o_unt, ov_tre, ov_unt, ot_tre, ot_unt, c_tre, c_unt, cv_tre, cv_unt, ct_tre, ct_unt, D, w, o, c, Dv, wv, ov, cv, Dt, wt, ot, ct = LoadDataFromCsvP3('data/rxgy_training_data_v4_samp0.3') 


## make dataframe as input to causal forest 

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

def _generate_column_names(X):
    """
    Generate column names for X numpy array
    :param X:
    :return: list of names like [x1, x2]
    """
    return ['V{}'.format(i+1) for i in range(X.shape[1])]

def _formula(X):
    """
    Generate fit formula for R model
    :param X:
    :return: R formula object
    """
    dim_inputs = X.shape[1]
    
    return Formula('V{}~{}'.format(dim_inputs + 1, '+'.join(_generate_column_names(X))))

print('calling function')
rdf = _temp_rdf(D, o) 
print('D features: '  + str(D.shape[1]))

rdf.insert(D.shape[1] + 1, 'treatment', w) 

rdf.to_csv('causal_forest_r_data.csv') 
exit() 
#wb = 2 * (w - 0.5)
r_is_treatment = IntVector(w.astype(int))

causalTreeR = importr('causalTree') 

split_rule='CT'
minsize=10
split_bucket=True
bucket_max=20
node_size=1000
num_trees=50
sample_frac=1.0
sample_split_frac=0.8
split_alpha=0.5
dim_inputs = D.shape[1]
num_features = int(np.ceil(np.sqrt(D.shape[1])))

fmla = _formula(D) 

print('num_features:' + str(num_features))
print('fmla: ' + str(fmla)) 

print(r_is_treatment) 
print(sum(r_is_treatment)) 
print(sum(w)) 

import ipdb; ipdb.set_trace() 

model = causalTreeR.causalForest(
            fmla, data=rdf, treatment=r_is_treatment, split_Rule=split_rule, minsize=minsize,
            split_Bucket=split_bucket, bucketMax=bucket_max, nodesize=node_size,
            num_trees=num_trees, ncolx=dim_inputs, ncov_sample=num_features, split_alpha=split_alpha,
            sample_size_total=sample_frac, sample_size_train_frac=sample_split_frac
        ) 
