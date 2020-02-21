### code implements ranking model for treatment effect 
### for optimizing with respect to CPIT/CPIGB or corresponding 
### lagrangian objectives using tensorflow 

import numpy as np, tensorflow as tf, pandas as pd, pickle as pkl 
from ModelDefinitions import * 

Dd = pkl.load(open('data/drm_train_data_from_hscls547_allcohorts11weeks_sav__seq_start0.pkl', 'r')) 

### parsing data from saved pickle 
D = Dd['nX_tr']; w = Dd['w_tr']; ni = Dd['n9d_ni_usd_tr']; o = Dd['values_tr']; c = -1.0 * ni 
Dv = Dd['nX_va']; wv = Dd['w_va']; niv = Dd['n9d_ni_usd_va']; ov = Dd['values_va']; cv = -1.0 * niv 

if type(c) != type(o): 
    c = c.as_matrix()
if type(cv) != type(ov):
    cv = cv.as_matrix() 

if type(ni) != type(o): 
    ni = ni.as_matrix()
if type(cv) != type(ov):
    niv = niv.as_matrix() 

if w.shape[-1] == 1: 
    w = np.reshape(w, (len(w), )) 
if wv.shape[-1] == 1: 
    wv = np.reshape(wv, (len(wv), )) 
if o.shape[-1] == 1: 
    o = np.reshape(o, (len(o), )) 
if c.shape[-1] == 1: 
    c = np.reshape(c, (len(c), ))  
if ov.shape[-1] == 1: 
    ov = np.reshape(ov, (len(ov), )) 
if cv.shape[-1] == 1: 
    cv = np.reshape(cv, (len(cv), )) 

### - todo: put this in DataProcFunctions 
# [todo: start] 
filter = ~ np.isnan(c) 
D = D[np.where(filter==True)[0], :] 
w = w[np.where(filter==True)[0]] 
c = c[np.where(filter==True)[0]] 
o = o[np.where(filter==True)[0]] 
filter = ~ np.isnan(cv) 
Dv = Dv[np.where(filter==True)[0], :] 
wv = wv[np.where(filter==True)[0]] 
cv = cv[np.where(filter==True)[0]] 
ov = ov[np.where(filter==True)[0]] 

D_tre = D[w > 0.5, :] 
D_unt = D[w < 0.5, :] 

Dv_tre = Dv[wv > 0.5, :] 
Dv_unt = Dv[wv < 0.5, :] 

o_tre = o[w > 0.5] 
o_unt = o[w < 0.5] 

ov_tre = ov[wv > 0.5] 
ov_unt = ov[wv < 0.5] 

c_tre = c[w > 0.5] 
c_unt = c[w < 0.5] 

cv_tre = cv[wv > 0.5] 
cv_unt = cv[wv < 0.5] 

print(np.average(c_tre)); print(np.average(c_unt)); print(np.average(o_tre)); print(np.average(o_unt)) 

from FeatureEng import * 
#Db, Dvb = TransformTreeFeatures(D, Dv, o, ov) 
#Db = np.reshape(Db, (Db.shape[0], -1)) 
#Dvb = np.reshape(Dvb, (Dvb.shape[0], -1)) 

#Dnew = np.concatenate((D, Db), axis = 1) 
#Dvnew = np.concatenate((Dv, Dvb), axis = 1) 

#train_nnReg(D, o, 10, 0.0, regtype='None') 

from sklearn.linear_model import Ridge #LogisticRegression 

#lr_model = LogisticRegression(penalty = 'l1', solver = 'saga', max_iter=300) 

"""
lr_model = Ridge() 

lr_model.fit(D, o) 
ptr = lr_model.predict(D) 
pval = lr_model.predict(Dv) 

tr_mse = 1.0 / len(o) * np.sum(np.square(ptr - o)) 
val_mse = 1.0 / len(ov) * np.sum(np.square(pval - ov)) 
print('linear regression with l2 training mse: ' + str(tr_mse)) 
print('linear regression with l2 validation mse: ' + str(val_mse)) 

print('training lr with sparse bin features: ... ') 
lr_model.fit(Dnew, o) 
ptr = lr_model.predict(Dnew) 
pval = lr_model.predict(Dvnew) 
print('done') 

tr_mse = 1.0 / len(o) * np.sum(np.square(ptr - o)) 
print('predicting lr with sparse bin features: ... ') 
val_mse = 1.0 / len(ov) * np.sum(np.square(pval - ov)) 
print('linear regression with l2 + bin features training mse: ' + str(tr_mse)) 
print('linear regression with l2 + bin features validation mse: ' + str(val_mse)) 
""" 

# !conda install -yc conda-forge xgboost 
import xgboost as xgb 
import sklearn.datasets 
import sklearn.metrics 
import sklearn.feature_selection 
import sklearn.feature_extraction 
import sklearn.model_selection 

xgb.__version__ 

#df = sklearn.datasets.load_boston() 
#print(df.keys()) 
#print(df['feature_names']) 

#X = df['data'] 
#y = df['target'] 

#x_tr, x_te, y_tr, y_te = sklearn.model_selection.train_test_split(X, y) 

x_tr = D 
x_te = Dv 
y_tr = o 
y_te = ov 

batch_size = len(x_tr) / 10 #100000 
print('training with batch_size : ' + str(batch_size)) 
iterations = 4
tree_depth = 6 
num_trees = 10 

total_num_trees = num_trees * 10 * iterations 
print('total number of boosted trees : ' + str(total_num_trees)) 

#model = None 
import time 
start_time = time.time() 
print('start incremental training ... : ') 
for i in range(iterations): 
    for start in range(0, len(x_tr), batch_size): 
        if i == 0 and start == 0: 
            model = xgb.train({ 
                    'max_depth' : tree_depth, 
                    'learning_rate': 0.009, #0.07, ##0.007, 
                    #'update':'refresh',
                    #'process_type': 'update',
                    #'refresh_leaf': True,
                    'reg_alpha': 3,  # L1 
                    'silent': False, 
                    }, dtrain=xgb.DMatrix(x_tr[start:start+batch_size], y_tr[start:start+batch_size]), num_boost_round=num_trees) 
        else: 
            model = xgb.train({ 
                    'max_depth' : tree_depth, 
                    'learning_rate': 0.009, #0.007, 
                    'update':'refresh',
                    #'process_type': 'update',
                    'refresh_leaf': True,
                    'reg_alpha': 3,  # L1 
                    'silent': False, 
                    }, dtrain=xgb.DMatrix(x_tr[start:start+batch_size], y_tr[start:start+batch_size]), num_boost_round=num_trees, xgb_model=model) 
        end = time.time() 
        print('iteratiton training elapsed time: ' + str(end - start_time)) 
        
        y_pr = model.predict(xgb.DMatrix(x_te)) 
        #test_pred_leaf = model.predict(xgb.DMatrix(x_te), pred_leaf=True) 
        
        #print('    MSE itr@{}: {}'.format(int(start/batch_size), sklearn.metrics.mean_squared_error(y_te, y_pr))) 
    print('MSE itr@{}: {}'.format(i, sklearn.metrics.mean_squared_error(y_te, y_pr))) 

y_pr = model.predict(xgb.DMatrix(x_te)) 
print('MSE at the end: {}'.format(sklearn.metrics.mean_squared_error(y_te, y_pr))) 

end = time.time() 
print('incremental training elapsed time: ' + str(end - start_time)) 
