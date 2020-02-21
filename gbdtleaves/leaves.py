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
Db, Dvb = TransformTreeFeatures(D, Dv, o, ov) 
Db = np.reshape(Db, (Db.shape[0], -1)) 
Dvb = np.reshape(Dvb, (Dvb.shape[0], -1)) 

Dnew = np.concatenate((D, Db), axis = 1) 
Dvnew = np.concatenate((Dv, Dvb), axis = 1) 

train_nnReg(D, o, 10, 0.0, regtype='None') 

from sklearn.linear_model import Ridge #LogisticRegression 

#lr_model = LogisticRegression(penalty = 'l1', solver = 'saga', max_iter=300) 
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
