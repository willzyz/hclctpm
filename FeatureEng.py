import numpy as np, tensorflow as tf, pandas as pd 
import xgboost as xgb 

### module with feature engineering functions 

def TreeIndicesToBinaryFeatures(Dtr_ind, Dval_ind, tree_depth): 
    
    ncols = np.power(2, tree_depth + 1) 
    
    Dtr_bin = np.zeros((Dtr_ind.size, ncols)) 
    Dtr_bin[np.arange(Dtr_ind.size), Dtr_ind.ravel()] = 1 
    Dtr_bin.shape = Dtr_ind.shape + (ncols,)

    Dval_bin = np.zeros((Dval_ind.size, ncols)) 
    Dval_bin[np.arange(Dval_ind.size), Dval_ind.ravel()] = 1 
    Dval_bin.shape = Dval_ind.shape + (ncols,)
    
    return Dtr_bin, Dval_bin 

def TransformTreeFeatures(Dtr, Dval, Ltr, Lval): 
    ## implements tree-leaves based feature transformation 
    ## Dtr: training data row-major features 
    ## Dval: validation data row-major features (not used for validation, returns with added features) 
    ## Ltr: training labels 
    
    tree_depth = 3 
    num_trees = 20 
    ## first fit xgboost gbdt to Dtr 
    regressor = xgb.XGBRegressor( 
        max_depth = tree_depth, 
        n_estimators = num_trees 
        ) 
    
    import time
    start = time.time()
    bst = regressor.fit(Dtr, Ltr) 
    end = time.time() 
    print('batch training elapsed time: ' + str(end - start)) 
    
    cols = ['f'+str(x) for x in range(Dtr.shape[1])] 
    
    pd_Dtr = pd.DataFrame(data = Dtr, index = range(Dtr.shape[0]), columns = cols) 
    dm_Dtr = xgb.DMatrix(pd_Dtr) 
    
    pd_Dval = pd.DataFrame(data = Dval, index = range(Dval.shape[0]), columns = cols) 
    dm_Dval = xgb.DMatrix(pd_Dval) 
    
    Dtr_tree_indices = bst.get_booster().predict(dm_Dtr, pred_leaf=True) 
    Dval_tree_indices = bst.get_booster().predict(dm_Dval, pred_leaf=True) 
    
    ptr = bst.predict(Dtr) 
    tr_mse = 1.0 / len(Ltr) * np.sum(np.square(ptr - Ltr)) 
    pval = bst.predict(Dval) 
    val_mse = 1.0 / len(Lval) * np.sum(np.square(pval - Lval)) 
    print('gbdt training mse: ' + str(tr_mse)) 
    print('gbdt validation mse: ' + str(val_mse)) 
    
    Dtr_bin, Dval_bin = TreeIndicesToBinaryFeatures(Dtr_tree_indices, Dval_tree_indices, tree_depth) 
    
    return Dtr_bin, Dval_bin 

##----------- code back-log ----------------- 
#pd_Dtr = pd.DataFrame(data = Dtr, index = range(Dtr.shape[0]), columns = Dtr[0, :])
#dtrain = xgb.DMatrix(pd_Dtr, Ltr) 
#params = {'max_depth':2, 'eta':1} 
#'reg:squarederror' }
#'reg:squarederror
#num_round = 2 
#bst = xgb.train(params, dtrain, num_round) 
#preds = bst.predict(Dval, pred_leaf=True) 
