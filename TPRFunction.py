### code implements ranking model for treatment effect 
### for optimizing with respect to CPIT/CPIGB or corresponding 
### lagrangian objectives using tensorflow 

import numpy as np, tensorflow as tf, pandas as pd, pickle as pkl 
from ModelDefinitions import * 

def TPRFunction(hsclsvalscores, num_optimize_iterations): 
    #Dd = pkl.load(open('data/LCLatamAug2018TrainingData', 'r')) 
    Dd = pkl.load(open('data/drm_train_data_from_hscls262582_allcohorts11weeks___seq_start0.pkl', 'r'))
    #drm_train_data_from_hscls547_allcohorts11weeks_sav__seq_start0.pkl', 'r')) 
    #drm_train_data_from_hscls262582_allcohorts11weeks___seq_start0.pkl', 'r')) 
    #
    
    p_quantile = 0.25 ## percentage of users to target 
    #num_optimize_iterations = 800 
    
    ### parsing data from saved pickle 
    D = Dd['nX_tr']; w = Dd['w_tr']; ni = Dd['n9d_ni_usd_tr']; o = Dd['values_tr']; c = -1.0 * ni 
    Dv = Dd['nX_va']; wv = Dd['w_va']; niv = Dd['n9d_ni_usd_va']; ov = Dd['values_va']; cv = -1.0 * niv 
    
    D = D[:, 0:18] 
    Dv = Dv[:, 0:18] 
    
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
    
    # [todo: end] 
    
    obj, opt, dumh, dumhu = TopPRankingModel(D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, p_quantile, 'train-first') 
    
    ### session definitions and variable initialization 
    sess = tf.Session() 
    ### initialize variables and run optimization 
    init = tf.global_variables_initializer() 
    sess.run(init) 
    for step in range(num_optimize_iterations): 
        print('step : ' + str(step)) 
        _, objres = sess.run([opt, obj]) 
        print(objres) 
    
    ### evaluate CPIT metric on validation set 
    objv, dumo, dumh, dumhu = TopPRankingModel(Dv_tre, Dv_unt, ov_tre, ov_unt, cv_tre, cv_unt, p_quantile, 'eval') 
    
    print('TPR validation CPIT:') 
    print(sess.run(objv)) 

    ### run scoring on whole validation set 
    with tf.variable_scope("toppranker") as scope: 
        h_tre = tf.contrib.layers.fully_connected(Dv, 1, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope) 
    topprnkscore = sess.run(h_tre) 
    
    ### ---- train cpit ranking model for comparison --- 
    objc, optc, dumh, dumu = DirectRankingModel(D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, 'train-first-cpit') 
        
    ### initialize variables and run optimization 
    init = tf.global_variables_initializer() 
    sess.run(init) 
    for step in range(num_optimize_iterations): 
        print('step : ' + str(step)) 
        _, objcres = sess.run([optc, objc]) 
        print(objcres) 
    
    ### evaluate CPIT metric on validation set 
    objv, dumo, dumh, dumhu = DirectRankingModel(Dv_tre, Dv_unt, ov_tre, ov_unt, cv_tre, cv_unt, 'eval-cpit') 
    
    print('DRM validation CPIT:') 
    print(sess.run(objv)) 
    
    ### run scoring on whole validation set 
    with tf.variable_scope("cpitranker") as scope: 
        h_tre = tf.contrib.layers.fully_connected(Dv, 1, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope) 
    cpitrnkscore = sess.run(h_tre) 
    
    ### ---- train hte model for comparison ---- 
    from LinearHTEModels import * 
    lhmodels = LinearHTEModels() 
    lambds = [0.05] #, 0.005, 0.0005] 
    rlearnerscores = [] 
    rl_ridge_model_L_list = [] 
    ## set-up lagrangian rlearner 
    for i in range(len(lambds)): 
        lambd = lambds[i] 
        rl_ridge_model_L = lhmodels.fit_rlearner_lagrangian(D, o, c, w, lambd) 
        rl_ridge_model_L_list.append(rl_ridge_model_L) 
        rlearnerscores.append(rl_ridge_model_L.predict(Dv)) 
    
    ### ---- experimentation and plotting cost-curves ----- 
    from Experimentation import * 
    exp = Experimentation() 
    ranscore = np.random.rand(topprnkscore.shape[0], ) 
    colors = ['b', 'c', 'g'] 
    plt.figure() 
    exp.AUC_cpit_cost_curve_deciles_cohort(ranscore, ov, wv, niv, 'k' ) 
    #for i in range(len(lambds)): 
        #exp.AUC_cpit_cost_curve_deciles_cohort(rlearnerscores[i], ov, wv, niv, colors[i] ) 
    exp.AUC_cpit_cost_curve_deciles_cohort(hsclsvalscores, ov, wv, niv, 'r' ) 
    #exp.AUC_cpit_cost_curve_deciles_cohort(topprnkscore, ov, wv, niv, 'm' ) 
    exp.AUC_cpit_cost_curve_deciles_cohort(cpitrnkscore, ov, wv, niv, 'b' ) 
    plt.title('Rider Incentives Cost Curves using targeting models') 
    
    leg_str = ['random'] 
    #for i in range(len(lambds)): 
        #leg_str.append('hetero-treatment-ridge-reg (lamb=' +str(lambds[i])+ ',lang-rlearner)') 
    leg_str.append('HSCLS') 
    #leg_str.append('TPR') 
    leg_str.append('DRM') 
    plt.legend(leg_str) 
    plt.show() 
