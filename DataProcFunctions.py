import numpy as np, pandas as pd, pickle as pkl #tensorflow as tf, 

### function to compute the incrementality statics 
### it is commonly used for verification of datasets 

def LoadDataFromPklMatchingD2Dist(filename, frac = 1.0, use_python3 = False, save_cf_data=False, with_intensity=False): 
    ### loades data from a pre-defined pkl from filename 
    ### the pkl should contain nX_tr (training features); w_tr (training treatment labels); negcost_tr (training -ve cost outcomes); values_tr (training gain outcomes) 
    ### the same variable names go for _va for validation and _te for testing 
    ### this code then prepares them into numpy nd-arrays for consumption of Tensorflow, SKlearn, Causal Forests etc 
    ### also split them into control and treatment cohorts to make more convenient for Tensorflow consumption 
    
    if use_python3: 
        Dd = pkl.load(open(filename, 'rb'), encoding="latin1") 
    else: 
        Dd = pkl.load(open(filename, 'r')) 
    
    print('load data from ' + filename) 
    
    if 'LC' in filename: 
        neg_cost_str = 'n9d_ni_usd'
    else: 
        neg_cost_str = 'negcost' 
    
    Da = Dd['nX_a_tr']; Db = Dd['nX_b_tr']; w = Dd['w_tr']; ni = Dd[neg_cost_str + '_tr']; o = Dd['values_tr']; c = -1.0 * ni 
    Dva = Dd['nX_a_va']; Dvb = Dd['nX_b_va']; wv = Dd['w_va']; niv = Dd[neg_cost_str + '_va']; ov = Dd['values_va']; cv = -1.0 * niv 
    Dta = Dd['nX_a_te']; Dtb = Dd['nX_b_te']; wt = Dd['w_te']; nit = Dd[neg_cost_str + '_te']; ot = Dd['values_te']; ct = -1.0 * nit 
    D = np.concatenate((Da, Db), axis = 1) 
    Dv = np.concatenate((Dva, Dvb), axis = 1) 
    Dt = np.concatenate((Dta, Dtb), axis = 1) 
    
    d2d = Dd['d2dist_tr'] 
    d2dv = Dd['d2dist_va'] 
    d2dt = Dd['d2dist_te'] 
    
    if with_intensity == True: 
        int = Dd['treatintensity_tr'] 
        if int.shape[-1] == 1:
            int = np.reshape(int, (len(int), )) 
        intv = Dd['treatintensity_va'] 
        if intv.shape[-1] == 1:
            intv = np.reshape(intv, (len(intv), )) 
        intt = Dd['treatintensity_te'] 
        if intt.shape[-1] == 1:
            intt = np.reshape(intt, (len(intt), )) 
    
    if type(c) != type(o): 
        c = c.as_matrix()
    if type(cv) != type(ov):
        cv = cv.as_matrix() 
    if type(ct) != type(ot):
        ct = ct.as_matrix() 
    
    if type(ni) != type(o): 
        ni = ni.as_matrix()
    if type(cv) != type(ov):
        niv = niv.as_matrix() 
    if type(ct) != type(ot):
        nit = nit.as_matrix() 
    
    if w.shape[-1] == 1: 
        w = np.reshape(w, (len(w), )) 
    if wv.shape[-1] == 1: 
        wv = np.reshape(wv, (len(wv), )) 
    if wt.shape[-1] == 1: 
        wt = np.reshape(wt, (len(wt), )) 
    if o.shape[-1] == 1: 
        o = np.reshape(o, (len(o), )) 
    if c.shape[-1] == 1: 
        c = np.reshape(c, (len(c), ))  
    if ov.shape[-1] == 1: 
        ov = np.reshape(ov, (len(ov), )) 
    if cv.shape[-1] == 1: 
        cv = np.reshape(cv, (len(cv), )) 
    if ot.shape[-1] == 1: 
        ot = np.reshape(ot, (len(ot), )) 
    if ct.shape[-1] == 1: 
        ct = np.reshape(ct, (len(ct), )) 
    
    filter = ~ np.isnan(c) 
    Da = Da[np.where(filter==True)[0], :] 
    Db = Db[np.where(filter==True)[0], :] 
    w = w[np.where(filter==True)[0]] 
    c = c[np.where(filter==True)[0]] 
    o = o[np.where(filter==True)[0]] 
    if with_intensity == True: 
        int = int[np.where(filter==True)[0]] 
    
    filter = ~ np.isnan(cv) 
    Dva = Dva[np.where(filter==True)[0], :] 
    Dva = Dva[np.where(filter==True)[0], :] 
    wv = wv[np.where(filter==True)[0]] 
    cv = cv[np.where(filter==True)[0]] 
    ov = ov[np.where(filter==True)[0]] 
    if with_intensity == True: 
        intv = intv[np.where(filter==True)[0]] 
    
    filter = ~ np.isnan(ct) 
    Dta = Dta[np.where(filter==True)[0], :] 
    Dtb = Dtb[np.where(filter==True)[0], :] 
    wt = wt[np.where(filter==True)[0]] 
    ct = ct[np.where(filter==True)[0]] 
    ot = ot[np.where(filter==True)[0]] 
    if with_intensity == True: 
        intt = intt[np.where(filter==True)[0]] 
    
    ### sub-sampling for smaller data set 
    if frac < 1.0 and frac > 0.0: 
        tr_idx = np.random.permutation(len(D)) 
        cut = int(np.ceil(len(D) * frac))
        Da = Da[tr_idx[0:cut], :] 
        Db = Db[tr_idx[0:cut], :] 
        w = w[tr_idx[0:cut]] 
        c = c[tr_idx[0:cut]] 
        o = o[tr_idx[0:cut]] 
        d2d = d2d[tr_idx[0:cut]] 
        if with_intensity == True: 
            int = int[tr_idx[0:cut]] 
        
        va_idx = np.random.permutation(len(Dv)) 
        cut = int(np.ceil(len(Dv) * frac))
        Dva = Dva[va_idx[0:cut], :] 
        Dvb = Dvb[va_idx[0:cut], :] 
        wv = wv[va_idx[0:cut]] 
        cv = cv[va_idx[0:cut]] 
        ov = ov[va_idx[0:cut]] 
        d2dv = d2dv[tv_idx[0:cut]] 
        if with_intensity == True: 
            intv = intv[va_idx[0:cut]] 
        
        te_idx = np.random.permutation(len(Dt)) 
        cut = int(np.ceil(len(Dt) * frac)) 
        Dta = Dta[te_idx[0:cut], :] 
        Dtb = Dtb[te_idx[0:cut], :] 
        wt = wt[te_idx[0:cut]] 
        ct = ct[te_idx[0:cut]] 
        ot = ot[te_idx[0:cut]] 
        d2dt = d2dt[te_idx[0:cut]] 
        if with_intensity == True: 
            intt = intt[ve_idx[0:cut]] 
    
    elif frac <= 0.0 or frac > 1.0: 
        print('error: sample fraction is out of range!') 
        exit()
    
    Da_tre = Da[w > 0.5, :] 
    Da_unt = Da[w < 0.5, :] 
    
    Dva_tre = Dva[wv > 0.5, :] 
    Dva_unt = Dva[wv < 0.5, :] 
    
    Dta_tre = Dta[wt > 0.5, :] 
    Dta_unt = Dta[wt < 0.5, :] 
    
    Db_tre = Db[w > 0.5, :] 
    Db_unt = Db[w < 0.5, :] 
    
    Dvb_tre = Dvb[wv > 0.5, :] 
    Dvb_unt = Dvb[wv < 0.5, :] 
    
    Dtb_tre = Dtb[wt > 0.5, :] 
    Dtb_unt = Dtb[wt < 0.5, :] 
    
    o_tre = o[w > 0.5] 
    o_unt = o[w < 0.5] 
    
    ov_tre = ov[wv > 0.5] 
    ov_unt = ov[wv < 0.5] 
    
    ot_tre = ot[wt > 0.5] 
    ot_unt = ot[wt < 0.5] 
    
    c_tre = c[w > 0.5] 
    c_unt = c[w < 0.5] 
    
    cv_tre = cv[wv > 0.5] 
    cv_unt = cv[wv < 0.5] 
    
    ct_tre = ct[wt > 0.5] 
    ct_unt = ct[wt < 0.5] 
    
    d2d_tre = d2d[w > 0.5] 
    d2d_unt = d2d[w < 0.5] 
    d2dv_tre = d2dv[wv > 0.5] 
    d2dv_unt = d2dv[wv < 0.5] 
    d2dt_tre = d2dt[wt > 0.5] 
    d2dt_unt = d2dt[wt < 0.5] 
    
    if with_intensity == True: 
        int_tre = int[w > 0.5] 
        int_unt = int[w < 0.5] 
        intv_tre = intv[wv > 0.5] 
        intv_unt = intv[wv < 0.5] 
        intt_tre = intt[wt > 0.5] 
        intt_unt = intt[wt < 0.5] 
    
    print('printing averages of c_tr, c_unt, o_tre, o_unt ... :') 
    print(np.average(c_tre)); print(np.average(c_unt)); print(np.average(o_tre)); print(np.average(o_unt)) 
    
    prefix = filename.split('/')[2].split('.')[0]
    if save_cf_data: 
        SaveCFDataD2Dist(D, w, o, c, d2d, Dv, wv, ov, cv, d2dv, Dt, wt, ot, ct, d2dt, prefix) 
    
    return Da_tre, Da_unt, Db_tre, Db_unt, Dva_tre, Dva_unt, Dvb_tre, Dvb_unt, Dta_tre, Dta_unt, Dtb_tre, Dtb_unt, o_tre, o_unt, ov_tre, ov_unt, ot_tre, ot_unt, c_tre, c_unt, cv_tre, cv_unt, ct_tre, ct_unt, Da, Db, w, o, c, Dva, Dvb, wv, ov, cv, Dta, Dtb, wt, ot, ct, int_tre, int_unt, intv_tre, intv_unt, intt_tre, intt_unt, int, intv, intt, d2d_tre, d2d_unt, d2dv_tre, d2dv_unt, d2dt_tre, d2dt_unt, d2d, d2dv, d2dt 

def LoadDataFromPklMatching(filename, frac = 1.0, use_python3 = False, save_cf_data=False, with_intensity=False): 
    ### loades data from a pre-defined pkl from filename 
    ### the pkl should contain nX_tr (training features); w_tr (training treatment labels); negcost_tr (training -ve cost outcomes); values_tr (training gain outcomes) 
    ### the same variable names go for _va for validation and _te for testing 
    ### this code then prepares them into numpy nd-arrays for consumption of Tensorflow, SKlearn, Causal Forests etc 
    ### also split them into control and treatment cohorts to make more convenient for Tensorflow consumption 
    
    if use_python3: 
        Dd = pkl.load(open(filename, 'rb'), encoding="latin1") 
    else: 
        Dd = pkl.load(open(filename, 'r')) 
    
    print('load data from ' + filename) 
    
    if 'LC' in filename: 
        neg_cost_str = 'n9d_ni_usd'
    else: 
        neg_cost_str = 'negcost' 
    
    Da = Dd['nX_a_tr']; Db = Dd['nX_b_tr']; w = Dd['w_tr']; ni = Dd[neg_cost_str + '_tr']; o = Dd['values_tr']; c = -1.0 * ni 
    Dva = Dd['nX_a_va']; Dvb = Dd['nX_b_va']; wv = Dd['w_va']; niv = Dd[neg_cost_str + '_va']; ov = Dd['values_va']; cv = -1.0 * niv 
    Dta = Dd['nX_a_te']; Dtb = Dd['nX_b_te']; wt = Dd['w_te']; nit = Dd[neg_cost_str + '_te']; ot = Dd['values_te']; ct = -1.0 * nit 
    
    if with_intensity == True: 
        int = Dd['treatintensity_tr'] 
        if int.shape[-1] == 1:
            int = np.reshape(int, (len(int), )) 
        intv = Dd['treatintensity_va'] 
        if intv.shape[-1] == 1:
            intv = np.reshape(intv, (len(intv), )) 
        intt = Dd['treatintensity_te'] 
        if intt.shape[-1] == 1:
            intt = np.reshape(intt, (len(intt), )) 
    
    if type(c) != type(o): 
        c = c.as_matrix()
    if type(cv) != type(ov):
        cv = cv.as_matrix() 
    if type(ct) != type(ot):
        ct = ct.as_matrix() 
    
    if type(ni) != type(o): 
        ni = ni.as_matrix()
    if type(cv) != type(ov):
        niv = niv.as_matrix() 
    if type(ct) != type(ot):
        nit = nit.as_matrix() 
    
    if w.shape[-1] == 1: 
        w = np.reshape(w, (len(w), )) 
    if wv.shape[-1] == 1: 
        wv = np.reshape(wv, (len(wv), )) 
    if wt.shape[-1] == 1: 
        wt = np.reshape(wt, (len(wt), )) 
    if o.shape[-1] == 1: 
        o = np.reshape(o, (len(o), )) 
    if c.shape[-1] == 1: 
        c = np.reshape(c, (len(c), ))  
    if ov.shape[-1] == 1: 
        ov = np.reshape(ov, (len(ov), )) 
    if cv.shape[-1] == 1: 
        cv = np.reshape(cv, (len(cv), )) 
    if ot.shape[-1] == 1: 
        ot = np.reshape(ot, (len(ot), )) 
    if ct.shape[-1] == 1: 
        ct = np.reshape(ct, (len(ct), )) 
    
    filter = ~ np.isnan(c) 
    Da = Da[np.where(filter==True)[0], :] 
    Db = Db[np.where(filter==True)[0], :] 
    w = w[np.where(filter==True)[0]] 
    c = c[np.where(filter==True)[0]] 
    o = o[np.where(filter==True)[0]] 
    if with_intensity == True: 
        int = int[np.where(filter==True)[0]] 
    
    filter = ~ np.isnan(cv) 
    Dva = Dva[np.where(filter==True)[0], :] 
    Dva = Dva[np.where(filter==True)[0], :] 
    wv = wv[np.where(filter==True)[0]] 
    cv = cv[np.where(filter==True)[0]] 
    ov = ov[np.where(filter==True)[0]] 
    if with_intensity == True: 
        intv = intv[np.where(filter==True)[0]] 
    
    filter = ~ np.isnan(ct) 
    Dta = Dta[np.where(filter==True)[0], :] 
    Dtb = Dtb[np.where(filter==True)[0], :] 
    wt = wt[np.where(filter==True)[0]] 
    ct = ct[np.where(filter==True)[0]] 
    ot = ot[np.where(filter==True)[0]] 
    if with_intensity == True: 
        intt = intt[np.where(filter==True)[0]] 
    
    ### sub-sampling for smaller data set 
    if frac < 1.0 and frac > 0.0: 
        tr_idx = np.random.permutation(len(D)) 
        cut = int(np.ceil(len(D) * frac))
        Da = Da[tr_idx[0:cut], :] 
        Db = Db[tr_idx[0:cut], :] 
        w = w[tr_idx[0:cut]] 
        c = c[tr_idx[0:cut]] 
        o = o[tr_idx[0:cut]] 
        if with_intensity == True: 
            int = int[tr_idx[0:cut]] 
        
        va_idx = np.random.permutation(len(Dv)) 
        cut = int(np.ceil(len(Dv) * frac))
        Dva = Dva[va_idx[0:cut], :] 
        Dvb = Dvb[va_idx[0:cut], :] 
        wv = wv[va_idx[0:cut]] 
        cv = cv[va_idx[0:cut]] 
        ov = ov[va_idx[0:cut]] 
        if with_intensity == True: 
            intv = intv[va_idx[0:cut]] 
        
        te_idx = np.random.permutation(len(Dt)) 
        cut = int(np.ceil(len(Dt) * frac)) 
        Dta = Dta[te_idx[0:cut], :] 
        Dtb = Dtb[te_idx[0:cut], :] 
        wt = wt[te_idx[0:cut]] 
        ct = ct[te_idx[0:cut]] 
        ot = ot[te_idx[0:cut]] 
        if with_intensity == True: 
            intt = intt[ve_idx[0:cut]] 
        
    elif frac <= 0.0 or frac > 1.0: 
        print('error: sample fraction is out of range!') 
        exit()
    
    Da_tre = Da[w > 0.5, :] 
    Da_unt = Da[w < 0.5, :] 
    
    Dva_tre = Dva[wv > 0.5, :] 
    Dva_unt = Dva[wv < 0.5, :] 
    
    Dta_tre = Dta[wt > 0.5, :] 
    Dta_unt = Dta[wt < 0.5, :] 
    
    Db_tre = Db[w > 0.5, :] 
    Db_unt = Db[w < 0.5, :] 
    
    Dvb_tre = Dvb[wv > 0.5, :] 
    Dvb_unt = Dvb[wv < 0.5, :] 
    
    Dtb_tre = Dtb[wt > 0.5, :] 
    Dtb_unt = Dtb[wt < 0.5, :] 
    
    o_tre = o[w > 0.5] 
    o_unt = o[w < 0.5] 
    
    ov_tre = ov[wv > 0.5] 
    ov_unt = ov[wv < 0.5] 
    
    ot_tre = ot[wt > 0.5] 
    ot_unt = ot[wt < 0.5] 
    
    c_tre = c[w > 0.5] 
    c_unt = c[w < 0.5] 
    
    cv_tre = cv[wv > 0.5] 
    cv_unt = cv[wv < 0.5] 
    
    ct_tre = ct[wt > 0.5] 
    ct_unt = ct[wt < 0.5] 
    
    if with_intensity == True: 
        int_tre = int[w > 0.5] 
        int_unt = int[w < 0.5] 
        intv_tre = intv[wv > 0.5] 
        intv_unt = intv[wv < 0.5] 
        intt_tre = intt[wt > 0.5] 
        intt_unt = intt[wt < 0.5] 
    
    print('printing averages of c_tr, c_unt, o_tre, o_unt ... :') 
    print(np.average(c_tre)); print(np.average(c_unt)); print(np.average(o_tre)); print(np.average(o_unt)) 
    
    prefix = filename.split('/')[1].split('.')[0]
    if save_cf_data: 
        SaveCFData(D, w, o, c, Dv, wv, ov, cv, Dt, wt, ot, ct, prefix) 
    
    return Da_tre, Da_unt, Db_tre, Db_unt, Dva_tre, Dva_unt, Dvb_tre, Dvb_unt, Dta_tre, Dta_unt, Dtb_tre, Dtb_unt, o_tre, o_unt, ov_tre, ov_unt, ot_tre, ot_unt, c_tre, c_unt, cv_tre, cv_unt, ct_tre, ct_unt, Da, Db, w, o, c, Dva, Dvb, wv, ov, cv, Dta, Dtb, wt, ot, ct, int_tre, int_unt, intv_tre, intv_unt, intt_tre, intt_unt, int, intv, intt 

def LoadDataFromPkl(filename, frac = 1.0, use_python3 = False, save_cf_data=False, with_intensity=False): 
    ### loades data from a pre-defined pkl from filename 
    ### the pkl should contain nX_tr (training features); w_tr (training treatment labels); negcost_tr (training -ve cost outcomes); values_tr (training gain outcomes) 
    ### the same variable names go for _va for validation and _te for testing 
    ### this code then prepares them into numpy nd-arrays for consumption of Tensorflow, SKlearn, Causal Forests etc 
    ### also split them into control and treatment cohorts to make more convenient for Tensorflow consumption 
    
    if use_python3: 
        Dd = pkl.load(open(filename, 'rb'), encoding="latin1") 
    else: 
        Dd = pkl.load(open(filename, 'r')) 
    
    print('load data from ' + filename) 
    
    if 'LC' in filename: 
        neg_cost_str = 'n9d_ni_usd'
    else: 
        neg_cost_str = 'negcost' 
    
    D = Dd['nX_tr']; w = Dd['w_tr']; ni = Dd[neg_cost_str + '_tr']; o = Dd['values_tr']; c = -1.0 * ni 
    Dv = Dd['nX_va']; wv = Dd['w_va']; niv = Dd[neg_cost_str + '_va']; ov = Dd['values_va']; cv = -1.0 * niv 
    Dt = Dd['nX_te']; wt = Dd['w_te']; nit = Dd[neg_cost_str + '_te']; ot = Dd['values_te']; ct = -1.0 * nit 
    
    if with_intensity == True: 
        int = Dd['treatintensity_tr'] 
        if int.shape[-1] == 1:
            int = np.reshape(int, (len(int), )) 
        intv = Dd['treatintensity_va'] 
        if intv.shape[-1] == 1:
            intv = np.reshape(intv, (len(intv), )) 
        intt = Dd['treatintensity_te'] 
        if intt.shape[-1] == 1:
            intt = np.reshape(intt, (len(intt), )) 
    
    if type(c) != type(o): 
        c = c.as_matrix()
    if type(cv) != type(ov):
        cv = cv.as_matrix() 
    if type(ct) != type(ot):
        ct = ct.as_matrix() 
    
    if type(ni) != type(o): 
        ni = ni.as_matrix()
    if type(cv) != type(ov):
        niv = niv.as_matrix() 
    if type(ct) != type(ot):
        nit = nit.as_matrix() 
    
    if w.shape[-1] == 1: 
        w = np.reshape(w, (len(w), )) 
    if wv.shape[-1] == 1: 
        wv = np.reshape(wv, (len(wv), )) 
    if wt.shape[-1] == 1: 
        wt = np.reshape(wt, (len(wt), )) 
    if o.shape[-1] == 1: 
        o = np.reshape(o, (len(o), )) 
    if c.shape[-1] == 1: 
        c = np.reshape(c, (len(c), ))  
    if ov.shape[-1] == 1: 
        ov = np.reshape(ov, (len(ov), )) 
    if cv.shape[-1] == 1: 
        cv = np.reshape(cv, (len(cv), )) 
    if ot.shape[-1] == 1: 
        ot = np.reshape(ot, (len(ot), )) 
    if ct.shape[-1] == 1: 
        ct = np.reshape(ct, (len(ct), )) 
    
    filter = ~ np.isnan(c) 
    D = D[np.where(filter==True)[0], :] 
    w = w[np.where(filter==True)[0]] 
    c = c[np.where(filter==True)[0]] 
    o = o[np.where(filter==True)[0]] 
    if with_intensity == True: 
        int = int[np.where(filter==True)[0]] 
    
    filter = ~ np.isnan(cv) 
    Dv = Dv[np.where(filter==True)[0], :] 
    wv = wv[np.where(filter==True)[0]] 
    cv = cv[np.where(filter==True)[0]] 
    ov = ov[np.where(filter==True)[0]] 
    if with_intensity == True: 
        intv = intv[np.where(filter==True)[0]] 
    
    filter = ~ np.isnan(ct) 
    Dt = Dt[np.where(filter==True)[0], :] 
    wt = wt[np.where(filter==True)[0]] 
    ct = ct[np.where(filter==True)[0]] 
    ot = ot[np.where(filter==True)[0]] 
    if with_intensity == True: 
        intt = intt[np.where(filter==True)[0]] 
    
    ### sub-sampling for smaller data set 
    if frac < 1.0 and frac > 0.0: 
        tr_idx = np.random.permutation(len(D)) 
        cut = int(np.ceil(len(D) * frac))
        D = D[tr_idx[0:cut], :] 
        w = w[tr_idx[0:cut]] 
        c = c[tr_idx[0:cut]] 
        o = o[tr_idx[0:cut]] 
        if with_intensity == True: 
            int = int[tr_idx[0:cut]] 
        
        va_idx = np.random.permutation(len(Dv)) 
        cut = int(np.ceil(len(Dv) * frac))
        Dv = Dv[va_idx[0:cut], :] 
        wv = wv[va_idx[0:cut]] 
        cv = cv[va_idx[0:cut]] 
        ov = ov[va_idx[0:cut]] 
        if with_intensity == True: 
            intv = intv[va_idx[0:cut]] 
        
        te_idx = np.random.permutation(len(Dt)) 
        cut = int(np.ceil(len(Dt) * frac)) 
        Dt = Dt[te_idx[0:cut], :] 
        wt = wt[te_idx[0:cut]] 
        ct = ct[te_idx[0:cut]] 
        ot = ot[te_idx[0:cut]] 
        if with_intensity == True: 
            intt = intt[ve_idx[0:cut]] 
        
    elif frac <= 0.0 or frac > 1.0: 
        print('error: sample fraction is out of range!') 
        exit()
    
    D_tre = D[w > 0.5, :] 
    D_unt = D[w < 0.5, :] 
        
    Dv_tre = Dv[wv > 0.5, :] 
    Dv_unt = Dv[wv < 0.5, :] 
        
    Dt_tre = Dt[wt > 0.5, :] 
    Dt_unt = Dt[wt < 0.5, :] 
    
    o_tre = o[w > 0.5] 
    o_unt = o[w < 0.5] 
    
    ov_tre = ov[wv > 0.5] 
    ov_unt = ov[wv < 0.5] 
    
    ot_tre = ot[wt > 0.5] 
    ot_unt = ot[wt < 0.5] 
    
    c_tre = c[w > 0.5] 
    c_unt = c[w < 0.5] 
    
    cv_tre = cv[wv > 0.5] 
    cv_unt = cv[wv < 0.5] 
    
    ct_tre = ct[wt > 0.5] 
    ct_unt = ct[wt < 0.5] 
    
    if with_intensity == True: 
        int_tre = int[w > 0.5] 
        int_unt = int[w < 0.5] 
        intv_tre = intv[wv > 0.5] 
        intv_unt = intv[wv < 0.5] 
        intt_tre = intt[wt > 0.5] 
        intt_unt = intt[wt < 0.5] 
    
    print('printing averages of c_tr, c_unt, o_tre, o_unt ... :') 
    print(np.average(c_tre)); print(np.average(c_unt)); print(np.average(o_tre)); print(np.average(o_unt)) 
    
    prefix = filename.split('/')[2].split('.')[0]
    if save_cf_data: 
        SaveCFData(D, w, o, c, Dv, wv, ov, cv, Dt, wt, ot, ct, prefix) 
    if with_intensity: 
        return D_tre, D_unt, Dv_tre, Dv_unt, Dt_tre, Dt_unt, o_tre, o_unt, ov_tre, ov_unt, ot_tre, ot_unt, c_tre, c_unt, cv_tre, cv_unt, ct_tre, ct_unt, D, w, o, c, Dv, wv, ov, cv, Dt, wt, ot, ct , int_tre, int_unt, intv_tre, intv_unt, intt_tre, intt_unt, int, intv, intt  
    else: 
        return D_tre, D_unt, Dv_tre, Dv_unt, Dt_tre, Dt_unt, o_tre, o_unt, ov_tre, ov_unt, ot_tre, ot_unt, c_tre, c_unt, cv_tre, cv_unt, ct_tre, ct_unt, D, w, o, c, Dv, wv, ov, cv, Dt, wt, ot, ct #, int_tre, int_unt, intv_tre, intv_unt, intt_tre, intt_unt, int, intv, intt  

def SaveCFData(D, w, o, c, Dv, wv, ov, cv, Dt, wt, ot, ct, prefix): 
    ### this function is used to save csv data from our framework 
    ### the saved data is intended to be consumped by the R implementation 
    ### of Causal Forest with the R grf (generalized random forest package) 
    
    def _temp_rdf(X, y=None):
        """
        Create temporary pandas data frame for R model training and predict
        :param X:
        :param y:
        :return: temp data frame in R
        """
        mat = X.astype(float)
        # insert y to the last column of the R dataframe
        if y is not None:
            mat = np.concatenate([X, y.reshape([-1, 1])], axis=1).astype(float)
        df = pd.DataFrame(mat, columns=['V{}'.format(i+1) for i in range(mat.shape[1])]) 
        return df 
    
    print('D number of features: '  + str(D.shape[1]))
    print('saving data for causal forests ...')
    
    rdf = _temp_rdf(D, o) 
    rdf.insert(D.shape[1] + 1, 'cost', c) 
    rdf.insert(D.shape[1] + 2, 'treatment', w) 
    
    rdf.to_csv('../data/causal_forest_r_data_' + prefix + '_train.csv') 
    
    rdf = _temp_rdf(Dv, ov) 
    rdf.insert(Dv.shape[1] + 1, 'cost', cv) 
    rdf.insert(Dv.shape[1] + 2, 'treatment', wv) 
    
    rdf.to_csv('../data/causal_forest_r_data_' + prefix + '_valid.csv') 
    
    rdf = _temp_rdf(Dt, ot) 
    rdf.insert(Dt.shape[1] + 1, 'cost', ct) 
    rdf.insert(Dt.shape[1] + 2, 'treatment', wt) 
    
    rdf.to_csv('../data/causal_forest_r_data_' + prefix + '_test.csv') 
    
    print('done') 

def SaveCFDataD2Dist(D, w, o, c, d2d, Dv, wv, ov, cv, d2dv, Dt, wt, ot, ct, d2dt, prefix): 
    ### this function is used to save csv data from our framework 
    ### the saved data is intended to be consumped by the R implementation 
    ### of Causal Forest with the R grf (generalized random forest package) 
    
    def _temp_rdf(X, y=None):
        """
        Create temporary pandas data frame for R model training and predict
        :param X:
        :param y:
        :return: temp data frame in R
        """
        mat = X.astype(float)
        # insert y to the last column of the R dataframe
        if y is not None:
            mat = np.concatenate([X, y.reshape([-1, 1])], axis=1).astype(float)
        df = pd.DataFrame(mat, columns=['V{}'.format(i+1) for i in range(mat.shape[1])]) 
        return df 
    
    print('D number of features: '  + str(D.shape[1]))
    print('saving data for causal forests ...')
    
    rdf = _temp_rdf(D, o) 
    rdf.insert(D.shape[1] + 1, 'cost', c) 
    rdf.insert(D.shape[1] + 2, 'treatment', w) 
    rdf.insert(D.shape[1] + 3, 'd2d', d2d) 
    
    rdf.to_csv('../data/causal_forest_r_data_' + prefix + '_train.csv') 
    
    rdf = _temp_rdf(Dv, ov) 
    rdf.insert(Dv.shape[1] + 1, 'cost', cv) 
    rdf.insert(Dv.shape[1] + 2, 'treatment', wv) 
    rdf.insert(Dv.shape[1] + 3, 'd2d', d2dv) 
    
    rdf.to_csv('../data/causal_forest_r_data_' + prefix + '_valid.csv') 
    
    rdf = _temp_rdf(Dt, ot) 
    rdf.insert(Dt.shape[1] + 1, 'cost', ct) 
    rdf.insert(Dt.shape[1] + 2, 'treatment', wt) 
    rdf.insert(Dt.shape[1] + 3, 'd2d', d2dt) 
    
    rdf.to_csv('../data/causal_forest_r_data_' + prefix + '_test.csv') 

def ComputePrintIncrementalityStats( 
        trainDataFeatures, 
        trainDataTreatmentLabels, 
        trainDataTripsLabels, 
        trainDataCostLabels, 
        trainDataGBLabels, 
        trainDataExistLabels 
    ): 
    
    ### compute and print the status of the incrementality/incentives dataset 
    not_treated_data_exist = np.multiply(~(trainDataTreatmentLabels > 0.5), trainDataExistLabels) 
    num_treated_instances = sum(sum(trainDataTreatmentLabels)) 
    num_untreated_instances = sum(sum(not_treated_data_exist)) 
    
    treated_rpu = sum(sum(np.multiply(trainDataTripsLabels, trainDataTreatmentLabels))) / num_treated_instances 
    untreated_rpu = sum(sum(np.multiply(trainDataTripsLabels, not_treated_data_exist))) / num_untreated_instances 
    
    treated_cpu = sum(sum(np.multiply(trainDataCostLabels, trainDataTreatmentLabels))) / num_treated_instances 
    untreated_cpu = sum(sum(np.multiply(trainDataCostLabels, not_treated_data_exist))) / num_untreated_instances 
    
    treated_gpu = sum(sum(np.multiply(trainDataGBLabels, trainDataTreatmentLabels))) / num_treated_instances 
    untreated_gpu = sum(sum(np.multiply(trainDataGBLabels, not_treated_data_exist))) / num_untreated_instances 
    
    print('----> verifying incentives/incrementality data statistics: --------') 
    print('treated_rpu: ' + str(treated_rpu))
    print('untreated_rpu: ' + str(untreated_rpu))
    print('treated_cpu: ' + str(treated_cpu))
    print('untreated_cpu: ' + str(untreated_cpu))
    print('treated_gpu: ' + str(treated_gpu))
    print('untreated_gpu: ' + str(untreated_gpu))
    print('num_treated_instances: ' + str(num_treated_instances))
    print('num_untreated_instances: ' + str(num_untreated_instances))
    
    cpit = (treated_cpu - untreated_cpu) / (treated_rpu - untreated_rpu) 
    cpigb = (treated_cpu - untreated_cpu) / (treated_gpu - untreated_gpu) 
    print('training data cpit: ' + str(cpit))
    print('training data cpigb: ' + str(cpigb))
    print(' ') 

### this function was adhocly created to save the benchmarking data 
### to compare sequential learning vs DRM 
def SaveBenchmarkDRMData(
        processed_data_filename, 
        seq_start, 
        trainDataFeatures, 
        trainDataTreatmentLabels, 
        trainDataTripsLabels, 
        trainDataCostLabels, 
        trainDataGBLabels, 
        trainDataExistLabels, 
        valDataFeatures, 
        valDataTreatmentLabels, 
        valDataTripsLabels, 
        valDataCostLabels, 
        valDataGBLabels, 
        valDataExistLabels 
    ): 
    
    ### extract all data to use it for DRM benchmarking 
    trainDataExistLabels_DRM = np.reshape(trainDataExistLabels, (-1,)) 
    
    trainDataFeatures_DRM = np.reshape(trainDataFeatures, (-1, trainDataFeatures.shape[-1])) 
    trainDataFeatures_DRM = trainDataFeatures_DRM[trainDataExistLabels_DRM > 0.5, :] 
    trainDataTreatmentLabels_DRM = np.reshape(trainDataTreatmentLabels, (-1,)) 
    trainDataTreatmentLabels_DRM = trainDataTreatmentLabels_DRM[trainDataExistLabels_DRM > 0.5] 
    trainDataTripsLabels_DRM = np.reshape(trainDataTripsLabels, (-1,)) 
    trainDataTripsLabels_DRM = trainDataTripsLabels_DRM[trainDataExistLabels_DRM > 0.5] 
    trainDataCostLabels_DRM = np.reshape(trainDataCostLabels, (-1,)) 
    trainDataCostLabels_DRM = trainDataCostLabels_DRM[trainDataExistLabels_DRM > 0.5] 
    
    valDataExistLabels_DRM = np.reshape(valDataExistLabels, (-1,)) 
    valDataFeatures_DRM = np.reshape(valDataFeatures, (-1, valDataFeatures.shape[-1])) 
    valDataFeatures_DRM = valDataFeatures_DRM[valDataExistLabels_DRM > 0.5, :]
    valDataTreatmentLabels_DRM = np.reshape(valDataTreatmentLabels, (-1,)) 
    valDataTreatmentLabels_DRM = valDataTreatmentLabels_DRM[valDataExistLabels_DRM > 0.5]
    valDataTripsLabels_DRM = np.reshape(valDataTripsLabels, (-1,)) 
    valDataTripsLabels_DRM = valDataTripsLabels_DRM[valDataExistLabels_DRM > 0.5]
    valDataCostLabels_DRM = np.reshape(valDataCostLabels, (-1,)) 
    valDataCostLabels_DRM = valDataCostLabels_DRM [valDataExistLabels_DRM > 0.5]
    
    print('----> saving data for Benchmark DRM ------------') 
    saveD = dict() 
    saveD['nX_tr'] = trainDataFeatures_DRM 
    saveD['w_tr'] = trainDataTreatmentLabels_DRM 
    saveD['values_tr'] = trainDataTripsLabels_DRM 
    saveD['n9d_ni_usd_tr'] = trainDataCostLabels_DRM * -1.0 
    saveD['nX_va'] = valDataFeatures_DRM 
    saveD['w_va'] = valDataTreatmentLabels_DRM 
    saveD['values_va'] = valDataTripsLabels_DRM 
    saveD['n9d_ni_usd_va'] = valDataCostLabels_DRM * -1.0 
    
    pkl.dump(saveD, open('data/drm_train_data_from_hscls' + processed_data_filename[-40:-15] + '__seq_start' + str(seq_start) + '.pkl', 'w')) 
    print('done') 
    print(' ') 

def SeqDrmProcessData(processed_data_filename, 
                   num_short = None, 
                   num_train = 200000, 
                   seq_start = 0 
                   ): 
    ### Process data for hscls and drm models 
    ### and separated elements of validation data 
    ### Inputs: 
    ### processed_data_filename, the filename for saved pickle data, the file is output from dataprep script prep_hscls_data.py 
    ###   processed_data_filename loads datastore and seqscore 
    
    ###   datastore, 3D tensor of shape [dataset_size, num_weeks, feature_dims] including the treatment labels 
    ###           num_weeks = 19/11; feature_dims = 28 
    ###           - 0-17 [18 dimensions] user features 
    ###           - 18-22 [5 dimensions] this week user behavior labels 
    ###             ['is_treatment', 'num_trips', 'gross_bookings_usd', 'variable_contribution_usd', 'data_exists'] 
    ###           - 23-27 [5 dimensions] next week user behavior labels 
    ###   seqstore, 2D tensor of shape [dataset_size, 2], the 2 dimensions contain a start-of-sequence (0-indexed) and length-of-sequence 
    
    ### num_short: option to shorten data to num_short samples 
    ### num_train: number of training examples, rest left for validation 
    ### seq_start: the 0-indexed position of start-of-sequence 
    
    ### Returns: 
    ### BatchedDataset:O a batched training dataset containing all the sequential data and lengths 
    ### TrainDataFeatures/ValDataFeatures, 
    
    print('----> loading data from dataprep ------------') 
    D = pkl.load(open(processed_data_filename, 'r')) 
    datastore = D['datastore'] 
    seqlenstore = D['seqlenstore'] 
    ### after dataprep, seqlenstore here is just the length of seqence 
    if num_short == None: 
        num_short = datastore.shape[0] 
    datastore = datastore[0 : num_short, seq_start:, :] 
    print('done ')
    print(' ') 
    
    print('----> normalize the labels from past week for recurrent input ------------') 
    d1 = datastore[:, :, 0:18]  
    d2 = datastore[:, :, 18:23]
    d3 = datastore[:, :, 23:]
    
    d2 = np.reshape(d2, (-1, 5)) 
    d2 = d2 - np.mean(d2, axis = 0) 
    d2 = d2 / np.std(d2, axis = 0) 
    d2 = np.reshape(d2, (num_short, -1, 5)) 
    
    d1 = np.concatenate((d1, d2), axis = 2) 
    datastore = np.concatenate((d1, d3), axis = 2) 
    ### --- end normalize 
    print('done ')
    print(' ') 
    
    ## deal with seq_start option to calibrate seqlenstore 
    seqlenstore = seqlenstore[0 : num_short] - seq_start 
    seqlenstore = np.maximum(seqlenstore, 0) 
    
    num_val = num_short - num_train 
    trainDataAll = datastore[0 : num_train, :, :] 
    trainDataSeqLens = seqlenstore[0 : num_train]  
    ## seqlenstore is shaped (len, ), we reshape to (len, 1) for input 
    ## to dynamic_rnn 
    trainDataSeqLens = np.reshape(trainDataSeqLens, (-1, 1)) 
    
    valDataAll = datastore[num_train:, :, :]  
    valDataSeqLens = seqlenstore[num_train:] 
    
    ### separate the features from labels 
    trainDataFeatures = trainDataAll[:, :, 0:23] 
    trainDataTreatmentLabels = trainDataAll[:, :, 23] 
    trainDataTripsLabels = trainDataAll[:, :, 24] 
    trainDataGBLabels = trainDataAll[:, :, 25] 
    trainDataCostLabels = trainDataAll[:, :, 26] * -1.0 ### cost = -1.0 * net-inflow or variable contributions 
    trainDataExistLabels = trainDataAll[:, :, 27]  
    
    valDataFeatures = valDataAll[:, :, 0:23] 
    valDataTreatmentLabels = valDataAll[:, :, 23] 
    valDataTripsLabels = valDataAll[:, :, 24] 
    valDataGBLabels = valDataAll[:, :, 25] 
    valDataCostLabels = valDataAll[:, :, 26] * -1.0 ### cost = -1.0 * net-inflow or variable contributions 
    valDataExistLabels = valDataAll[:, :, 27] 
    
    ComputePrintIncrementalityStats( 
        trainDataFeatures, 
        trainDataTreatmentLabels, 
        trainDataTripsLabels, 
        trainDataCostLabels, 
        trainDataGBLabels, 
        trainDataExistLabels) 
    
    SaveBenchmarkDRMData(
        processed_data_filename, 
        seq_start, 
        trainDataFeatures, 
        trainDataTreatmentLabels, 
        trainDataTripsLabels, 
        trainDataCostLabels, 
        trainDataGBLabels, 
        trainDataExistLabels, 
        valDataFeatures, 
        valDataTreatmentLabels, 
        valDataTripsLabels, 
        valDataCostLabels, 
        valDataGBLabels, 
        valDataExistLabels 
    ) 
    
    batchedDataset = [trainDataAll, np.reshape(trainDataSeqLens, (-1, ))] 
    
    return (batchedDataset, 
            valDataFeatures, 
            valDataSeqLens,
            valDataTreatmentLabels, 
            valDataTripsLabels, 
            valDataCostLabels, 
            valDataGBLabels, 
            valDataExistLabels) 

## function to pull a small minibatch from the 
## entire dataset to enable stochastic training for dl 

def PullMinibatch(batchedDataset, batch_size): 
    
    ### pulls a random minibatch from the 
    ### full dataset tensor 
    
    trainDataAll = batchedDataset[0]
    trainDataSeqLensAll = batchedDataset[1]
    
    num_samples = trainDataAll.shape[0] 
    batch_idx = np.random.randint(0, num_samples - batch_size - 1) 
    databatch = trainDataAll[batch_idx : batch_idx + batch_size, :, :] 
    trainDataSeqLens = trainDataSeqLensAll[batch_idx : batch_idx + batch_size] 
    
    trainDataFeatures = databatch[:, :, 0:23] 
    trainDataTreatmentLabels = databatch[:, :, 23] 
    trainDataTripsLabels = databatch[:, :, 24] 
    trainDataCostLabels = databatch[:, :, 26] * -1.0 
    trainDataGBLabels = databatch[:, :, 25] 
    trainDataExistLabels = databatch[:, :, 27]  
    
    return (trainDataFeatures, 
            trainDataSeqLens, 
            trainDataTreatmentLabels, 
            trainDataTripsLabels, 
            trainDataCostLabels, 
            trainDataGBLabels, 
            trainDataExistLabels) 

### Appendix 
"""
def gather_nd_drm_data_from_sequences(trainDataFeatures, trainDataSeqLens, num_train, sess): 
    trainDataSeqLens1 = np.reshape(trainDataSeqLens, (num_train, 1)).astype(np.int) - 1 
    trainDataSeqLensIndices1 = np.reshape(np.asarray(np.arange(num_train)), (num_train, 1)) 
    trainDataSeqLens2 = tf.concat((trainDataSeqLensIndices1, trainDataSeqLens1), axis = 1) 
    trainDataFeatures_perm = tf.transpose(trainDataFeatures, perm = [0, 1, 2]) 
    trainDataFeatures_DRM_node = tf.gather_nd(trainDataFeatures_perm, trainDataSeqLens2) 
    trainDataFeatures_DRM = sess.run(trainDataFeatures_DRM_node) 
    
    return trainDataFeatures_DRM 

def gather_nd_drm_data_from_sequences_labels(trainDataLabels, trainDataSeqLens, num_train, sess):     
    trainDataSeqLens1 = np.reshape(trainDataSeqLens, (num_train, 1)).astype(np.int) - 1 
    trainDataSeqLensIndices1 = np.reshape(np.asarray(np.arange(num_train)), (num_train, 1)) 
    trainDataSeqLens2 = tf.concat((trainDataSeqLensIndices1, trainDataSeqLens1), axis = 1) 
    #trainDataFeatures_perm = tf.transpose(trainDataFeatures, perm = [0, 1, 2]) 
    trainDataLabels = np.reshape(trainDataLabels, (num_train, -1, 1))
    trainDataFeatures_DRM_node = tf.gather_nd(trainDataLabels, trainDataSeqLens2) 
    trainDataFeatures_DRM = sess.run(trainDataFeatures_DRM_node) 
    
    return trainDataFeatures_DRM 
"""
#### code-snippet to use Dataset in Tensorflow 
""" 
Dset = tf.data.Dataset.from_tensors(D) 
wset = tf.data.Dataset.from_tensors(w) 
cset = tf.data.Dataset.from_tensors(c) 
oset = tf.data.Dataset.from_tensors(o) 

combined_dataset = tf.data.Dataset.zip((Dset, wset, cset, oset)) 
batched_combined_dataset = combined_dataset.batch(256) 

iterator = batched_combined_dataset.make_one_shot_iterator() 
next_batch = iterator.get_next() 

for i in range(10): 
  value = sess.run(next_element) 
  assert i == value 
""" 
