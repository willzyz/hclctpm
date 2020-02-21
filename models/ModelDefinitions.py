import numpy as np, tensorflow as tf 

### module with model defitions 
### build tensorflow graphs 

def CTPMMatcherDPFeatureD2DistDNN(graph, Da_tre, Da_unt, Db_tre, Db_unt, o_tre, o_unt, c_tre, c_unt, i_tre, i_unt, d2d_tre, d2d_unt, d2dlamb, idstr, num_hidden, temp, p_quantile, dropout_rate): 
    
    ## implements the Direct Ranking Model based on CPIT 
    ## D_tre: user context features for treatment 
    ## D_unt: user context features for control (untreated) 
    ## o_tre: user next week order labels for treatment 
    ## o_unt: user next week order labels for control (untreated) 
    ## c_tre: user next week cost(negative net-inflow/variable contribution) labels for treatment 
    ## c_unt: user next week cost(negative net-inflow/variable contribution) labels for control (untreated) 
    ## i_tre: treatment intensity of treatment group 
    ## i_unt: treatment intensity of control group 
    ## idstr: a string to indicate whether function is used for train/val, avoid tensorflor parameter re-use 
    
    ## returns: 
    ## obj, objective node 
    ## opt, optimizer node 
    ## h_tre_rnkscore, ranker scores for treated users 
    ## h_unt_rnkscore, ranker scores for control users 
    
    D_tre = np.concatenate((Da_tre, Db_tre), axis = 1) 
    D_unt = np.concatenate((Da_unt, Db_unt), axis = 1) 
    
    ## define size of cohort datasets 
    size_tre = D_tre.shape[0] 
    size_unt = D_unt.shape[0] 
    
    with graph.as_default() as g: 
        ### ------ define model graph of direct ranking ------ 
        
        ### define ranker/scorer with one or more layers 
        if num_hidden > 0: 
            with tf.variable_scope("ctpmmatcherhidden") as scope: 
                h_tre_priorhidden = tf.contrib.layers.fully_connected(Da_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h_unt_priorhidden = tf.contrib.layers.fully_connected(Da_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("ctpmmatcher") as scope: 
                tre_priorscore = tf.contrib.layers.fully_connected(h_tre_priorhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                unt_priorscore = tf.contrib.layers.fully_connected(h_unt_priorhidden, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        else: 
            with tf.variable_scope("ctpmmatcher") as scope: 
                tre_priorscore = tf.contrib.layers.fully_connected(Da_tre, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                unt_priorscore = tf.contrib.layers.fully_connected(Da_unt, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        
        if num_hidden > 0: 
            with tf.variable_scope("ctpmmatcherhidden_b") as scope: 
                h_tre_matchhidden_b = tf.contrib.layers.fully_connected(D_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h_unt_matchhidden_b = tf.contrib.layers.fully_connected(D_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("ctpmmatcher_b") as scope: 
                tre_matchscore = tf.contrib.layers.fully_connected(h_tre_matchhidden_b, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                unt_matchscore = tf.contrib.layers.fully_connected(h_unt_matchhidden_b, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        else: 
            with tf.variable_scope("ctpmmatcher_b") as scope: 
                tre_matchscore = tf.contrib.layers.fully_connected(D_tre, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                unt_matchscore = tf.contrib.layers.fully_connected(D_unt, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        
        if num_hidden > 0: 
            with tf.variable_scope("ctpmpolicysighidden") as scope: 
                h_tre_policyhidden = tf.contrib.layers.fully_connected(D_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h_unt_policyhidden = tf.contrib.layers.fully_connected(D_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("ctpmpolicysig") as scope: 
                h_tre_policyscore = tf.contrib.layers.fully_connected(h_tre_policyhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_policyscore = tf.contrib.layers.fully_connected(h_unt_policyhidden, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        else: 
            with tf.variable_scope("ctpmpolicysig") as scope: 
                h_tre_policyscore = tf.contrib.layers.fully_connected(D_tre, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_policyscore = tf.contrib.layers.fully_connected(D_unt, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        
        ## this is the un-normalized bayesian weighting score with matching 
        s_tre_unnorm_1 = tf.math.multiply(tre_priorscore, tre_matchscore) 
        s_unt_unnorm_1 = tf.math.multiply(unt_priorscore, unt_matchscore) 
        
        s_tre_partfunc_1 = tf.reduce_sum(s_tre_unnorm_1) + 1e-17
        s_unt_partfunc_1 = tf.reduce_sum(s_unt_unnorm_1) + 1e-17
        
        ## this is the normalized bayesian weighting score for p(i, j) 
        s_tre_1 = tf.math.divide(s_tre_unnorm_1, s_tre_partfunc_1) 
        s_unt_1 = tf.math.divide(s_unt_unnorm_1, s_unt_partfunc_1) 
        
        ## use the bell-shape cost function in treatment intensity 
        diff_tre = np.reshape(i_tre, (-1, 1)) - h_tre_policyscore 
        diff_unt = np.reshape(i_unt, (-1, 1)) - h_unt_policyscore 
        
        lh_tre_policyscore = tf.math.multiply(tf.math.sigmoid(diff_tre), (1 - tf.math.sigmoid(diff_tre))) 
        lh_unt_policyscore = tf.math.multiply(tf.math.sigmoid(diff_unt), (1 - tf.math.sigmoid(diff_unt))) 
        
        ## this is the un-normalized bayesian weighting score 
        s_tre_unnorm = tf.math.multiply(s_tre_1, lh_tre_policyscore) 
        s_unt_unnorm = tf.math.multiply(s_unt_1, lh_unt_policyscore) 
        
        #tre_normalize_a = tf.nn.l2_normalize(h_tre_matchhidden,1)
        #tre_normalize_b = tf.nn.l2_normalize(h_tre_matchhidden_b,1)
        #tre_cos_similarity=tf.reduce_sum(tf.multiply(tre_normalize_a,tre_normalize_b), axis = 1)
        #tre_matchscore = 1 + tf.reshape(tre_cos_similarity, [-1, 1]) 
        
        #unt_normalize_a = tf.nn.l2_normalize(h_unt_matchhidden,1)
        #unt_normalize_b = tf.nn.l2_normalize(h_unt_matchhidden_b,1)
        #unt_cos_similarity=tf.reduce_sum(tf.multiply(unt_normalize_a,unt_normalize_b), axis = 1)
        #unt_matchscore = 1 + tf.reshape(unt_cos_similarity, [-1, 1])
        
        ## this is the un-normalized bayesian weighting score with continuous policy 
        s_tre_partfunc = tf.reduce_sum(s_tre_unnorm) + 1e-17 
        s_unt_partfunc = tf.reduce_sum(s_unt_unnorm) + 1e-17 
        
        ## this is the normalized bayesian weighting score with continuous policy 
        ss_tre = tf.math.divide(s_tre_unnorm, s_tre_partfunc) 
        ss_unt = tf.math.divide(s_unt_unnorm, s_unt_partfunc) 
        
        ### ---- start ----- 
        ### adopt a sorting operator that's also differentiable 
        ### for application of back-propagation and gradient optimization 
        h_tre_sorted = tf.contrib.framework.sort(ss_tre, axis=0, direction='DESCENDING') 
        h_unt_sorted = tf.contrib.framework.sort(ss_unt, axis=0, direction='DESCENDING') 
        
        top_k_tre = tf.cast(tf.ceil(size_tre * p_quantile), tf.int32) 
        top_k_unt = tf.cast(tf.ceil(size_unt * p_quantile), tf.int32) 
        
        intercept_tre = tf.slice(h_tre_sorted, [top_k_tre - 1, 0], [1, 1]) 
        intercept_unt = tf.slice(h_unt_sorted, [top_k_unt - 1, 0], [1, 1]) 
        
        ### stop gradients at the tunable intercept for sigmoid 
        ### to stabilize gradient-based optimization 
        intercept_tre = tf.stop_gradient(intercept_tre) 
        intercept_unt = tf.stop_gradient(intercept_unt) 
        
        ### use sigmoid to threshold top-k candidates, or use more sophisticated hinge loss 
        h_tre = tf.sigmoid(temp * (ss_tre - intercept_tre)) 
        h_unt = tf.sigmoid(temp * (ss_unt - intercept_unt)) 
        
        h_tre = tf.nn.dropout(h_tre, rate=dropout_rate) 
        h_unt = tf.nn.dropout(h_unt, rate=dropout_rate) 
        
        ### using softmax and weighted reduce-sum to compute the expected value 
        ### of treatment effect functions 
        s_tre = tf.nn.softmax(h_tre, axis=0) 
        s_unt = tf.nn.softmax(h_unt, axis=0) 
        
        ### ---- end ----- 
        
        s_tre = tf.reshape(s_tre, (size_tre, )) 
        s_unt = tf.reshape(s_unt, (size_unt, )) 
        
        dc_tre = tf.reduce_sum(tf.multiply(s_tre, c_tre)) 
        dc_unt = tf.reduce_sum(tf.multiply(s_unt, c_unt)) 
        
        do_mult_tre = tf.multiply(s_tre, o_tre) 
        do_mult_unt = tf.multiply(s_unt, o_unt) 
        
        do_tre = tf.reduce_sum(do_mult_tre) 
        do_unt = tf.reduce_sum(do_mult_unt) 
        
        dd_tre = tf.reduce_sum(tf.multiply(s_tre, d2d_tre)) 
        dd_unt = tf.reduce_sum(tf.multiply(s_unt, d2d_unt)) 
        
        ### implement the cost-gain effectiveness objective 
        #obj = tf.divide(tf.nn.leaky_relu(dc_tre - dc_unt), tf.nn.leaky_relu(do_tre - do_unt)) + d2dlamb * tf.nn.leaky_relu(dd_tre - dd_unt) 
        
        ### for application/production purposes, the above equation is more interpretable 
        ### and works well for most datasets, below is 
        ### an option to use relu, and math.log to ensure the objective is differentiable         
        #obj = tf.divide(tf.nn.leaky_relu(dc_tre - dc_unt), tf.nn.leaky_relu(do_tre - do_unt)) 
        #obj = tf.subtract(tf.math.log(tf.nn.leaky_relu(dc_tre - dc_unt)), tf.math.log(tf.nn.leaky_relu(do_tre - do_unt)))
        
        with tf.variable_scope("optimizer" + idstr) as scope: 
            opt = tf.train.AdamOptimizer().minimize(obj) 
        
        return obj, opt, s_tre, s_unt #h_tre_rnkscore, h_unt_rnkscore 

def forwardCTPMMatcherFeatureD2DistDNN(Dta, Dt, intt, num_hidden): 
    ### define ranker/scorer with one or more layers 
    if num_hidden > 0: 
        with tf.variable_scope("ctpmmatcherhidden") as scope: 
            h_tre_matchhidden = tf.contrib.layers.fully_connected(Dta, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
        with tf.variable_scope("ctpmmatcher") as scope: 
            h_tre_matchscore = tf.contrib.layers.fully_connected(h_tre_matchhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
    else: 
        with tf.variable_scope("ctpmmatcher") as scope: 
            h_tre_matchscore = tf.contrib.layers.fully_connected(Dta, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
    
    if num_hidden > 0: 
        with tf.variable_scope("ctpmmatcherhidden_b") as scope: 
            h_tre_matchhidden_b = tf.contrib.layers.fully_connected(Dt, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
        with tf.variable_scope("ctpmmatcher_b") as scope: 
            tre_matchscore = tf.contrib.layers.fully_connected(h_tre_matchhidden_b, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope)     
    else: 
        with tf.variable_scope("ctpmmatcher_b") as scope: 
            tre_matchscore = tf.contrib.layers.fully_connected(Dt, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
    
    if num_hidden > 0: 
        with tf.variable_scope("ctpmpolicysighidden") as scope: 
            h_tre_policyhidden = tf.contrib.layers.fully_connected(Dt, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
        with tf.variable_scope("ctpmpolicysig") as scope: 
            h_tre_policyscore = tf.contrib.layers.fully_connected(h_tre_policyhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
    else: 
        with tf.variable_scope("ctpmpolicysig") as scope: 
            h_tre_policyscore = tf.contrib.layers.fully_connected(Dt, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
    
    ## use the bell-shape cost function in treatment intensity 
    diff_tre = np.reshape(intt, (-1, 1)) - h_tre_policyscore 
    
    lh_tre_policyscore = tf.math.multiply(tf.math.sigmoid(diff_tre), (1 - tf.math.sigmoid(diff_tre))) 
    
    ## this is the un-normalized bayesian weighting score 
    s_tre_unnorm = tf.math.multiply(h_tre_matchscore, lh_tre_policyscore) 
    s_tre_unnorm = tf.math.multiply(s_tre_unnorm, tre_matchscore) 
    
    return s_tre_unnorm 

def CTPMMatcherFeatureD2DistDNN(graph, Da_tre, Da_unt, Db_tre, Db_unt, o_tre, o_unt, c_tre, c_unt, i_tre, i_unt, d2d_tre, d2d_unt, d2dlamb, idstr, num_hidden): 
    
    ## implements the Direct Ranking Model based on CPIT 
    ## D_tre: user context features for treatment 
    ## D_unt: user context features for control (untreated) 
    ## o_tre: user next week order labels for treatment 
    ## o_unt: user next week order labels for control (untreated) 
    ## c_tre: user next week cost(negative net-inflow/variable contribution) labels for treatment 
    ## c_unt: user next week cost(negative net-inflow/variable contribution) labels for control (untreated) 
    ## i_tre: treatment intensity of treatment group 
    ## i_unt: treatment intensity of control group 
    ## idstr: a string to indicate whether function is used for train/val, avoid tensorflor parameter re-use 
    
    ## returns: 
    ## obj, objective node 
    ## opt, optimizer node 
    ## h_tre_rnkscore, ranker scores for treated users 
    ## h_unt_rnkscore, ranker scores for control users 
    
    D_tre = np.concatenate((Da_tre, Db_tre), axis = 1) 
    D_unt = np.concatenate((Da_unt, Db_unt), axis = 1) 
    
    ## define size of cohort datasets 
    size_tre = D_tre.shape[0] 
    size_unt = D_unt.shape[0] 
    
    with graph.as_default() as g: 
        ### ------ define model graph of direct ranking ------ 
        
        ### define ranker/scorer with one or more layers 
        if num_hidden > 0: 
            with tf.variable_scope("ctpmmatcherhidden") as scope: 
                h_tre_priorhidden = tf.contrib.layers.fully_connected(Da_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h_unt_priorhidden = tf.contrib.layers.fully_connected(Da_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("ctpmmatcher") as scope: 
                tre_priorscore = tf.contrib.layers.fully_connected(h_tre_priorhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                unt_priorscore = tf.contrib.layers.fully_connected(h_unt_priorhidden, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        else: 
            with tf.variable_scope("ctpmmatcher") as scope: 
                tre_priorscore = tf.contrib.layers.fully_connected(Da_tre, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                unt_priorscore = tf.contrib.layers.fully_connected(Da_unt, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        
        if num_hidden > 0: 
            with tf.variable_scope("ctpmmatcherhidden_b") as scope: 
                h_tre_matchhidden_b = tf.contrib.layers.fully_connected(D_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h_unt_matchhidden_b = tf.contrib.layers.fully_connected(D_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("ctpmmatcher_b") as scope: 
                tre_matchscore = tf.contrib.layers.fully_connected(h_tre_matchhidden_b, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                unt_matchscore = tf.contrib.layers.fully_connected(h_unt_matchhidden_b, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        else: 
            with tf.variable_scope("ctpmmatcher_b") as scope: 
                tre_matchscore = tf.contrib.layers.fully_connected(D_tre, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                unt_matchscore = tf.contrib.layers.fully_connected(D_unt, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        
        if num_hidden > 0: 
            with tf.variable_scope("ctpmpolicysighidden") as scope: 
                h_tre_policyhidden = tf.contrib.layers.fully_connected(D_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h_unt_policyhidden = tf.contrib.layers.fully_connected(D_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("ctpmpolicysig") as scope: 
                h_tre_policyscore = tf.contrib.layers.fully_connected(h_tre_policyhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_policyscore = tf.contrib.layers.fully_connected(h_unt_policyhidden, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        else: 
            with tf.variable_scope("ctpmpolicysig") as scope: 
                h_tre_policyscore = tf.contrib.layers.fully_connected(D_tre, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_policyscore = tf.contrib.layers.fully_connected(D_unt, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        
        ## this is the un-normalized bayesian weighting score with matching 
        s_tre_unnorm_1 = tf.math.multiply(tre_priorscore, tre_matchscore) 
        s_unt_unnorm_1 = tf.math.multiply(unt_priorscore, unt_matchscore) 
        
        s_tre_partfunc_1 = tf.reduce_sum(s_tre_unnorm_1) + 1e-17
        s_unt_partfunc_1 = tf.reduce_sum(s_unt_unnorm_1) + 1e-17
        
        ## this is the normalized bayesian weighting score for p(i, j) 
        s_tre_1 = tf.math.divide(s_tre_unnorm_1, s_tre_partfunc_1) 
        s_unt_1 = tf.math.divide(s_unt_unnorm_1, s_unt_partfunc_1) 
        
        ## use the bell-shape cost function in treatment intensity 
        diff_tre = np.reshape(i_tre, (-1, 1)) - h_tre_policyscore 
        diff_unt = np.reshape(i_unt, (-1, 1)) - h_unt_policyscore 
        
        lh_tre_policyscore = tf.math.multiply(tf.math.sigmoid(diff_tre), (1 - tf.math.sigmoid(diff_tre))) 
        lh_unt_policyscore = tf.math.multiply(tf.math.sigmoid(diff_unt), (1 - tf.math.sigmoid(diff_unt))) 
        
        ## this is the un-normalized bayesian weighting score 
        s_tre_unnorm = tf.math.multiply(s_tre_1, lh_tre_policyscore) 
        s_unt_unnorm = tf.math.multiply(s_unt_1, lh_unt_policyscore) 
        
        #tre_normalize_a = tf.nn.l2_normalize(h_tre_matchhidden,1)
        #tre_normalize_b = tf.nn.l2_normalize(h_tre_matchhidden_b,1)
        #tre_cos_similarity=tf.reduce_sum(tf.multiply(tre_normalize_a,tre_normalize_b), axis = 1)
        #tre_matchscore = 1 + tf.reshape(tre_cos_similarity, [-1, 1]) 
        
        #unt_normalize_a = tf.nn.l2_normalize(h_unt_matchhidden,1)
        #unt_normalize_b = tf.nn.l2_normalize(h_unt_matchhidden_b,1)
        #unt_cos_similarity=tf.reduce_sum(tf.multiply(unt_normalize_a,unt_normalize_b), axis = 1)
        #unt_matchscore = 1 + tf.reshape(unt_cos_similarity, [-1, 1])
        
        ## this is the un-normalized bayesian weighting score with continuous policy 
        s_tre_partfunc = tf.reduce_sum(s_tre_unnorm) + 1e-17 
        s_unt_partfunc = tf.reduce_sum(s_unt_unnorm) + 1e-17 
        
        ## this is the normalized bayesian weighting score with continuous policy 
        s_tre = tf.math.divide(s_tre_unnorm, s_tre_partfunc) 
        s_unt = tf.math.divide(s_unt_unnorm, s_unt_partfunc) 
        
        s_tre = tf.reshape(s_tre, (size_tre, )) 
        s_unt = tf.reshape(s_unt, (size_unt, )) 
        
        dc_tre = tf.reduce_sum(tf.multiply(s_tre, c_tre)) 
        dc_unt = tf.reduce_sum(tf.multiply(s_unt, c_unt)) 
        
        do_mult_tre = tf.multiply(s_tre, o_tre) 
        do_mult_unt = tf.multiply(s_unt, o_unt) 
        
        do_tre = tf.reduce_sum(do_mult_tre) 
        do_unt = tf.reduce_sum(do_mult_unt) 
        
        dd_tre = tf.reduce_sum(tf.multiply(s_tre, d2d_tre)) 
        dd_unt = tf.reduce_sum(tf.multiply(s_unt, d2d_unt)) 
        
        ### implement the cost-gain effectiveness objective 
        obj = tf.divide(tf.nn.leaky_relu(dc_tre - dc_unt), tf.nn.leaky_relu(do_tre - do_unt)) + d2dlamb * tf.nn.leaky_relu(dd_tre - dd_unt) 
        
        ### for application/production purposes, the above equation is more interpretable 
        ### and works well for most datasets, below is 
        ### an option to use relu, and math.log to ensure the objective is differentiable         
        #obj = tf.divide(tf.nn.leaky_relu(dc_tre - dc_unt), tf.nn.leaky_relu(do_tre - do_unt)) 
        #obj = tf.subtract(tf.math.log(tf.nn.leaky_relu(dc_tre - dc_unt)), tf.math.log(tf.nn.leaky_relu(do_tre - do_unt)))
        
        with tf.variable_scope("optimizer" + idstr) as scope: 
            opt = tf.train.AdamOptimizer().minimize(obj) 
        
        return obj, opt, s_tre, s_unt #h_tre_rnkscore, h_unt_rnkscore 

def forwardCTPMMatcherD2DistDNN(Dta, Dtb, intt, num_hidden): 
    Dt = np.concatenate((Dta, Dtb), axis = 1) 
    if num_hidden > 0: 
        with tf.variable_scope("ctpmmatcherhidden_a") as scope: 
            h_tre_matchhidden = tf.contrib.layers.fully_connected(Dta, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
        with tf.variable_scope("ctpmmatcher") as scope: 
            h_tre_matchscore = tf.contrib.layers.fully_connected(h_tre_matchhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
    else: 
        with tf.variable_scope("ctpmmatcher") as scope: 
            h_tre_matchscore = tf.contrib.layers.fully_connected(Dta, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
        
    with tf.variable_scope("ctpmmatcherhidden_b") as scope: 
        h_tre_matchhidden_b = tf.contrib.layers.fully_connected(Dtb, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
    
    if num_hidden > 0: 
        with tf.variable_scope("ctpmpolicysighidden") as scope: 
            h_tre_policyhidden = tf.contrib.layers.fully_connected(Dt, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
        with tf.variable_scope("ctpmpolicysig") as scope: 
            h_tre_policyscore = tf.contrib.layers.fully_connected(h_tre_policyhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
    else: 
        with tf.variable_scope("ctpmpolicysig") as scope: 
            h_tre_policyscore = tf.contrib.layers.fully_connected(Dt, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
    
    tre_normalize_a = tf.nn.l2_normalize(h_tre_matchhidden,1)
    tre_normalize_b = tf.nn.l2_normalize(h_tre_matchhidden_b,1)
    tre_cos_similarity=tf.reduce_sum(tf.multiply(tre_normalize_a,tre_normalize_b), axis = 1)
    tre_matchscore = 1 + tf.reshape(tre_cos_similarity, [-1, 1]) 
    
    ## this is the un-normalized bayesian weighting score with matching 
    s_tre_unnorm_1 = tf.math.multiply(h_tre_matchscore, tre_matchscore) 
    
    s_tre_partfunc_1 = tf.reduce_sum(s_tre_unnorm_1) + 1e-17
    
    ## this is the normalized bayesian weighting score 
    s_tre_1 = tf.math.divide(s_tre_unnorm_1, s_tre_partfunc_1) 
    
    ## use the bell-shape cost function in treatment intensity 
    diff_tre = np.reshape(intt, (-1, 1)) - h_tre_policyscore 
    
    ## now use the bell-shape curve to weight using continuous policy 
    lh_tre_policyscore = tf.math.multiply(tf.math.sigmoid(diff_tre), (1 - tf.math.sigmoid(diff_tre))) 
    
    ## normalize with respect to policy 
    s_tre_unnorm = tf.math.multiply(s_tre_1, lh_tre_policyscore) 
    
    return s_tre_unnorm 

def CTPMMatcherD2DistDNN(graph, Da_tre, Da_unt, Db_tre, Db_unt, o_tre, o_unt, c_tre, c_unt, i_tre, i_unt, d2d_tre, d2d_unt, d2dlamb, idstr, num_hidden): 
    ## implements the Direct Ranking Model based on CPIT 
    ## D_tre: user context features for treatment 
    ## D_unt: user context features for control (untreated) 
    ## o_tre: user next week order labels for treatment 
    ## o_unt: user next week order labels for control (untreated) 
    ## c_tre: user next week cost(negative net-inflow/variable contribution) labels for treatment 
    ## c_unt: user next week cost(negative net-inflow/variable contribution) labels for control (untreated) 
    ## i_tre: treatment intensity of treatment group 
    ## i_unt: treatment intensity of control group 
    ## idstr: a string to indicate whether function is used for train/val, avoid tensorflor parameter re-use 
    
    ## returns: 
    ## obj, objective node 
    ## opt, optimizer node 
    ## h_tre_rnkscore, ranker scores for treated users 
    ## h_unt_rnkscore, ranker scores for control users 
    
    D_tre = np.concatenate((Da_tre, Db_tre), axis = 1) 
    D_unt = np.concatenate((Da_unt, Db_unt), axis = 1) 
    
    ## define size of cohort datasets 
    size_tre = D_tre.shape[0] 
    size_unt = D_unt.shape[0] 
    
    with graph.as_default() as g: 
        ### ------ define model graph of direct ranking ------ 
        
        ### define ranker/scorer with one or more layers 
        if num_hidden > 0: 
            with tf.variable_scope("ctpmmatcherhidden_a") as scope: 
                h_tre_matchhidden = tf.contrib.layers.fully_connected(Da_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h_unt_matchhidden = tf.contrib.layers.fully_connected(Da_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("ctpmmatcher") as scope: 
                h_tre_matchscore = tf.contrib.layers.fully_connected(h_tre_matchhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_matchscore = tf.contrib.layers.fully_connected(h_unt_matchhidden, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        else: 
            with tf.variable_scope("ctpmmatcher") as scope: 
                h_tre_matchscore = tf.contrib.layers.fully_connected(Da_tre, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_matchscore = tf.contrib.layers.fully_connected(Da_unt, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        
        with tf.variable_scope("ctpmmatcherhidden_b") as scope: 
            h_tre_matchhidden_b = tf.contrib.layers.fully_connected(Db_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            h_unt_matchhidden_b = tf.contrib.layers.fully_connected(Db_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            
        if num_hidden > 0: 
            with tf.variable_scope("ctpmpolicysighidden") as scope: 
                h_tre_policyhidden = tf.contrib.layers.fully_connected(D_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h_unt_policyhidden = tf.contrib.layers.fully_connected(D_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("ctpmpolicysig") as scope: 
                h_tre_policyscore = tf.contrib.layers.fully_connected(h_tre_policyhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_policyscore = tf.contrib.layers.fully_connected(h_unt_policyhidden, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        else: 
            with tf.variable_scope("ctpmpolicysig") as scope: 
                h_tre_policyscore = tf.contrib.layers.fully_connected(D_tre, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_policyscore = tf.contrib.layers.fully_connected(D_unt, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        
        tre_normalize_a = tf.nn.l2_normalize(h_tre_matchhidden,1)
        tre_normalize_b = tf.nn.l2_normalize(h_tre_matchhidden_b,1)
        tre_cos_similarity=tf.reduce_sum(tf.multiply(tre_normalize_a,tre_normalize_b), axis = 1)
        tre_matchscore = 1 + tf.reshape(tre_cos_similarity, [-1, 1]) 
        
        unt_normalize_a = tf.nn.l2_normalize(h_unt_matchhidden,1)
        unt_normalize_b = tf.nn.l2_normalize(h_unt_matchhidden_b,1)
        unt_cos_similarity=tf.reduce_sum(tf.multiply(unt_normalize_a,unt_normalize_b), axis = 1)
        unt_matchscore = 1 + tf.reshape(unt_cos_similarity, [-1, 1])
        
        ## this is the un-normalized bayesian weighting score with matching 
        s_tre_unnorm_1 = tf.math.multiply(h_tre_matchscore, tre_matchscore) 
        s_unt_unnorm_1 = tf.math.multiply(h_unt_matchscore, unt_matchscore) 
        
        s_tre_partfunc_1 = tf.reduce_sum(s_tre_unnorm_1) + 1e-17
        s_unt_partfunc_1 = tf.reduce_sum(s_unt_unnorm_1) + 1e-17
        
        ## this is the normalized bayesian weighting score 
        s_tre_1 = tf.math.divide(s_tre_unnorm_1, s_tre_partfunc_1) 
        s_unt_1 = tf.math.divide(s_unt_unnorm_1, s_unt_partfunc_1) 
        
        ## use the bell-shape cost function in treatment intensity 
        diff_tre = np.reshape(i_tre, (-1, 1)) - h_tre_policyscore 
        diff_unt = np.reshape(i_unt, (-1, 1)) - h_unt_policyscore 
        
        ## now use the bell-shape curve to weight using continuous policy 
        lh_tre_policyscore = tf.math.multiply(tf.math.sigmoid(diff_tre), (1 - tf.math.sigmoid(diff_tre))) 
        lh_unt_policyscore = tf.math.multiply(tf.math.sigmoid(diff_unt), (1 - tf.math.sigmoid(diff_unt))) 
        
        ## normalize with respect to policy 
        s_tre_unnorm = tf.math.multiply(s_tre_1, lh_tre_policyscore) 
        s_unt_unnorm = tf.math.multiply(s_unt_1, lh_unt_policyscore) 
        
        s_tre_partfunc = tf.reduce_sum(s_tre_unnorm) + 1e-17 
        s_unt_partfunc = tf.reduce_sum(s_unt_unnorm) + 1e-17 
        
        ## this is the normalized bayesian weighting score with cont. policy 
        s_tre = tf.math.divide(s_tre_unnorm, s_tre_partfunc) 
        s_unt = tf.math.divide(s_unt_unnorm, s_unt_partfunc) 
        
        s_tre = tf.reshape(s_tre, (size_tre, )) 
        s_unt = tf.reshape(s_unt, (size_unt, )) 
        
        dc_tre = tf.reduce_sum(tf.multiply(s_tre, c_tre)) 
        dc_unt = tf.reduce_sum(tf.multiply(s_unt, c_unt)) 
        
        do_mult_tre = tf.multiply(s_tre, o_tre) 
        do_mult_unt = tf.multiply(s_unt, o_unt) 
        
        do_tre = tf.reduce_sum(do_mult_tre) 
        do_unt = tf.reduce_sum(do_mult_unt) 
        
        dd_tre = tf.reduce_sum(tf.multiply(s_tre, d2d_tre)) 
        dd_unt = tf.reduce_sum(tf.multiply(s_unt, d2d_unt)) 
        
        ### implement the cost-gain effectiveness objective 
        #[ponpare] obj = tf.divide(tf.nn.leaky_relu(dc_tre - dc_unt), tf.nn.leaky_relu(do_tre - do_unt)) + d2dlamb * tf.nn.leaky_relu(dd_tre - dd_unt) 
        #[uscensus] obj = tf.multiply(-1.0 * (dd_tre - dd_unt), (do_tre - do_unt) - d2dlamb * (dc_tre - dc_unt)) 
        
        obj = tf.multiply(-1.0 * (dd_tre - dd_unt), (do_tre - do_unt) - d2dlamb * (dc_tre - dc_unt)) 
        #obj = tf.multiply(-1.0 * tf.nn.leaky_relu(1e-7 + dd_tre - dd_unt), d2dlamb * (tf.nn.leaky_relu(1e-7 + dc_tre - dc_unt) - tf.nn.leaky_relu(1e-7 + do_tre - do_unt))) 
        
        #-1.0 * tf.multiply( do_tre - do_unt, (dd_tre - dd_unt - d2dlamb * (dc_tre - dc_unt))) 
        
        ### for application/production purposes, the above equation is more interpretable 
        ### and works well for most datasets, below is 
        ### an option to use relu, and math.log to ensure the objective is differentiable 
        #obj = tf.divide(tf.nn.leaky_relu(dc_tre - dc_unt), tf.nn.leaky_relu(do_tre - do_unt)) 
        #obj = tf.subtract(tf.math.log(tf.nn.leaky_relu(dc_tre - dc_unt)), tf.math.log(tf.nn.leaky_relu(do_tre - do_unt))) 
        
        with tf.variable_scope("optimizer" + idstr) as scope: 
            opt = tf.train.AdamOptimizer().minimize(obj) 
        saver = tf.compat.v1.train.Saver() 
        return obj, opt, s_tre, s_unt, saver, h_tre_policyscore, h_unt_policyscore, h_tre_matchhidden, h_unt_matchhidden, h_tre_matchhidden_b, h_unt_matchhidden_b #h_tre_rnkscore, h_unt_rnkscore 

def CTPMDNN(graph, D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, i_tre, i_unt, idstr, num_hidden): 
    
    ## implements the Direct Ranking Model based on CPIT 
    ## D_tre: user context features for treatment 
    ## D_unt: user context features for control (untreated) 
    ## o_tre: user next week order labels for treatment 
    ## o_unt: user next week order labels for control (untreated) 
    ## c_tre: user next week cost(negative net-inflow/variable contribution) labels for treatment 
    ## c_unt: user next week cost(negative net-inflow/variable contribution) labels for control (untreated) 
    ## i_tre: treatment intensity of treatment group 
    ## i_unt: treatment intensity of control group 
    ## idstr: a string to indicate whether function is used for train/val, avoid tensorflor parameter re-use 
    
    ## returns: 
    ## obj, objective node 
    ## opt, optimizer node 
    ## h_tre_rnkscore, ranker scores for treated users 
    ## h_unt_rnkscore, ranker scores for control users 
    
    ## define size of cohort datasets 
    size_tre = D_tre.shape[0] 
    size_unt = D_unt.shape[0] 
    
    with graph.as_default() as g: 
        ### ------ define model graph of direct ranking ------ 
        
        ### define ranker/scorer with one or more layers 
        if num_hidden > 0: 
            with tf.variable_scope("ctpmmatcherhidden") as scope: 
                h_tre_matchhidden = tf.contrib.layers.fully_connected(D_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h_unt_matchhidden = tf.contrib.layers.fully_connected(D_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("ctpmmatcher") as scope: 
                h_tre_matchscore = tf.contrib.layers.fully_connected(h_tre_matchhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_matchscore = tf.contrib.layers.fully_connected(h_unt_matchhidden, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        else: 
            with tf.variable_scope("ctpmmatcher") as scope: 
                h_tre_matchscore = tf.contrib.layers.fully_connected(D_tre, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_matchscore = tf.contrib.layers.fully_connected(D_unt, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        
        if num_hidden > 0: 
            with tf.variable_scope("ctpmpolicysighidden") as scope: 
                h_tre_policyhidden = tf.contrib.layers.fully_connected(D_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h_unt_policyhidden = tf.contrib.layers.fully_connected(D_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("ctpmpolicysig") as scope: 
                h_tre_policyscore = tf.contrib.layers.fully_connected(h_tre_policyhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_policyscore = tf.contrib.layers.fully_connected(h_unt_policyhidden, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        else: 
            with tf.variable_scope("ctpmpolicysig") as scope: 
                h_tre_policyscore = tf.contrib.layers.fully_connected(D_tre, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_policyscore = tf.contrib.layers.fully_connected(D_unt, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        
        ## use the bell-shape cost function in treatment intensity 
        diff_tre = np.reshape(i_tre, (-1, 1)) - h_tre_policyscore 
        diff_unt = np.reshape(i_unt, (-1, 1)) - h_unt_policyscore 
        
        lh_tre_policyscore = tf.math.multiply(tf.math.sigmoid(diff_tre), (1 - tf.math.sigmoid(diff_tre))) 
        lh_unt_policyscore = tf.math.multiply(tf.math.sigmoid(diff_unt), (1 - tf.math.sigmoid(diff_unt))) 
        
        ## this is the un-normalized bayesian weighting score 
        s_tre_unnorm = tf.math.multiply(h_tre_matchscore, lh_tre_policyscore) 
        s_unt_unnorm = tf.math.multiply(h_unt_matchscore, lh_unt_policyscore) 
        
        s_tre_partfunc = tf.reduce_sum(s_tre_unnorm) + 1e-17
        s_unt_partfunc = tf.reduce_sum(s_unt_unnorm) + 1e-17
        
        ## this is the normalized bayesian weighting score 
        s_tre = tf.math.divide(s_tre_unnorm, s_tre_partfunc) 
        s_unt = tf.math.divide(s_unt_unnorm, s_unt_partfunc) 
        
        s_tre = tf.reshape(s_tre, (size_tre, )) 
        s_unt = tf.reshape(s_unt, (size_unt, )) 
        
        dc_tre = tf.reduce_sum(tf.multiply(s_tre, c_tre)) 
        dc_unt = tf.reduce_sum(tf.multiply(s_unt, c_unt)) 
        
        do_mult_tre = tf.multiply(s_tre, o_tre) 
        do_mult_unt = tf.multiply(s_unt, o_unt) 
        
        do_tre = tf.reduce_sum(do_mult_tre) 
        do_unt = tf.reduce_sum(do_mult_unt) 
        
        ### implement the cost-gain effectiveness objective 
        obj = tf.divide(dc_tre - dc_unt, do_tre - do_unt) 
        
        ### for application/production purposes, the above equation is more interpretable 
        ### and works well for most datasets, below is 
        ### an option to use relu, and math.log to ensure the objective is differentiable         
        #obj = tf.divide(tf.nn.leaky_relu(dc_tre - dc_unt), tf.nn.leaky_relu(do_tre - do_unt)) 
        #obj = tf.subtract(tf.math.log(tf.nn.leaky_relu(dc_tre - dc_unt)), tf.math.log(tf.nn.leaky_relu(do_tre - do_unt)))
        
        with tf.variable_scope("optimizer" + idstr) as scope: 
            opt = tf.train.AdamOptimizer().minimize(obj) 
        
        return obj, opt, s_tre, s_unt #h_tre_rnkscore, h_unt_rnkscore 

def CTPMMatcherDNN(graph, Da_tre, Da_unt, Db_tre, Db_unt, o_tre, o_unt, c_tre, c_unt, i_tre, i_unt, idstr, num_hidden): 
    
    ## implements the Direct Ranking Model based on CPIT 
    ## D_tre: user context features for treatment 
    ## D_unt: user context features for control (untreated) 
    ## o_tre: user next week order labels for treatment 
    ## o_unt: user next week order labels for control (untreated) 
    ## c_tre: user next week cost(negative net-inflow/variable contribution) labels for treatment 
    ## c_unt: user next week cost(negative net-inflow/variable contribution) labels for control (untreated) 
    ## i_tre: treatment intensity of treatment group 
    ## i_unt: treatment intensity of control group 
    ## idstr: a string to indicate whether function is used for train/val, avoid tensorflor parameter re-use 
    
    ## returns: 
    ## obj, objective node 
    ## opt, optimizer node 
    ## h_tre_rnkscore, ranker scores for treated users 
    ## h_unt_rnkscore, ranker scores for control users 
    
    D_tre = np.concatenate((Da_tre, Db_tre), axis = 1) 
    D_unt = np.concatenate((Da_unt, Db_unt), axis = 1) 
    
    ## define size of cohort datasets 
    size_tre = D_tre.shape[0] 
    size_unt = D_unt.shape[0] 
    
    with graph.as_default() as g: 
        ### ------ define model graph of direct ranking ------ 
        
        ### define ranker/scorer with one or more layers 
        if num_hidden > 0: 
            with tf.variable_scope("ctpmmatcherhidden_a") as scope: 
                h_tre_matchhidden = tf.contrib.layers.fully_connected(Da_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h_unt_matchhidden = tf.contrib.layers.fully_connected(Da_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            #with tf.variable_scope("ctpmmatcherhidden_2_a") as scope: 
            #    h_tre_matchhidden_2 = tf.contrib.layers.fully_connected(h_tre_matchhidden_1, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            #    h_unt_matchhidden_2 = tf.contrib.layers.fully_connected(Da_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("ctpmmatcher") as scope: 
                h_tre_matchscore = tf.contrib.layers.fully_connected(h_tre_matchhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_matchscore = tf.contrib.layers.fully_connected(h_unt_matchhidden, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        else: 
            with tf.variable_scope("ctpmmatcher") as scope: 
                h_tre_matchscore = tf.contrib.layers.fully_connected(Da_tre, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_matchscore = tf.contrib.layers.fully_connected(Da_unt, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        
        with tf.variable_scope("ctpmmatcherhidden_b") as scope: 
            h_tre_matchhidden_b = tf.contrib.layers.fully_connected(Db_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            h_unt_matchhidden_b = tf.contrib.layers.fully_connected(Db_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            
        if num_hidden > 0: 
            with tf.variable_scope("ctpmpolicysighidden") as scope: 
                h_tre_policyhidden = tf.contrib.layers.fully_connected(D_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h_unt_policyhidden = tf.contrib.layers.fully_connected(D_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("ctpmpolicysig") as scope: 
                h_tre_policyscore = tf.contrib.layers.fully_connected(h_tre_policyhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_policyscore = tf.contrib.layers.fully_connected(h_unt_policyhidden, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        else: 
            with tf.variable_scope("ctpmpolicysig") as scope: 
                h_tre_policyscore = tf.contrib.layers.fully_connected(D_tre, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_policyscore = tf.contrib.layers.fully_connected(D_unt, 1, activation_fn=tf.nn.sigmoid, reuse=True, scope=scope) 
        
        ## use the bell-shape cost function in treatment intensity 
        diff_tre = np.reshape(i_tre, (-1, 1)) - h_tre_policyscore 
        diff_unt = np.reshape(i_unt, (-1, 1)) - h_unt_policyscore 
        
        lh_tre_policyscore = tf.math.multiply(tf.math.sigmoid(diff_tre), (1 - tf.math.sigmoid(diff_tre))) 
        lh_unt_policyscore = tf.math.multiply(tf.math.sigmoid(diff_unt), (1 - tf.math.sigmoid(diff_unt))) 
        
        ## this is the un-normalized bayesian weighting score 
        s_tre_unnorm = tf.math.multiply(h_tre_matchscore, lh_tre_policyscore) 
        s_unt_unnorm = tf.math.multiply(h_unt_matchscore, lh_unt_policyscore) 
        
        tre_normalize_a = tf.nn.l2_normalize(h_tre_matchhidden,1)
        tre_normalize_b = tf.nn.l2_normalize(h_tre_matchhidden_b,1)
        tre_cos_similarity=tf.reduce_sum(tf.multiply(tre_normalize_a,tre_normalize_b), axis = 1)
        tre_matchscore = 1 + tf.reshape(tre_cos_similarity, [-1, 1]) 
        
        unt_normalize_a = tf.nn.l2_normalize(h_unt_matchhidden,1)
        unt_normalize_b = tf.nn.l2_normalize(h_unt_matchhidden_b,1)
        unt_cos_similarity=tf.reduce_sum(tf.multiply(unt_normalize_a,unt_normalize_b), axis = 1)
        unt_matchscore = 1 + tf.reshape(unt_cos_similarity, [-1, 1])
        
        ## this is the un-normalized bayesian weighting score with matching 
        s_tre_unnorm = tf.math.multiply(s_tre_unnorm, tre_matchscore) 
        s_unt_unnorm = tf.math.multiply(s_unt_unnorm, unt_matchscore) 
        
        s_tre_partfunc = tf.reduce_sum(s_tre_unnorm) + 1e-17
        s_unt_partfunc = tf.reduce_sum(s_unt_unnorm) + 1e-17
        
        ## this is the normalized bayesian weighting score 
        s_tre = tf.math.divide(s_tre_unnorm, s_tre_partfunc) 
        s_unt = tf.math.divide(s_unt_unnorm, s_unt_partfunc) 
        
        s_tre = tf.reshape(s_tre, (size_tre, )) 
        s_unt = tf.reshape(s_unt, (size_unt, )) 
        
        dc_tre = tf.reduce_sum(tf.multiply(s_tre, c_tre)) 
        dc_unt = tf.reduce_sum(tf.multiply(s_unt, c_unt)) 
        
        do_mult_tre = tf.multiply(s_tre, o_tre) 
        do_mult_unt = tf.multiply(s_unt, o_unt) 
        
        do_tre = tf.reduce_sum(do_mult_tre) 
        do_unt = tf.reduce_sum(do_mult_unt) 
        
        ### implement the cost-gain effectiveness objective 
        obj = tf.divide(tf.nn.leaky_relu(dc_tre - dc_unt), tf.nn.leaky_relu(do_tre - do_unt)) 
        
        ### for application/production purposes, the above equation is more interpretable 
        ### and works well for most datasets, below is 
        ### an option to use relu, and math.log to ensure the objective is differentiable         
        #obj = tf.divide(tf.nn.leaky_relu(dc_tre - dc_unt), tf.nn.leaky_relu(do_tre - do_unt)) 
        #obj = tf.subtract(tf.math.log(tf.nn.leaky_relu(dc_tre - dc_unt)), tf.math.log(tf.nn.leaky_relu(do_tre - do_unt)))
        
        with tf.variable_scope("optimizer" + idstr) as scope: 
            opt = tf.train.AdamOptimizer().minimize(obj) 
        
        return obj, opt, s_tre, s_unt #h_tre_rnkscore, h_unt_rnkscore 

def TunableTQRankingModelDNN(graph, D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, idstr, initial_temp, p_quantile, num_hidden, use_schedule=False): 
    ## implements the top-p-quantile operator for Constrained Ranking Model 
    ## with tunable temperature through gradient descent and temperature schedule 
    ## D_tre: user context features for treatment 
    ## D_unt: user context features for control (untreated) 
    ## o_tre: user next week order labels for treatment 
    ## o_unt: user next week order labels for control (untreated) 
    ## c_tre: user next week cost(negative net-inflow/variable contribution) labels for treatment 
    ## c_unt: user next week cost(negative net-inflow/variable contribution) labels for control (untreated)  
    ## p_quantile: the top-p-quantile number between (0, 1) 
    ## idstr: a string to indicate whether function is used for train/val, avoid tensorflor parameter re-use 
    ## initial_temp: initial temperature (this is tunable) of the sigmoid governing p_quantile cut-off 
    
    ## returns: 
    ## obj, objective node 
    ## opt, optimizer node 
    ## h_tre_rnkscore, ranker scores for treated users 
    ## h_unt_rnkscore, ranker scores for control users 
    
    ## Consider the contrast TQRanking Model 
    ## as opposed to improving cpit upon control cohort, 
    ## [E(Ctqr) - E(Cctrl)] / [E(Ttqr) - E(Tctrl)] 
    ## let's think about improving cpit upon treatment cohort 
    ## [E(Ctqr) - E(Ctre)] / [E(Ttqr) - E(Ttre)] 
    ## or, let's think about improving upon DRM 
    ## 
    
    ## temperature of the sigmoid governing p_quantile cut-off 
    with graph.as_default() as g: 
        #if use_schedule == False: 
        init = tf.constant(initial_temp, dtype=tf.float64) 
        with tf.variable_scope("temp", reuse=tf.AUTO_REUSE) as scope: 
            temp = tf.get_variable('temp', initializer=init, dtype=tf.float64) 
        ### ---- the following code makes the temperature tunable ---- 
        ### deleted for use of temperature schedule, but keep for future applications 
        #else: 
        #    temp = tf.constant(initial_temp, dtype=tf.float64) 
            #tf.Variable(2.5, dtype=tf.float64, trainable=True) 
        #init2 = tf.constant(p_quantile, dtype=tf.float64) 
        #with tf.variable_scope("p_quantile", reuse=tf.AUTO_REUSE) as scope: 
        #    p_quantile = tf.get_variable('p_quantile', initializer=init2, dtype=tf.float64)
            #tf.Variable(0.3, dtype=tf.float32, trainable=True, reuse=tf.AUTO_REUSE)
        
        ## define size of cohort datasets 
        size_tre = D_tre.shape[0] 
        size_unt = D_unt.shape[0] 
        
        ### ----- define model graph of Top Quantile Constrained ranking ----- 

        ### we can define either a linear or a multi-layer neural network 
        ### for the ranker or scorer 
        if num_hidden > 0: 
            with tf.variable_scope("tqrhidden") as scope: 
                h1_tre = tf.contrib.layers.fully_connected(D_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h1_unt = tf.contrib.layers.fully_connected(D_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("tqranker") as scope: 
                h_tre_rnkscore = tf.contrib.layers.fully_connected(h1_tre, 1, activation_fn=None, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h_unt_rnkscore = tf.contrib.layers.fully_connected(h1_unt, 1, activation_fn=None, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
        else: 
            with tf.variable_scope("tqranker") as scope: 
                h_tre_rnkscore = tf.contrib.layers.fully_connected(D_tre, 1, activation_fn=None, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h_unt_rnkscore = tf.contrib.layers.fully_connected(D_unt, 1, activation_fn=None, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
        
        ### adopt a sorting operator that's also differentiable 
        ### for application of back-propagation and gradient optimization 
        h_tre_sorted = tf.contrib.framework.sort(h_tre_rnkscore, axis=0, direction='DESCENDING') 
        h_unt_sorted = tf.contrib.framework.sort(h_unt_rnkscore, axis=0, direction='DESCENDING') 
        
        top_k_tre = tf.cast(tf.ceil(size_tre * p_quantile), tf.int32) 
        top_k_unt = tf.cast(tf.ceil(size_unt * p_quantile), tf.int32) 
        
        intercept_tre = tf.slice(h_tre_sorted, [top_k_tre - 1, 0], [1, 1]) 
        intercept_unt = tf.slice(h_unt_sorted, [top_k_unt - 1, 0], [1, 1]) 
        
        ### stop gradients at the tunable intercept for sigmoid 
        ### to stabilize gradient-based optimization 
        intercept_tre = tf.stop_gradient(intercept_tre) 
        intercept_unt = tf.stop_gradient(intercept_unt) 
        
        ### use sigmoid to threshold top-k candidates, or use more sophisticated hinge loss 
        h_tre = tf.sigmoid(temp * (h_tre_rnkscore - intercept_tre)) 
        h_unt = tf.sigmoid(temp * (h_unt_rnkscore - intercept_unt)) 
        
        ### using softmax and weighted reduce-sum to compute the expected value 
        ### of treatment effect functions 
        s_tre = tf.nn.softmax(h_tre, axis=0) 
        s_unt = tf.nn.softmax(h_unt, axis=0) 
        
        s_tre = tf.reshape(s_tre, (size_tre, ))
        s_unt = tf.reshape(s_unt, (size_unt, ))
        
        dc_tre = tf.reduce_sum(tf.multiply(s_tre, c_tre)) 
        dc_unt = tf.reduce_sum(tf.multiply(s_unt, c_unt)) 
        
        do_mult_tre = tf.multiply(s_tre, o_tre) 
        do_mult_unt = tf.multiply(s_unt, o_unt) 
        
        do_tre = tf.reduce_sum(do_mult_tre) 
        do_unt = tf.reduce_sum(do_mult_unt) 
        
        ### implement the cost-gain effectiveness objective 
        obj = tf.divide(dc_tre - dc_unt, do_tre - do_unt) 
        ### for application/production purposes, the above equation is more interpretable 
        ### and works well for most datasets, below is 
        ### an option to use relu, and math.log to ensure the objective is differentiable 
        #obj = tf.divide(tf.nn.leaky_relu(dc_tre - dc_unt), tf.nn.leaky_relu(do_tre - do_unt)) 
        #obj = tf.subtract(tf.math.log(tf.nn.leaky_relu(dc_tre - dc_unt)), tf.math.log(tf.nn.leaky_relu(do_tre - do_unt)))
        
        with tf.variable_scope("optimizer" + idstr) as scope: 
            opt = tf.train.AdamOptimizer().minimize(obj) 
    
    return obj, opt, h_tre_rnkscore, h_unt_rnkscore, temp, p_quantile 

def forwardSimpleTCModelDNN(Dt, num_hidden): 
    if num_hidden > 0: 
        with tf.variable_scope("drmhidden") as scope: 
            h1_test = tf.contrib.layers.fully_connected(Dt, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
        with tf.variable_scope("drmranker") as scope: 
            h_test = tf.contrib.layers.fully_connected(h1_test, 1, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope) 
    else: 
        with tf.variable_scope("drmranker") as scope: 
            h_test = tf.contrib.layers.fully_connected(Dt, 1, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope) 
    return h_test 

def SimpleTCModelDNN(graph, D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, idstr, num_hidden): 
    
    ## implements the Direct Ranking Model based on CPIT 
    ## D_tre: user context features for treatment 
    ## D_unt: user context features for control (untreated) 
    ## o_tre: user next week order labels for treatment 
    ## o_unt: user next week order labels for control (untreated) 
    ## c_tre: user next week cost(negative net-inflow/variable contribution) labels for treatment 
    ## c_unt: user next week cost(negative net-inflow/variable contribution) labels for control (untreated)  
    ## idstr: a string to indicate whether function is used for train/val, avoid tensorflor parameter re-use 
    
    ## returns: 
    ## obj, objective node 
    ## opt, optimizer node 
    ## h_tre_rnkscore, ranker scores for treated users 
    ## h_unt_rnkscore, ranker scores for control users 
    
    ## define size of cohort datasets 
    size_tre = D_tre.shape[0] 
    size_unt = D_unt.shape[0] 
    
    with graph.as_default() as g: 
        ### ------ define model graph of direct ranking ------ 
        
        ### define ranker/scorer with one or more layers 
        if num_hidden > 0: 
            with tf.variable_scope("drmhidden") as scope: 
                h1_tre = tf.contrib.layers.fully_connected(D_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h1_unt = tf.contrib.layers.fully_connected(D_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("drmranker") as scope: 
                h_tre_rnkscore = tf.contrib.layers.fully_connected(h1_tre, 1, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_rnkscore = tf.contrib.layers.fully_connected(h1_unt, 1, activation_fn=tf.nn.tanh, reuse=True, scope=scope) 
        else: 
            with tf.variable_scope("drmranker") as scope: 
                h_tre_rnkscore = tf.contrib.layers.fully_connected(D_tre, 1, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_rnkscore = tf.contrib.layers.fully_connected(D_unt, 1, activation_fn=tf.nn.tanh, reuse=True, scope=scope) 
        
        ### use softmax normalization and weighted reduce-sum for 
        ### compute of expected value of treatment effects 
        s_tre = tf.nn.softmax(h_tre_rnkscore, axis=0) 
        s_unt = tf.nn.softmax(h_unt_rnkscore, axis=0) 
        
        s_tre = tf.reshape(s_tre, (size_tre, )) 
        s_unt = tf.reshape(s_unt, (size_unt, )) 
        
        dc_tre = tf.reduce_sum(tf.multiply(s_tre, c_tre)) 
        dc_unt = tf.reduce_sum(tf.multiply(s_unt, c_unt)) 
        
        do_mult_tre = tf.multiply(s_tre, o_tre) 
        do_mult_unt = tf.multiply(s_unt, o_unt) 
        
        do_tre = tf.reduce_sum(do_mult_tre) 
        do_unt = tf.reduce_sum(do_mult_unt) 
        
        ### implement the cost-gain effectiveness objective 
        obj = tf.divide(dc_tre - dc_unt, do_tre - do_unt)         
        
        ### for application/production purposes, the above equation is more interpretable 
        ### and works well for most datasets, below is 
        ### an option to use relu, and math.log to ensure the objective is differentiable         
        #obj = tf.divide(tf.nn.leaky_relu(dc_tre - dc_unt), tf.nn.leaky_relu(do_tre - do_unt)) 
        #obj = tf.subtract(tf.math.log(tf.nn.leaky_relu(dc_tre - dc_unt)), tf.math.log(tf.nn.leaky_relu(do_tre - do_unt)))
        
        with tf.variable_scope("optimizer" + idstr) as scope: 
            opt = tf.train.AdamOptimizer().minimize(obj) 
        saver = tf.compat.v1.train.Saver()         
        return obj, opt, h_tre_rnkscore, h_unt_rnkscore, saver  

def SimpleTCModelD2DistDNN(graph, D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, d2d_tre, d2d_unt, d2dlamb, idstr, num_hidden): 
    
    ## implements the Direct Ranking Model based on CPIT 
    ## D_tre: user context features for treatment 
    ## D_unt: user context features for control (untreated) 
    ## o_tre: user next week order labels for treatment 
    ## o_unt: user next week order labels for control (untreated) 
    ## c_tre: user next week cost(negative net-inflow/variable contribution) labels for treatment 
    ## c_unt: user next week cost(negative net-inflow/variable contribution) labels for control (untreated)  
    ## idstr: a string to indicate whether function is used for train/val, avoid tensorflor parameter re-use 
    
    ## returns: 
    ## obj, objective node 
    ## opt, optimizer node 
    ## h_tre_rnkscore, ranker scores for treated users 
    ## h_unt_rnkscore, ranker scores for control users 
    
    ## define size of cohort datasets 
    size_tre = D_tre.shape[0] 
    size_unt = D_unt.shape[0] 
    
    with graph.as_default() as g: 
        ### ------ define model graph of direct ranking ------ 
        
        ### define ranker/scorer with one or more layers 
        if num_hidden > 0: 
            with tf.variable_scope("drmhidden") as scope: 
                h1_tre = tf.contrib.layers.fully_connected(D_tre, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
                h1_unt = tf.contrib.layers.fully_connected(D_unt, num_hidden, activation_fn=tf.nn.tanh, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("drmranker") as scope: 
                h_tre_rnkscore = tf.contrib.layers.fully_connected(h1_tre, 1, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_rnkscore = tf.contrib.layers.fully_connected(h1_unt, 1, activation_fn=tf.nn.tanh, reuse=True, scope=scope) 
        else: 
            with tf.variable_scope("drmranker") as scope: 
                h_tre_rnkscore = tf.contrib.layers.fully_connected(D_tre, 1, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope) 
                h_unt_rnkscore = tf.contrib.layers.fully_connected(D_unt, 1, activation_fn=tf.nn.tanh, reuse=True, scope=scope) 
        
        ### use softmax normalization and weighted reduce-sum for 
        ### compute of expected value of treatment effects 
        s_tre = tf.nn.softmax(h_tre_rnkscore, axis=0) 
        s_unt = tf.nn.softmax(h_unt_rnkscore, axis=0) 
        
        s_tre = tf.reshape(s_tre, (size_tre, )) 
        s_unt = tf.reshape(s_unt, (size_unt, )) 
        
        dc_tre = tf.reduce_sum(tf.multiply(s_tre, c_tre)) 
        dc_unt = tf.reduce_sum(tf.multiply(s_unt, c_unt)) 
        
        do_mult_tre = tf.multiply(s_tre, o_tre) 
        do_mult_unt = tf.multiply(s_unt, o_unt) 
        
        do_tre = tf.reduce_sum(do_mult_tre) 
        do_unt = tf.reduce_sum(do_mult_unt) 
        
        dd_tre = tf.reduce_sum(tf.multiply(s_tre, d2d_tre)) 
        dd_unt = tf.reduce_sum(tf.multiply(s_unt, d2d_unt)) 
        
        ### implement the cost-gain effectiveness objective 
        #obj = tf.divide(dc_tre - dc_unt, do_tre - do_unt) 
        
        #obj = -1.0 * tf.multiply( do_tre - do_unt, (dd_tre - dd_unt - d2dlamb * (dc_tre - dc_unt))) 
        #obj = tf.multiply(-1.0 * tf.nn.leaky_relu(1e-7 + dd_tre - dd_unt), tf.nn.leaky_relu(1e-7 + do_tre - do_unt) - d2dlamb * (tf.nn.leaky_relu(1e-7 + dc_tre - dc_unt)))  
        obj = tf.multiply(-1.0 * (dd_tre - dd_unt), do_tre - do_unt - d2dlamb * (dc_tre - dc_unt)) 
        
        #obj = tf.divide(tf.nn.leaky_relu(dc_tre - dc_unt), tf.nn.leaky_relu(do_tre - do_unt)) + d2dlamb * tf.nn.leaky_relu(dd_tre - dd_unt) 
        
        ### for application/production purposes, the above equation is more interpretable 
        ### and works well for most datasets, below is 
        ### an option to use relu, and math.log to ensure the objective is differentiable         
        #obj = tf.divide(tf.nn.leaky_relu(dc_tre - dc_unt), tf.nn.leaky_relu(do_tre - do_unt)) 
        #obj = tf.subtract(tf.math.log(tf.nn.leaky_relu(dc_tre - dc_unt)), tf.math.log(tf.nn.leaky_relu(do_tre - do_unt)))
        
        with tf.variable_scope("optimizer" + idstr) as scope: 
            opt = tf.train.AdamOptimizer().minimize(obj) 
        
        return obj, opt, h_tre_rnkscore, h_unt_rnkscore 

def OnOffSimpleTCModelDNN(graph, D, o, c, idstr, num_hidden): 
    
    ## implements the Direct Ranking Model based on CPIT 
    ## D_tre: user context features for treatment 
    ## D_unt: user context features for control (untreated) 
    ## o_tre: user next week order labels for treatment 
    ## o_unt: user next week order labels for control (untreated) 
    ## c_tre: user next week cost(negative net-inflow/variable contribution) labels for treatment 
    ## c_unt: user next week cost(negative net-inflow/variable contribution) labels for control (untreated)  
    ## idstr: a string to indicate whether function is used for train/val, avoid tensorflor parameter re-use 
    
    ## returns: 
    ## obj, objective node 
    ## opt, optimizer node 
    ## h_tre_rnkscore, ranker scores for treated users 
    ## h_unt_rnkscore, ranker scores for control users 
    
    ## define size of cohort datasets 
    size_tre = D.shape[0] 
    
    with graph.as_default() as g: 
        ### ------ define model graph of direct ranking ------ 
        
        ### define ranker/scorer with one or more layers 
        if num_hidden > 0: 
            with tf.variable_scope("drmhidden") as scope: 
                h1_tre = tf.contrib.layers.fully_connected(D, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("drmranker") as scope: 
                h_tre_rnkscore = tf.contrib.layers.fully_connected(h1_tre, 1, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope) 
        else: 
            with tf.variable_scope("drmranker") as scope: 
                h_tre_rnkscore = tf.contrib.layers.fully_connected(D, 1, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope) 
        
        ### use softmax normalization and weighted reduce-sum for 
        ### compute of expected value of treatment effects 
        s_tre = tf.nn.softmax(h_tre_rnkscore, axis=0) 
        
        #s_tre = tf.reshape(s_tre, (size_tre, )) 
        #s_unt = tf.reshape(s_unt, (size_unt, )) 
        
        s_on = tf.reshape(s_tre, (size_tre, )) 
        s_off = np.ones((size_tre, )) * 1.0 / size_tre 
        
        dc_on = tf.reduce_sum(tf.multiply(s_on, c)) 
        dc_off = tf.reduce_sum(tf.multiply(s_off, c)) ## replace this later with entire dataset 
        
        do_mult_on = tf.multiply(s_on, o) 
        do_mult_off = tf.multiply(s_off, o) 
        
        do_on = tf.reduce_sum(do_mult_on) 
        do_off = tf.reduce_sum(do_mult_off) 
        
        ### implement the cost-gain effectiveness objective 
        obj = tf.divide(tf.nn.relu(dc_on - dc_off), tf.nn.relu(do_on - do_off)) 
        
        ### for application/production purposes, the above equation is more interpretable 
        ### and works well for most datasets, below is 
        ### an option to use relu, and math.log to ensure the objective is differentiable         
        #obj = tf.divide(tf.nn.leaky_relu(dc_tre - dc_unt), tf.nn.leaky_relu(do_tre - do_unt)) 
        #obj = tf.subtract(tf.math.log(tf.nn.leaky_relu(dc_tre - dc_unt)), tf.math.log(tf.nn.leaky_relu(do_tre - do_unt)))
        
        with tf.variable_scope("optimizer" + idstr) as scope: 
            opt = tf.train.AdamOptimizer().minimize(obj) 
        
        return obj, opt, s_on, s_off 

def HetSequentialCausalModelDropout(num_units, features, seq_lens, treat_labels, trips_labels, cost_labels, gb_labels, exist_labels, idstr, dropout_ratio=0.5): 
    ### Defines the Heterogeneous Seqential Causal Model, builds tensorflow graph 
    ### 
    ### Inputs: 
    ### features, 3D tensor of shape [dataset_size, num_weeks, feature_dims] including the treatment labels 
    ###           num_weeks = 19/11; feature_dims = 18 with user features  
    ### seq_lens, 2D tensor of shape [dataset_size, 1] indicating length of sequence for each user 
    ### treat_labels, 2D tensor of shape [dataset_size, 1] indicating treatment for user in XP week 
    ### trips_labels, 2D tensor of shapexb [dataset_size, 1], trips number for user in XP week 
    ### cost_labels, 2D tensor of shape [dataset_size, 1], cost (-ve net-inflow/variable contribution) for users in XP week 
    ### gb_labels, 2D tensor of shape [dataset_size, 1], gross-bookings for users in XP week 
    ### exist_labels, 2D tensor of shape [dataset_size, 1] indicating whether data exists for the user in that XP week 
    ### idstr: a string to indicate whether function is used for train/val, avoid tensorflor parameter re-use 
    
    ### Returns: 
    ### cpitobj, cpit objective node 
    ### cpitopt, cpit optimizer node 
    ### cpigbobj, cpigb objective node 
    ### cpigbopt, cpigb optimizer node 
    
    ### recurrent model architecture 
    ### e.g. 3 layer each layer 8 units will be [8, 8, 8]         
    
    ### optimizer parameters 
    adam_learning_rate = 0.01 
    adam_beta1 = 0.5 
    adam_beta2 = 0.999 
    gradient_clip_norm_value = 2.3 
    
    ### optional (rarely used) l2 regularization 
    l2_reg_weight = 0.000 
    
    num_weeks = features.shape[1] 
    ### propagate forward the data through the LSTM and through a ranker 
    with tf.variable_scope("hscls_model_lstm") as scope: 
        cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n, reuse=tf.AUTO_REUSE) for n in num_units] 
        stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)         
        outputs, state = tf.nn.dynamic_rnn(stacked_rnn_cell, 
                                           inputs=features, 
                                           sequence_length=seq_lens, 
                                           dtype = np.float64 
                                           ) 
    with tf.variable_scope("hscls_model_extra_layer") as scope: 
        extra_layer = tf.contrib.layers.fully_connected(outputs, 16, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
    tf.layers.dropout(extra_layer, rate = dropout_ratio) 
    with tf.variable_scope("hscls_model_scoring_layer") as scope: 
        h_rscore = tf.contrib.layers.fully_connected(extra_layer, 1, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
    
    ### branch out scores for treatment and control 
    h_rscore_tr = tf.reshape(h_rscore, (-1, num_weeks)) 
    h_rscore_un = tf.reshape(h_rscore, (-1, num_weeks)) 
    
    ### now, mask the score with the treatment label for treatment cohort and control cohort 
    untreat_labels = tf.cast(tf.multiply( 
            exist_labels, ## data exists for cost and trips 
            tf.cast(tf.logical_not(tf.cast(treat_labels, tf.bool)), tf.float64)), tf.float64) 
    masked_h_rscore_tr = tf.multiply(treat_labels, h_rscore_tr) + np.finfo(np.float32).min * tf.cast(tf.logical_not(tf.cast(treat_labels, tf.bool)), tf.float64)
    masked_h_rscore_un = tf.multiply(untreat_labels, h_rscore_un) + np.finfo(np.float32).min * tf.cast(tf.logical_not(tf.cast(untreat_labels, tf.bool)), tf.float64) 
    
    s_tre = tf.nn.softmax(masked_h_rscore_tr, axis=0) ## softmax across the entire population 
    s_unt = tf.nn.softmax(masked_h_rscore_un, axis=0) ## softmax across the entire population 
    
    s_tre = tf.reshape(s_tre, (-1, num_weeks, )) 
    s_unt = tf.reshape(s_unt, (-1, num_weeks, ))         
    
    cost_labels = tf.multiply(cost_labels, exist_labels)
    trips_labels = tf.multiply(trips_labels, exist_labels) 
    gb_labels = tf.multiply(gb_labels, exist_labels) 
    
    dc_tre = tf.reduce_sum(tf.multiply(s_tre, cost_labels)) 
    dc_unt = tf.reduce_sum(tf.multiply(s_unt, cost_labels))  
    
    dt_tre = tf.reduce_sum(tf.multiply(s_tre, trips_labels)) 
    dt_unt = tf.reduce_sum(tf.multiply(s_unt, trips_labels)) 
    
    dg_tre = tf.reduce_sum(tf.multiply(s_tre, gb_labels)) 
    dg_unt = tf.reduce_sum(tf.multiply(s_unt, gb_labels)) 
    
    vars   = tf.trainable_variables() 
    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * l2_reg_weight 
    
    ### implement the CPIT/CPIGB objectives 
    cpitobj = tf.divide(dc_tre - dc_unt, dt_tre - dt_unt) + lossL2 
    cpigbobj = tf.divide(dc_tre - dc_unt, dg_tre - dg_unt) + lossL2 
    
    with tf.variable_scope("cpitoptimizer" + idstr) as scope: 
        cpitoptimizer = tf.train.AdamOptimizer( 
                learning_rate=adam_learning_rate, 
                beta1=adam_beta1, 
                beta2=adam_beta2) 
        """ 
        cpitoptimizer = tf.train.MomentumOptimizer( 
                learning_rate=adam_learning_rate, 
                momentum=adam_beta1, 
                ) 
        """
        gvs = cpitoptimizer.compute_gradients(cpitobj) 
        clipped_gvs = [(tf.clip_by_norm(gradient, gradient_clip_norm_value), var) for gradient, var in gvs] 
        cpitopt = cpitoptimizer.apply_gradients(clipped_gvs) 
    
    with tf.variable_scope("cpigboptimizer" + idstr) as scope: 
        cpigboptimizer = tf.train.AdamOptimizer(
                learning_rate=adam_learning_rate,
                beta1=adam_beta1,
                beta2=adam_beta2)
        gradients, variables = zip(*cpigboptimizer.compute_gradients(cpigbobj)) 
        gradients = [
            None if gradient is None else tf.clip_by_norm(gradient, gradient_clip_norm_value) 
            for gradient in gradients]
        cpigbopt = cpigboptimizer.apply_gradients(zip(gradients, variables))
    
    return cpitobj, cpitopt, cpigbobj, cpigbopt, h_rscore 

def HetSequentialCausalModel(num_units, features, seq_lens, treat_labels, trips_labels, cost_labels, gb_labels, exist_labels, idstr): 
    ### Defines the Heterogeneous Seqential Causal Model, builds tensorflow graph 
    ### 
    ### Inputs: 
    ### features, 3D tensor of shape [dataset_size, num_weeks, feature_dims] including the treatment labels 
    ###           num_weeks = 19/11; feature_dims = 18 with user features  
    ### seq_lens, 2D tensor of shape [dataset_size, 1] indicating length of sequence for each user 
    ### treat_labels, 2D tensor of shape [dataset_size, 1] indicating treatment for user in XP week 
    ### trips_labels, 2D tensor of shapexb [dataset_size, 1], trips number for user in XP week 
    ### cost_labels, 2D tensor of shape [dataset_size, 1], cost (-ve net-inflow/variable contribution) for users in XP week 
    ### gb_labels, 2D tensor of shape [dataset_size, 1], gross-bookings for users in XP week 
    ### exist_labels, 2D tensor of shape [dataset_size, 1] indicating whether data exists for the user in that XP week 
    ### idstr: a string to indicate whether function is used for train/val, avoid tensorflor parameter re-use 
    
    ### Returns: 
    ### cpitobj, cpit objective node 
    ### cpitopt, cpit optimizer node 
    ### cpigbobj, cpigb objective node 
    ### cpigbopt, cpigb optimizer node 
    
    ### recurrent model architecture 
    ### e.g. 3 layer each layer 8 units will be [8, 8, 8]         
    
    ### optimizer parameters 
    adam_learning_rate = 0.01 
    adam_beta1 = 0.5 
    adam_beta2 = 0.999 
    gradient_clip_norm_value = 2.3 
    
    ### optional (rarely used) l2 regularization 
    l2_reg_weight = 0.000 
    
    num_weeks = features.shape[1] 
    ### propagate forward the data through the LSTM and through a ranker 
    with tf.variable_scope("hscls_model") as scope: 
        cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n, reuse=tf.AUTO_REUSE) for n in num_units] 
        stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)         
        outputs, state = tf.nn.dynamic_rnn(stacked_rnn_cell, 
                                           inputs=features, 
                                           sequence_length=seq_lens, 
                                           dtype = np.float64 
                                           ) 
        h_rscore = tf.contrib.layers.fully_connected(outputs, 1, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
    
    ### branch out scores for treatment and control 
    h_rscore_tr = tf.reshape(h_rscore, (-1, num_weeks)) 
    h_rscore_un = tf.reshape(h_rscore, (-1, num_weeks)) 
    
    ### now, mask the score with the treatment label for treatment cohort and control cohort 
    untreat_labels = tf.cast(tf.multiply( 
            exist_labels, ## data exists for cost and trips 
            tf.cast(tf.logical_not(tf.cast(treat_labels, tf.bool)), tf.float64)), tf.float64) 
    masked_h_rscore_tr = tf.multiply(treat_labels, h_rscore_tr) + np.finfo(np.float32).min * tf.cast(tf.logical_not(tf.cast(treat_labels, tf.bool)), tf.float64)
    masked_h_rscore_un = tf.multiply(untreat_labels, h_rscore_un) + np.finfo(np.float32).min * tf.cast(tf.logical_not(tf.cast(untreat_labels, tf.bool)), tf.float64) 
    
    s_tre = tf.nn.softmax(masked_h_rscore_tr, axis=0) ## softmax across the entire population 
    s_unt = tf.nn.softmax(masked_h_rscore_un, axis=0) ## softmax across the entire population 
    
    s_tre = tf.reshape(s_tre, (-1, num_weeks, )) 
    s_unt = tf.reshape(s_unt, (-1, num_weeks, ))         
    
    cost_labels = tf.multiply(cost_labels, exist_labels)
    trips_labels = tf.multiply(trips_labels, exist_labels) 
    gb_labels = tf.multiply(gb_labels, exist_labels) 
    
    dc_tre = tf.reduce_sum(tf.multiply(s_tre, cost_labels)) 
    dc_unt = tf.reduce_sum(tf.multiply(s_unt, cost_labels))  
    
    dt_tre = tf.reduce_sum(tf.multiply(s_tre, trips_labels)) 
    dt_unt = tf.reduce_sum(tf.multiply(s_unt, trips_labels)) 
    
    dg_tre = tf.reduce_sum(tf.multiply(s_tre, gb_labels)) 
    dg_unt = tf.reduce_sum(tf.multiply(s_unt, gb_labels)) 
    
    vars   = tf.trainable_variables() 
    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * l2_reg_weight 
    
    ### implement the CPIT/CPIGB objectives 
    cpitobj = tf.divide(dc_tre - dc_unt, dt_tre - dt_unt) + lossL2 
    cpigbobj = tf.divide(dc_tre - dc_unt, dg_tre - dg_unt) + lossL2 
    
    with tf.variable_scope("cpitoptimizer" + idstr) as scope: 
        cpitoptimizer = tf.train.AdamOptimizer( 
                learning_rate=adam_learning_rate, 
                beta1=adam_beta1, 
                beta2=adam_beta2) 
        """ 
        cpitoptimizer = tf.train.MomentumOptimizer( 
                learning_rate=adam_learning_rate, 
                momentum=adam_beta1, 
                ) 
        """
        gvs = cpitoptimizer.compute_gradients(cpitobj) 
        clipped_gvs = [(tf.clip_by_norm(gradient, gradient_clip_norm_value), var) for gradient, var in gvs] 
        cpitopt = cpitoptimizer.apply_gradients(clipped_gvs) 
    
    with tf.variable_scope("cpigboptimizer" + idstr) as scope: 
        cpigboptimizer = tf.train.AdamOptimizer(
                learning_rate=adam_learning_rate,
                beta1=adam_beta1,
                beta2=adam_beta2)
        gradients, variables = zip(*cpigboptimizer.compute_gradients(cpigbobj)) 
        gradients = [
            None if gradient is None else tf.clip_by_norm(gradient, gradient_clip_norm_value) 
            for gradient in gradients]
        cpigbopt = cpigboptimizer.apply_gradients(zip(gradients, variables))
    
    return cpitobj, cpitopt, cpigbobj, cpigbopt, h_rscore 


def TopPRankingModel(D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, p_quantile, idstr):     
    ## older version of TQR model without tunable temperature 
    ## and without temperature schedules 
    
    ## implements the top-p-quantile fall-off at ranker training 
    ## D_tre: user context features for treatment 
    ## D_unt: user context features for control (untreated) 
    ## o_tre: user next week order labels for treatment 
    ## o_unt: user next week order labels for control (untreated) 
    ## c_tre: user next week cost(negative net-inflow/variable contribution) labels for treatment 
    ## c_unt: user next week cost(negative net-inflow/variable contribution) labels for control (untreated)  
    ## p_quantile: the top-p-quantile number between (0, 1) 
    ## idstr: a string to indicate whether function is used for train/val, avoid tensorflor parameter re-use 
    
    ## returns: 
    ## obj, objective node 
    ## opt, optimizer node 
    ## h_tre_rnkscore, ranker scores for treated users 
    ## h_unt_rnkscore, ranker scores for control users 
    
    ## temperature of the sigmoid governing p_quantile cut-off 
    temp = 2.5 
    
    ## define size of cohort datasets 
    size_tre = D_tre.shape[0] 
    size_unt = D_unt.shape[0] 
    
    ### define model graph of fixed quantile ranking 
    with tf.variable_scope("toppranker") as scope: 
        h_tre_rnkscore = tf.contrib.layers.fully_connected(D_tre, 1, activation_fn=None, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
        h_unt_rnkscore = tf.contrib.layers.fully_connected(D_unt, 1, activation_fn=None, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
    
    h_tre_sorted = tf.contrib.framework.sort(h_tre_rnkscore, axis=0, direction='DESCENDING') 
    h_unt_sorted = tf.contrib.framework.sort(h_unt_rnkscore, axis=0, direction='DESCENDING') 
    
    top_k_tre = tf.cast(tf.ceil(size_tre * p_quantile), tf.int32) 
    top_k_unt = tf.cast(tf.ceil(size_unt * p_quantile), tf.int32) 
    
    intercept_tre = tf.slice(h_tre_sorted, [top_k_tre - 1, 0], [1, 1]) 
    intercept_unt = tf.slice(h_unt_sorted, [top_k_unt - 1, 0], [1, 1]) 
    
    #intercept_tre = tf.stop_gradient(intercept_tre) 
    #intercept_unt = tf.stop_gradient(intercept_unt) 
    
    ### use sigmoid to threshold top-k candidates, or use more sophisticated hinge loss 
    h_tre = tf.sigmoid(temp * (h_tre_rnkscore - intercept_tre)) 
    h_unt = tf.sigmoid(temp * (h_unt_rnkscore - intercept_unt)) 
    
    s_tre = tf.nn.softmax(h_tre, axis=0) 
    s_unt = tf.nn.softmax(h_unt, axis=0) 
    
    s_tre = tf.reshape(s_tre, (size_tre, ))
    s_unt = tf.reshape(s_unt, (size_unt, ))
    
    dc_tre = tf.reduce_sum(tf.multiply(s_tre, c_tre)) 
    dc_unt = tf.reduce_sum(tf.multiply(s_unt, c_unt)) 
    
    do_mult_tre = tf.multiply(s_tre, o_tre) 
    do_mult_unt = tf.multiply(s_unt, o_unt) 
    
    do_tre = tf.reduce_sum(do_mult_tre) 
    do_unt = tf.reduce_sum(do_mult_unt) 
    
    ### implement the CPIT objective 
    obj = tf.divide(dc_tre - dc_unt, do_tre - do_unt) 
    
    with tf.variable_scope("optimizer" + idstr) as scope: 
        opt = tf.train.AdamOptimizer().minimize(obj) 
    
    return obj, opt, h_tre_rnkscore, h_unt_rnkscore 

def nnReg(D, l, num_hidden, C, regtype = 'l2', lr = 0.01, beta1 = 0.5): 
    ## implements a sparse neural net logistic regression with tensorflow 
    #D = tf.contrib.layers.dense_to_sparse(D) 
    with tf.variable_scope("nnreg-l1") as scope: 
        h1 = tf.contrib.layers.fully_connected(D, num_hidden, activation_fn=tf.sigmoid, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
    with tf.variable_scope("nnreg-l2") as scope: 
        h2 = tf.contrib.layers.fully_connected(h1, 1, activation_fn=None, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
    
    loss = 1.0 / len(l) * tf.reduce_sum(tf.square(h2 - l)) 
    
    if regtype == 'l1': 
        l1_regularizer = tf.contrib.layers.l1_regularizer(
            scale=C, scope=None
        )
        weights = tf.trainable_variables() # all vars of your graph
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
    elif regtype == 'l2': 
        l2_regularizer = tf.contrib.layers.l2_regularizer(
            scale=C, scope=None
        )
        weights = tf.trainable_variables() # all vars of your graph
        regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)        
    else: 
        regularization_penalty = 0.0 
    
    regularized_loss = loss + regularization_penalty # this loss needs to be minimized
    opt = tf.train.AdamOptimizer(learning_rate = lr, beta1 = beta1).minimize(regularized_loss) 
    
    return regularized_loss, opt 

def train_nnReg(D, l, num_hidden, C, regtype = 'l2', lr = 0.01, beta1 = 0.5, num_iters = 100): 
    loss, opt = nnReg(D, l, num_hidden, C, regtype, lr, beta1) 
    sess = tf.Session() 
    init = tf.global_variables_initializer() 
    sess.run(init) 
    
    for it in range(num_iters): 
        l, _ = sess.run([loss, opt]) 
        if it % 1 == 0: 
            print('step: ' + str(it) + 'loss: ' + str(l)) 

#### Appendix 
"""
def TunableTQRankingModel(graph, D_tre, D_unt, o_tre, o_unt, c_tre, c_unt, idstr, initial_temp, p_quantile): 
    ## implements the top-p-quantile fall-off at ranker training 
    ## D_tre: user context features for treatment 
    ## D_unt: user context features for control (untreated) 
    ## o_tre: user next week order labels for treatment 
    ## o_unt: user next week order labels for control (untreated) 
    ## c_tre: user next week cost(negative net-inflow/variable contribution) labels for treatment 
    ## c_unt: user next week cost(negative net-inflow/variable contribution) labels for control (untreated)  
    
    ## p_quantile: the top-p-quantile number between (0, 1) 
    ## idstr: a string to indicate whether function is used for train/val, avoid tensorflor parameter re-use 
    
    ## returns: 
    ## obj, objective node 
    ## opt, optimizer node 
    ## h_tre_rnkscore, ranker scores for treated users 
    ## h_unt_rnkscore, ranker scores for control users 
    
    ## temperature of the sigmoid governing p_quantile cut-off 
    #temp = 2.5 
    with graph.as_default() as g: 
        init = tf.constant(initial_temp, dtype=tf.float64)
        with tf.variable_scope("temp", reuse=tf.AUTO_REUSE) as scope: 
            temp = tf.get_variable('temp', initializer=init, dtype=tf.float64) 
            #tf.Variable(2.5, dtype=tf.float64, trainable=True) 
        init2 = tf.constant(p_quantile, dtype=tf.float64) 
        with tf.variable_scope("p_quantile", reuse=tf.AUTO_REUSE) as scope: 
            p_quantile = tf.get_variable('p_quantile', initializer=init2, dtype=tf.float64)
            #tf.Variable(0.3, dtype=tf.float32, trainable=True, reuse=tf.AUTO_REUSE)
        
        ## define size of cohort datasets 
        size_tre = D_tre.shape[0] 
        size_unt = D_unt.shape[0] 
        
        ### define model graph of fixed quantile ranking 
        with tf.variable_scope("tqranker") as scope: 
            h_tre_rnkscore = tf.contrib.layers.fully_connected(D_tre, 1, activation_fn=None, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            h_unt_rnkscore = tf.contrib.layers.fully_connected(D_unt, 1, activation_fn=None, reuse=True, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 

        h_tre_sorted = tf.contrib.framework.sort(h_tre_rnkscore, axis=0, direction='DESCENDING') 
        h_unt_sorted = tf.contrib.framework.sort(h_unt_rnkscore, axis=0, direction='DESCENDING') 
        
        top_k_tre = tf.cast(tf.ceil(size_tre * p_quantile), tf.int32) 
        top_k_unt = tf.cast(tf.ceil(size_unt * p_quantile), tf.int32) 
        
        intercept_tre = tf.slice(h_tre_sorted, [top_k_tre - 1, 0], [1, 1]) 
        intercept_unt = tf.slice(h_unt_sorted, [top_k_unt - 1, 0], [1, 1]) 
        
        intercept_tre = tf.stop_gradient(intercept_tre) 
        intercept_unt = tf.stop_gradient(intercept_unt) 
        
        ### use sigmoid to threshold top-k candidates, or use more sophisticated hinge loss 
        h_tre = tf.sigmoid(temp * (h_tre_rnkscore - intercept_tre)) 
        h_unt = tf.sigmoid(temp * (h_unt_rnkscore - intercept_unt)) 
        
        s_tre = tf.nn.softmax(h_tre, axis=0) 
        s_unt = tf.nn.softmax(h_unt, axis=0) 
        
        s_tre = tf.reshape(s_tre, (size_tre, ))
        s_unt = tf.reshape(s_unt, (size_unt, ))
        
        dc_tre = tf.reduce_sum(tf.multiply(s_tre, c_tre)) 
        dc_unt = tf.reduce_sum(tf.multiply(s_unt, c_unt)) 
        
        do_mult_tre = tf.multiply(s_tre, o_tre) 
        do_mult_unt = tf.multiply(s_unt, o_unt) 
        
        do_tre = tf.reduce_sum(do_mult_tre) 
        do_unt = tf.reduce_sum(do_mult_unt) 
        
        ### implement the CPIT objective 
        #obj = tf.divide(dc_tre - dc_unt, do_tre - do_unt) 
        obj = tf.math.log(dc_tre - dc_unt) - tf.math.log(do_tre - do_unt) 
        
        with tf.variable_scope("optimizer" + idstr) as scope: 
            opt = tf.train.AdamOptimizer().minimize(obj) 
    
    return obj, opt, h_tre_rnkscore, h_unt_rnkscore, temp, p_quantile 
"""


def OnOffCTPMDNN(graph, D, o, c, iint, idstr, num_hidden): 
    
    ## implements the Direct Ranking Model based on CPIT 
    ## D_tre: user context features for treatment 
    ## D_unt: user context features for control (untreated) 
    ## o_tre: user next week order labels for treatment 
    ## o_unt: user next week order labels for control (untreated) 
    ## c_tre: user next week cost(negative net-inflow/variable contribution) labels for treatment 
    ## c_unt: user next week cost(negative net-inflow/variable contribution) labels for control (untreated) 
    ## i_tre: treatment intensity of treatment group 
    ## i_unt: treatment intensity of control group 
    ## idstr: a string to indicate whether function is used for train/val, avoid tensorflor parameter re-use 
    
    ## returns: 
    ## obj, objective node 
    ## opt, optimizer node 
    ## h_tre_rnkscore, ranker scores for treated users 
    ## h_unt_rnkscore, ranker scores for control users 
    
    ## define size of cohort datasets 
    size_tre = D.shape[0] 
    
    with graph.as_default() as g: 
        ### ------ define model graph of direct ranking ------ 
        
        ### define ranker/scorer with one or more layers 
        if num_hidden > 0: 
            with tf.variable_scope("ctpmmatcherhidden") as scope: 
                h_tre_matchhidden = tf.contrib.layers.fully_connected(D, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("ctpmmatcher") as scope: 
                h_tre_matchscore = tf.contrib.layers.fully_connected(h_tre_matchhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
        else: 
            with tf.variable_scope("ctpmmatcher") as scope: 
                h_tre_matchscore = tf.contrib.layers.fully_connected(D, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
        
        if num_hidden > 0: 
            with tf.variable_scope("ctpmpolicysighidden") as scope: 
                h_tre_policyhidden = tf.contrib.layers.fully_connected(D, num_hidden, activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope=scope, weights_initializer=tf.contrib.layers.xavier_initializer()) 
            with tf.variable_scope("ctpmpolicysig") as scope: 
                h_tre_policyscore = tf.contrib.layers.fully_connected(h_tre_policyhidden, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
        else: 
            with tf.variable_scope("ctpmpolicysig") as scope: 
                h_tre_policyscore = tf.contrib.layers.fully_connected(D, 1, activation_fn=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, scope=scope) 
        
        ## use the bell-shape cost function in treatment intensity 
        diff_tre = np.reshape(iint, (-1, 1)) - h_tre_policyscore 
        
        lh_tre_policyscore = tf.math.multiply(tf.math.sigmoid(diff_tre), (1 - tf.math.sigmoid(diff_tre))) 
        
        ## this is the un-normalized bayesian weighting score 
        s_tre_unnorm = tf.math.multiply(h_tre_matchscore, lh_tre_policyscore) 
        
        s_tre_partfunc = tf.reduce_sum(s_tre_unnorm) + 1e-17
        
        ## this is the normalized bayesian weighting score 
        s_tre = tf.math.divide(s_tre_unnorm, s_tre_partfunc) 
        
        s_on = tf.reshape(s_tre, (size_tre, )) 
        s_off = np.ones((size_tre, )) * 1.0 / size_tre 
        
        dc_on = tf.reduce_sum(tf.multiply(s_on, c)) 
        dc_off = tf.reduce_sum(tf.multiply(s_off, c)) ## replace this later with entire dataset 
        
        do_mult_on = tf.multiply(s_on, o) 
        do_mult_off = tf.multiply(s_off, o) 
        
        do_on = tf.reduce_sum(do_mult_on) 
        do_off = tf.reduce_sum(do_mult_off) 
        
        ### implement the cost-gain effectiveness objective 
        #obj = tf.divide(dc_on - dc_off, do_on - do_off) 
        obj = tf.divide(tf.nn.relu(dc_on - dc_off), tf.nn.relu(do_on - do_off)) 
        
        ### for application/production purposes, the above equation is more interpretable 
        ### and works well for most datasets, below is 
        ### an option to use relu, and math.log to ensure the objective is differentiable         
        #obj = tf.divide(tf.nn.leaky_relu(dc_tre - dc_unt), tf.nn.leaky_relu(do_tre - do_unt)) 
        #obj = tf.subtract(tf.math.log(tf.nn.leaky_relu(dc_tre - dc_unt)), tf.math.log(tf.nn.leaky_relu(do_tre - do_unt))) 
        
        with tf.variable_scope("optimizer" + idstr) as scope: 
            opt = tf.train.AdamOptimizer().minimize(obj) 
        
        return obj, opt, s_on, s_off #h_tre_rnkscore, h_unt_rnkscore 
