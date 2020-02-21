from sklearn.ensemble import RandomForestRegressor 
from rlearner import RLearner 
import numpy as np 
import matplotlib.pyplot as plt 

class LinearHTEModels: 
    
    def fit_rf (self, X_tr, y_tr): 
        ## the standard random forest for 
        ## regressing X_tr to target y_tr 
        
        ## model parameters exactly the same with
        ## previous promotion experiments 
        
        ##   X_tr: training data each row is one data vector 
        ##   Y_tr: labels (rewards or value), vertical vector 
        ##   return: no return value 
        
        self.rfmodel = RandomForestRegressor(
            max_depth=6, 
            random_state=0, 
            n_estimators=100, 
            min_samples_leaf=100, 
            min_samples_split=10) 
        print(self.rfmodel)
        self.rfmodel.fit(X_tr, y_tr) 
        return self.rfmodel 
    
    def predict_rf(self, X): 
        ## prediction for random forest 
        return self.rfmodel.predict(X) 
        
    def fit_slearner(self, X_tr, Y_tr, w_tr): 
        ## fit the s-learner model using 
        ## one regressor to model the treatment group 
        ## then use another regressor to model non-treatment group 
        
        ##   X_tr: training data each row is one data vector 
        ##   Y_tr: labels (rewards or value), vertical vector 
        ##   w_tr: treatment label {1, 0}, vertical vector 
        ## return: fitted model objects 
        
        self.slearnerPmodel = RandomForestRegressor(
            max_depth=6, 
            random_state=0, 
            n_estimators=100, 
            min_samples_leaf=100, 
            min_samples_split=10) 
        
        self.slearnerNmodel = RandomForestRegressor(
            max_depth=6, 
            random_state=0, 
            n_estimators=100, 
            min_samples_leaf=100, 
            min_samples_split=10) 
        
        pfilter = (w_tr > 0.5) 
        nfilter = (w_tr <= 0.5) 
        X_tr_P = X_tr[pfilter, :] 
        Y_tr_P = Y_tr[pfilter] 
        
        X_tr_N = X_tr[nfilter, :] 
        Y_tr_N = Y_tr[nfilter] 
        
        self.slearnerPmodel.fit(X_tr_P, Y_tr_P) 
        self.slearnerNmodel.fit(X_tr_N, Y_tr_N) 
        
        return self.slearnerPmodel, self.slearnerNmodel 
    
    def predict_slearner(self, X_va): 
        ## predictor for s-learner 
        ## one model for regressing labels in treatment group 
        ## another for regressing labels in non-treatment group 
        ##   X_va: validation/test data for prediction 
        ##   returns: difference across the regressed results 
        
        rP = self.slearnerPmodel.predict(X_va) 
        rN = self.slearnerNmodel.predict(X_va) 
        return (rP - rN), rN
    
    def fit_rlearner(self, X_tr, O_tr, C_tr, w_tr, m_model_specs = '', tau_model_specs = '', p_model_specs = ''): 
        
        ## fit the r-learner model 
        ## model from paper: "Quasi-Oracle Estimation of Heterogeneous Treatment Effects" 
        ## "https://arxiv.org/pdf/1712.04912.pdf" 
        ## code sources: 
        ## "https://code.uberinternal.com/diffusion/MASAP/browse/master/sapphire_optimization/models/targeting_models/rlearner/core/rlearner.py" 
        ## "https://github.com/xnie/rlearner" 
        ##   X_tr: training data, each row is a data vector 
        ##   O_tr: labels (rewards or value), vertical vector  
        ##   C_tr: cost labels, vertical vector 
        ##   w_tr: treatment labels {1, 0}, vertical vector 
        ## return: fitted model object 
        
        ## process values_tr, zero cost vector and w_tr into rlearner input 
        if m_model_specs == '' and tau_model_specs == '' and p_model_specs == '': 
            self.rlearnermodel_O = RLearner() 
            self.rlearnermodel_C = RLearner() 
        else: 
            self.rlearnermodel_O = RLearner(m_model_specs=m_model_specs, tau_model_specs=tau_model_specs, p_model_specs=p_model_specs) 
            self.rlearnermodel_C = RLearner(m_model_specs=m_model_specs, tau_model_specs=tau_model_specs, p_model_specs=p_model_specs) 
        
        z = np.zeros([len(O_tr), 1]) 
        
        o = np.concatenate((np.reshape(O_tr, [-1, 1]), z), axis=1) 
        o = np.concatenate((o, np.reshape(w_tr, [-1, 1])), axis=1) 
        
        c = np.concatenate((np.reshape(C_tr, [-1, 1]), z), axis=1)
        c = np.concatenate((c, np.reshape(w_tr, [-1, 1])), axis=1) 
        
        self.rlearnermodel_O.fit(X_tr, o) 
        self.rlearnermodel_C.fit(X_tr, c) 
        
        return [self.rlearnermodel_O.tau_model, self.rlearnermodel_C.tau_model] 
    
    def fit_rlearner_lagrangian(self, X_tr, O_tr, C_tr, w_tr, lambd, m_model_specs = '', tau_model_specs = '', p_model_specs = ''): 
        
        ## fit the r-learner model 
        ## model from paper: "Quasi-Oracle Estimation of Heterogeneous Treatment Effects" 
        ## "https://arxiv.org/pdf/1712.04912.pdf" 
        ## code sources: 
        ## "https://code.uberinternal.com/diffusion/MASAP/browse/master/sapphire_optimization/models/targeting_models/rlearner/core/rlearner.py" 
        ## "https://github.com/xnie/rlearner" 
        ##   X_tr: training data, each row is a data vector 
        ##   O_tr: labels (rewards or value), vertical vector  
        ##   C_tr: cost labels, vertical vector 
        ##   w_tr: treatment labels {1, 0}, vertical vector 
        ## return: fitted model object 
        
        ## process values_tr, zero cost vector and w_tr into rlearner input 
        if m_model_specs == '' and tau_model_specs == '' and p_model_specs == '': 
            self.rlearnermodel_L = RLearner() 
        else: 
            self.rlearnermodel_L = RLearner(m_model_specs=m_model_specs, tau_model_specs=tau_model_specs, p_model_specs=p_model_specs)         
        
        y = np.concatenate((np.reshape(O_tr, [-1, 1]), np.reshape(C_tr, [-1, 1])), axis=1)
        y = np.concatenate((y, np.reshape(w_tr, [-1, 1])), axis=1) 
        
        self.rlearnermodel_L.fit(X_tr, y, lambd) 
        
        return self.rlearnermodel_L.tau_model
    
    def predict_rlearner(self, X_va): 
        ## predicts with r-learner model 
        ##   X_va: validation or test data [num_sampes, num_features] 
        ## return: list of regressor prediction values  
        ## note: rlearner returns the negative of the tau_model 
        ## it defines cost as 'Y', so 
        # Y ~ - value 
        # tau predicts E[Y(1) - Y(0)] 
        # - tau predicts E[Y(0) - Y(1)] 
        # - ( - tao) predicts E[-value(0) + value (1)] = E[ value(1) - value (0)]
        
        return self.rlearnermodel.predict(X_va) 
    
    ## back-log functions/snippets: older ones that are no-longer useful 

    def visualize_importances(self, rmodel, feature_dims): 
        importances = rmodel.feature_importances_
        std = np.std([tree.feature_importances_ for tree in rmodel.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        # Print the feature ranking
        print("Feature ranking:")
        for f in range(feature_dims):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(feature_dims), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(feature_dims), indices)
        plt.xlim([-1, feature_dims])
        plt.show() 
