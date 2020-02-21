from __future__ import print_function

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor 

class RLearner:
    """
    R-learner, estimate the heterogeneous causal effect
    replicate paper: https://arxiv.org/pdf/1712.04912.pdf
    Github R reference: https://github.com/xnie/rlearner
    """    
    
    def __init__( 
            self, 
            p_model_specs=None, 
            # ToDo: change the default to Ridge regression as OLS will have explosive coefficients
            m_model_specs={'model': linear_model.Ridge, 'params': {'alpha': 1.0}},
            tau_model_specs={'model': linear_model.Ridge, 'params': {'alpha': 1.0}},
            shadow=None,
            k_fold=5,
    ):
        """
        Constructor
        :param p_model_specs: a dictionary of model and hyper-params, specification for the model of E[W|X],
        propensity of the sample in treatment, if None, assume perfect randomized experiment and will use a constant p
        calculated from is_treatment from y
        :param m_model_specs: specification for the model of E[Y|X], example args
        {'model': linear_model.Ridge, 'params': {'alpha': 1.0}}
        :param tau_model_specs: specification for the model of E[Y(1) - Y(0)|X]
        :param shadow: shadow scale for objective cost - shadow * value
        :param k_fold: number of folds to use k-fold to predict p_hat and m_hat
        """
        
        self.p_model_specs = p_model_specs
        self.m_model_specs = m_model_specs
        self.tau_model_specs = tau_model_specs
        
        # self.p_model = None
        # self.m_model = None
        self.tau_model = None
        
        self.shadow = shadow
        self.k_fold = k_fold
    
    def _fit_predict_p_hat(self, X, w): 
        """ 
        Fit and predict for p_hat 
        :param X: feature matrix 
        :param w: binary indicator for treatment / control 
        :return: a numpy array of predicted p_hat, same shape as w 
        """ 
        
        if self.p_model_specs is None: 
            return np.sum(w) / float(len(w)) * np.ones_like(w)
        
        kf = KFold(n_splits=self.k_fold)
        
        p_hat = np.zeros_like(w)
        
        # initialize m model 
        p_model = self.p_model_specs['model'](**self.p_model_specs['params']) 
        
        for fit_idx, pred_idx in kf.split(X):
            
            # split data into fit and predict
            X_fit, X_pred = X[fit_idx], X[pred_idx]
            w_fit = w[fit_idx]
                        
            p_model.fit(X_fit, w_fit)
            p_hat[pred_idx] = p_model.predict(X=X_pred)
        
        p_hat = np.clip(p_hat, 0 + 1e-7, 1 - 1e-7) 
        
        return p_hat 
    
    def _fit_predict_m_hat(self, X, m):
        """
        Fit and predict for m_hat
        :param X: feature matrix
        :param m: cost - shadow * value
        :return: a numpy array of predicted m_hat, same shape as m
        """
        # ToDo: add hyper-param tuning for m_hat model here
        
        kf = KFold(n_splits=self.k_fold)
        
        m_hat = np.zeros_like(m)
        
        # initialize m model
        self.m_model = self.m_model_specs['model'](**self.m_model_specs['params'])
        
        for fit_idx, pred_idx in kf.split(X):
            
            # split data into fit and predict
            X_fit, X_pred = X[fit_idx], X[pred_idx]
            m_fit = m[fit_idx]
            
            self.m_model.fit(X_fit, m_fit)
            m_hat[pred_idx] = self.m_model.predict(X=X_pred)
                
        return m_hat
    
    def fit(self, X, y, lambd=0.0, sample_weight=None, **kwargs):
        """
        Fit
        :param X: feature matrix
        :param y: label array with columns [value, cost, is_treatment]
        :param sample_weight:
        :return: None
        """
        
        self.lambd = lambd
        y_ = y[:, 0] - self.lambd * y[:, 1] 
        w = y[:, -1] 
        
        p_hat = self._fit_predict_p_hat(X, w)
        m_hat = self._fit_predict_m_hat(X, y_)
        
        r_pseudo_y = (y_ - m_hat) / (w - p_hat)
        r_weight = np.square(w - p_hat)
        
        self.tau_model = self.tau_model_specs['model'](**self.tau_model_specs['params'])
        
        # fit tau model with pseudo y and sample weights 
        self.tau_model.fit(X, r_pseudo_y) 
        #self.tau_model.fit(X, r_pseudo_y, sample_weight=r_weight) 
        
        return None
    
    def predict(self, X, **kwargs):
        """
        Predict
        :param X: feature matrix
        :return: - predicted tau, aka -(cost - shadow * value)
        """
        return self.tau_model.predict(X)
    
    def get_params(self):
        """
        :return: dictionary of hyper-parameters of the model.
        """
        return {
            'p_model_specs': self.p_model_specs,
            'm_model_specs': self.m_model_specs,
            'tau_model_specs': self.tau_model_specs,
            'shadow': self.shadow,
            'k_fold': self.k_fold,
        }

    @property
    def coef_(self):
        """
        Estimated coefficients for tau model
        :return: array, shape (n_features, )
        """
        return self.tau_model.coef_
