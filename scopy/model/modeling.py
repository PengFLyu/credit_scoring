import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from math import *

class Modeling:
    """
    The modeling process
    """
    _logit_mdl = None
    
    def __init__(self, df, target):
        self.df = df
        self.target = target
    
    def fit(self):
        """
        """
        X = self.df.drop(self.target, axis = 1)
        X['constant'] = 1
        y = self.df[self.target]
        
        self._logit_mdl = sm.Logit(y, X).fit()
        
        return self._logit_mdl
    
    
    def scoring(self, df, target, P = 660, PDO = 50, theta = 1/20):
        """
        Transformation from probabilities to standard scoring.
        """ 
        dfc = df.copy()
        
        X = dfc.drop(target, axis = 1)
        X['constant'] = 1
        dfc['pred_Y'] = self._logit_mdl.predict(X)
    
        B = PDO / log(2, e)
        A = P + B * log(theta, e)
    
        dfc['odds'] = np.log((dfc['pred_Y'] / (1 - dfc['pred_Y'])))
        dfc['score'] = round(A - B * dfc['odds'], 0)
    
        return dfc

    
    def KS(self, df, target, bin_num = 20):
        """
        """
        dfc = df.copy()
        
        dfc['pred_Y'] = self._logit_mdl.predict(dfc.drop(target, axis = 1))
        
        dfc = dfc.sort_values('pred_Y', ascending = False)
        dfc['group'] = [ceil(x / (dfc.shape[0] / bin_num)) for x in range(1, dfc.shape[0] + 1)]

        grouped = dfc.groupby('group')[target].agg([np.size,np.mean,np.sum]).rename(columns={'size':'Total_cnt', 
                                                                                             'mean':'Y_rate', 
                                                                                             'sum':'Y_cnt'})
        grouped['nY_cnt'] = grouped['Total_cnt'] - grouped['Y_cnt']
        grouped['Y_pct'] = (grouped['Y_cnt'] / grouped['Y_cnt'].sum())
        grouped['nY_pct'] = (grouped['nY_cnt'] / grouped['nY_cnt'].sum())
        
        grouped['Cum_Y_pct'] = grouped['Y_pct'].cumsum()
        grouped['Cum_nY_pct'] = grouped['nY_pct'].cumsum()
        
        grouped['KS'] = np.absolute(grouped['Cum_Y_pct'] - grouped['Cum_nY_pct'])

        KS = grouped['KS'].max()

        return (KS, grouped)
    
    def AUC(self, df, target):
        """
        """        
        return roc_auc_score(df[target], self._logit_mdl.predict(df.drop(target, axis = 1)))