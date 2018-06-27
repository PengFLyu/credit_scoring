import pandas as pd
from sklearn import tree
from math import *


class VarPreprocess:
    """
    Class for variable preprocessing.
    """
    
    # Some default values
    _bin_num = 10
    _max_depth = 3
    _min_samples_leaf = 1
    
    def __init__(self, df, keep_list, target):
        self.df = df
        self.keep_list = keep_list
        self.target = target
        
    def _numiv_eq_freq(self, col_series):
        """
        Rough IV calculation for numerical columns: Equal frequency binning.
        """
        temp_df = pd.DataFrame()
        
        temp_df['var'] = col_series
        temp_df['target'] = self.df[self.target]
    
        bin_index = dict(((temp_df['var'].value_counts().sort_index().cumsum()/temp_df.shape[0])*self._bin_num).apply(lambda x: ceil(x)))
        temp_df['bin_index'] = temp_df['var'].apply(lambda x: bin_index[x])
       
        return groupedresult(temp_df, 'bin_index', 'target')[1]


    def _numiv_tree(self, col_series):
        """
        Rough IV calculation for numerical columns: Single variable decision tree.
        """
        temp_df = pd.DataFrame()
    
        temp_df['var'] = col_series
        temp_df['target'] = self.df[self.target]

        treeclf = tree.DecisionTreeClassifier(max_depth = self._max_depth, 
                                              min_samples_leaf = self._min_samples_leaf).fit(temp_df.drop('target',1), self.df[self.target])
        temp_df['bin_index'] = treeclf.apply(temp_df.drop('target',1))
    
        return groupedresult(temp_df, 'bin_index', 'target')[1]


    def _cativ(self, col_series):
        """
        Default iv calculation for categorical columns.
        """
        temp_df = pd.DataFrame()
    
        temp_df['var'] = col_series
        temp_df['target'] = self.df[self.target]
    
        bins = groupedresult(temp_df, 'var', 'target')[0].sort_values('Y_rate')
        bins['bin_index'] = (bins['Total_pct'].cumsum() * self._bin_num).apply(lambda x: ceil(x))
        temp_df['bin_index'] = temp_df['var'].apply(lambda x: dict(bins['bin_index'])[x])
    
        return groupedresult(temp_df, 'bin_index', 'target')[1]
    

    def _varroughiv(self, df, keep_list, target, method = 'eq_freq', bin_num = 10, max_depth = 3, min_samples_leaf = 1):
        """
        Roughly calculate iv values for every varaible.
        """
        if method not in ['eq_freq', 'tree']:
            raise ValueError("%s method is currently not supported.") % method
            
        self._bin_num = bin_num
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
    
        num_data, obj_data = varTypeSplit(df, keep_list)
    
        num_data.fillna(-1, inplace = True)
        obj_data.fillna('Empty', inplace = True)
    
        objvar_iv = obj_data.drop(keep_list, 1).apply(self._cativ)
    
        if method == 'eq_freq':
            numvar_iv = num_data.drop(keep_list, 1).apply(self._numiv_eq_freq)
        else:
            numvar_iv = num_data.drop(keep_list, 1).apply(self._numiv_tree)
    
        return numvar_iv.append(objvar_iv)


    def var_statistics(self, if_iv = True, method = 'eq_freq', bin_num = 10, max_depth = 3, min_samples_leaf = 1):
        """
        Return a DataFrame with some column statistics.
        """
        var_states = pd.DataFrame()
        
        var_states['mis_rate'] = missingRate(self.df, self.keep_list)
        var_states['dom_rate'] = dominantRate(self.df, self.keep_list)
        var_states['objlevels'] = levels(self.df, self.keep_list)

        if if_iv:
            var_states['iv'] = self._varroughiv(self.df, self.keep_list, self.target, 
                                                method, bin_num, max_depth, min_samples_leaf)

        return var_states
    

    def var_filter(self, mis_cutoff = 0.9, dom_cutoff = 0.9, lev_cutoff = 40, iv_cutoff = 0.02, 
                         method = 'eq_freq', bin_num = 10, max_depth = 3, min_samples_leaf = 1):
        """
        Directly remove statistically insignificant variables
        
        Return: DataFrame with 
        """
        if iv_cutoff is None:
            stats = self.var_statistics(if_iv = False)
            drop_mask = (stats['mis_rate'] >= mis_cutoff) | \
                        (stats['dom_rate'] >= dom_cutoff) | \
                        (stats['objlevels'] >= lev_cutoff)
        else: 
            stats = self.var_statistics(if_iv = True, 
                                        method = method,
                                        bin_num = bin_num,
                                        max_depth = max_depth,
                                        min_samples_leaf = min_samples_leaf)
            drop_mask = (stats['mis_rate'] >= mis_cutoff) | \
                        (stats['dom_rate'] >= dom_cutoff) | \
                        (stats['objlevels'] >= lev_cutoff) | \
                        (stats['iv'] < iv_cutoff)
        
        drop_list = drop_mask[drop_mask == True].index.tolist()
        
        return self.df.drop(drop_list, axis = 1)