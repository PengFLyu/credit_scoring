import pandas as pd
import numpy as np

class VarBinning(VarPreprocess):
    """
    """
    
    def __init__(self, df, keep_list, target):
        self.df = df
        self.keep_list = keep_list
        self.target = target
        
    def binning(self, var, method = None, **kw):
        """
        """
        method_list = ['eq_freq', 'tree', 'manual']
        
        temp_df = pd.DataFrame()
        
        temp_df['var'] = self.df[var]
        temp_df['target'] = self.df[self.target]
        
        if method == 'eq_freq':
            if 'bin_num' not in kw.keys():
                raise NameError('Under eq_freq method, bin_num must be specified')
            else:
                bin_index = dict(((temp_df['var'].value_counts().sort_index().cumsum()/temp_df.shape[0])*self._bin_num).apply(lambda x: ceil(x)))
                temp_df['bin_index'] = temp_df['var'].apply(lambda x: bin_index[x]) 
                
            return groupedresult(temp_df, 'bin_index', 'target')[1]
            
        elif method == 'tree':
            if ('max_depth' not in kw.keys() | 'min_samples_leaf' not in kw.keys()):
                raise NameError("Under tree method, 'max_depth' and 'min_samples_leaf' must be specified")
            else:
                treeclf = tree.DecisionTreeClassifier(max_depth = max_depth, 
                                                      min_samples_leaf = min_samples_leaf).fit(temp_df.drop('target',1), 
                                                                                               self.df[self.target])
                temp_df['bin_index'] = treeclf.apply(temp_df.drop('target',1))
            return groupedresult(temp_df, 'bin_index', 'target')[1]
            
        elif method == 'manual':
            if 'bins' not in kw.keys():
                raise NameError("Under manual method, 'bins' must be specified")
            elif not isinstance(bins, list):
                raise ValueError("The parameter 'bins' should either be list or dictionary")
            else:   
                temp_df['bin_index'] = _var_bins_series(temp_df[var], bins, side = side)
            
            return groupedresult(temp_df, 'bin_index', 'target')[1]
        
        else: raise ValueError('%s method is currently not supported' % method)



    def _format_bucket(self, bins, side = "left"):
        '''
        Return successive buckets based on cut-off values specified by 'bins'.

        Parameters
        ==========
        bins: an iterable object (lists, tuples, arrays etc.)
        side: 'left'/'right', the bucket would be left-sided open or right-sided open

        Return
        ======
        levels: list

        Example
        =======
        bins = [0, 5, 10] / side = 'left'
        return = ['(-inf, 0]', '(0, 5]', '(5, 10]', '(10, +inf]']
        '''
        if np.iterable(bins):
            if (np.diff(bins) < 0).any():
                raise ValueError('bins must increase monotonically.')
            else:
                if side == "left":
                    levels = ["({0}, {1}]".format(bins[i-1],bins[i]) for i in range(1,len(bins))]
                    levels.insert(0, "(-inf, {0}]".format(bins[0]))
                    levels.append("({0}, +inf)".format(bins[-1]))
                elif side == "right":
                    levels = ["[{0}, {1})".format(bins[i-1],bins[i]) for i in range(1,len(bins))]
                    levels.insert(0, "(-inf, {0})".format(bins[0]))
                    levels.append("[{0}, +inf)".format(bins[-1]))
                else: raise ValueError("side should either be 'right' or 'left'.")
        else: raise ValueError("bins should be iterables.")

        return levels

    def _var_bins_series(sr, bins, side = "left"):
        var_index = sr.index
        bins = np.asarray(bins)

        ist_position = bins.searchsorted(sr, side = side)
        na_mask = sr.isnull()
        has_na = na_mask.any()

        buckets = np.asarray(pd.Categorical(ist_position, _format_bucket(bins, side = side), ordered = True, fastpath = True))
        if has_na:
            np.putmask(buckets, na_mask, np.nan)

        return pd.Series(buckets, index = var_index)


    def var_bin_woe(df, target, var, bins, side = "left"):

        df[var].fillna(-9999, inplace = True)

        var_series = _var_bins_series(df[var], bins, side)
        var_woe_dict = dict(variable_binning(df, target, var, bins , 'bins')["WOE"])

        var_woe_series = var_series.map(lambda x: var_woe_dict.get(x))

        return var_woe_series
