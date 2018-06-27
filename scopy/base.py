# -*- coding: utf-8 -*-
"""
Created on Wed May  9 11:10:32 2018

@author: lvpengfei
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from math import *
from sklearn import tree
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

#%%
path = 'D:\\LPF\\pyspace\\cash_dd\\dd_data_new.csv'

raw = pd.read_csv(path, encoding = 'gbk')

target_list = ['first_od_days', 'his_od_days', 'if_1st_od_15p','if_1st_never', 'if_his_od_15p']
target = 'if_his_od_15p'
idcol = 'personal_basic_info_id'
time_stamp = ['apply_date','order_trade_date']
keep_list = list(set(target_list + time_stamp + [target, idcol]))


#%%

num_data, obj_data = varTypeSplit(raw, keep_list)
#%%


bins = groupedresult(obj_data,'spread_way', target)[0]

#%%
class VarPreprocess():
    """
    
    """
    
    def __init__(self, keep_list, target):
        self.keep_list = keep_list
        self.target = target
        
    def var_statistics(self, df, keep_list, if_iv = True, target, method = 'equal_freq'):
        """
        """
        var_states = pd.DataFrame()
        
        var_states['mis_rate'] = missingRate(df, keep_list)
        var_states['dom_rate'] = dominantRate(df, keep_list)
        var_states['objlevels'] = levels(df, keep_list)
        
        if if_iv:
            var_states['iv'] = varroughiv(df, keep_list, target, method, bin_num)
            
        return var_states
        

    def varFilter(self, df, 
                        keep_list, 
                        target, 
                        mis_cutoff = 0.9, 
                        dom_cutoff = 0.9, 
                        lev_cutoff = 40,
                        iv_cutoff = 0.02, 
                        method = 'equal_freq'):
        """
        """

        mis_rate = missingRate(df, keep_list)
        dom_rate = dominantRate(df, keep_list)
        objlevels = levels(df, keep_list)
        variv = varRoughiv(df, keep_list, target, method)
        
        mis_list = [] if mis_cutoff is None else mis_rate[mis_rate >= mis_cutoff].index.values.tolist()
        dom_list = [] if dom_cutoff is None else dom_rate[dom_rate >= dom_cutoff].index.values.tolist()
        obj_list = [] if lev_cutoff is None else objlevels[objlevels >= lev_cutoff].index.values.tolist()
        iv_list = [] if iv_cutoff is None else variv[variv >= iv_cutoff].index.values.tolist()
        
        drop_col = list(set(mis_list + dom_list + obj_list + iv_list))

        return df.drop(drop_col, 1)


        

#%%
#Utilities

def varTypeSplit(df, keep_list = []):
    """
    split data base on var dtypes
    """
    num_data = df.drop(keep_list, 1).select_dtypes(exclude = [object])
    obj_data = df.drop(keep_list, 1).select_dtypes(include = [object])
    
    num_data = pd.concat([num_data, df[keep_list]],1)
    obj_data = pd.concat([obj_data, df[keep_list]],1)
    
    return (num_data, obj_data)

#%%
def missingRate(df, keep_list = []):
    """
    Calculate varaible missing rate
    
    Input: 
    1.DataFrame
    2.keep_list: columns you want to keep no matter what.(primary key, timestamps, target etc.)
    
    Output: Series indexed with column name 
    """
    if not isinstance(keep_list, list):
        raise ValueError("keep_list must be a list.")
        
    return df.drop(keep_list, 1).apply(lambda x: x.isnull().sum(), axis = 0)/df.shape[0]

#%%
def dominantRate(df, keep_list = []):
    """
    Calculate dominant rate
    
    Input: DataFrame
    
    Output: Series indexed with column name 
    """
    if not isinstance(keep_list, list):
        raise ValueError("keep_list must be a list.")
        
    return df.drop(keep_list, 1).apply(lambda x: (x.value_counts()/df.shape[0]).max(), axis = 0)

#%%

def levels(df, keep_list = []):
    """
    """
    if not isinstance(keep_list, list):
        raise ValueError("keep_list must be a list.")
    
    obj_list = varTypeSplit(df, keep_list)[1].columns.values.tolist() - keep_list

    return df[obj_list].apply(lambda x: x.value_counts().count(), axis = 0)

#%%
def groupedresult(df, var, target):
    """
    Returns: 
        1.DataFrame, binning statistics for a given column
        2.IV value
    """
    bins = df.groupby(var)[target].agg([np.size,np.mean,np.sum]).rename(columns={'size':'Total_cnt', 
                                                                                 'mean':'Y_rate', 
                                                                                 'sum':'Y_cnt'})
    
    bins['Total_pct'] = bins['Total_cnt'] / bins['Total_cnt'].sum()
    bins['nY_cnt'] = bins['Total_cnt'] - bins['Y_cnt']
    bins['Y_pct'] = (bins['Y_cnt'] / bins['Y_cnt'].sum()).replace(0, 10e-7)
    bins['nY_pct'] = (bins['nY_cnt'] / bins['nY_cnt'].sum()).replace(0, 10e-7)
    
    bins['WOE'] = (bins['Y_pct']/bins['nY_pct']).map(lambda x: log(x))
    bins['IV'] = (bins['Y_pct'] - bins['nY_pct']) * bins['WOE']
    
    return (bins, bins['IV'].sum())

    
#%%
def numiv_eq_freq(col_series):
    """
    Rough IV calculation for numerical variables.
    """
    temp_df = pd.DataFrame()
    
    temp_df['var'] = col_series
    temp_df['target'] = input_target
    
    bin_index = dict(((temp_df['var'].value_counts().sort_index().cumsum()/temp_df.shape[0])*bin_num).apply(lambda x: ceil(x)))
    temp_df['bin_index'] = temp_df['var'].apply(lambda x: map_dict[x])
    
    return groupedresult(temp_df, 'bin_index', 'target')[1]

#%%
def numiv_opt(col_series):
    """
    """
    temp_df = pd.DataFrame()
    
    temp_df['var'] = col_series
    temp_df['target'] = input_target

    treeclf = DecisionTreeClassifier(max_depth = 3).fit(temp_df.drop('target',1), input_target)
    temp_df['bin_index'] = treeclf.apply(temp_df.drop('target',1))
    
    return groupedresult(temp_df, 'bin_index', 'target')[1]


#%%

def cativ(col_series):
    """
    
    """
    temp_df = pd.DataFrame()
    
    temp_df['var'] = col_series
    temp_df['target'] = input_target
    
    bins = groupedresult(temp_df, 'bin_index', 'target')[0].sort_values('Y_rate')
    bins['bin_index'] = (bins['Total_pct'].cumsum() * bin_num).apply(lambda x: ceil(x))
    temp_df['bin_index'] = temp_df['var'].apply(lambda x: dict(bins['bin_index'])[x])
    
    return groupedresult(temp_df, 'bin_index', 'target')[1]



#%%
def varroughiv(df, keep_list, target, method, bin_num):
    """
    """
    if method not in ['equal_freq', 'optimal']:
        raise ValueError("%s method is currently not supported.")
    
    num_data, obj_data = varTypeSplit(df, keep_list)
    
    num_data.fillna(-1, inplace = True)
    obj_data.fillna('Empty', inplace = True)
    
    objvar_iv = obj_data.drop(keep_list, 1).apply(cativ)
    
    if method == 'equal_freq':
        numvar_iv = num_data.drop(keep_list, 1).apply(numiv_eq_freq)
    else:
        numvar_iv = num_data.drop(keep_list, 1).apply(numiv_opt)
    
    return numvar_iv.append(objvar_iv)

    

#%%

def varbinning(df, var, target, method, bin_num = 10):
    """
    """
    



#%%

def variable_binning(df, target, var, bin_rule, return_type):
    '''
    binning for numerical variables
    '''
    # Preparation: some statistics and calculation
    df[var].fillna(-9999, inplace = True)
    n_sample = df.shape[0]
    y_count = df[target].sum()
    test_var = df[[target, var]]

    if type(bin_rule) == type(1):
        percentile = 1.0/ bin_rule
        value_percent = (test_var[var].value_counts().sort_index().cumsum()/n_sample)
        map_dict = dict((value_percent/percentile).apply(lambda x: ceil(x)))
        test_var['qt_binning'] = test_var[var].apply(lambda x: map_dict[x])

    elif type(bin_rule) == type([]):
        test_var['qt_binning'] = pd.cut(test_var[var], bin_rule, include_lowest = True)

    else: raise NameError("'Binning rule' must be integer or list")

    gb_var = test_var.groupby('qt_binning')
    binning_result = gb_var[target].agg({'Totalcnt': 'count',
                                         'Y_rate':np.mean,
                                         'Y_count': np.sum,
                                         'n_Y_count': lambda x: np.sum(1-x),
                                         'Y_pct' : lambda x: np.sum(x / y_count),
                                         'n_Y_pct': lambda x: np.sum((1-x) / (n_sample-y_count))})

    binning_bound = gb_var[var].agg({'Lbound': 'min', 'Ubound': 'max'})

    binning_result['Lbound'] = binning_bound['Lbound']
    binning_result['Ubound'] = binning_bound['Ubound']
    binning_result['PctTotal'] = binning_result['Totalcnt']/n_sample
    binning_result['Y_pct'] = binning_result['Y_pct'].replace(0, 10e-7)

    binning_result['WOE'] = (binning_result['Y_pct']/binning_result['n_Y_pct']).map(lambda x:log(x))
    binning_result['IV'] = (binning_result['Y_pct'] - binning_result['n_Y_pct']) * binning_result['WOE']

    var_iv = binning_result['IV'].sum()

    if return_type == 'iv':
        return pd.Series({var: var_iv})

    elif return_type == 'bins':
        binning_result = binning_result[['Lbound','Ubound','Totalcnt','PctTotal','Y_count','n_Y_count',
                                    'Y_pct','n_Y_pct','Y_rate','WOE','IV']]
        return binning_result


#%%

bins = variable_binning(num_data, target, 'ln_mob_max', [-1, 5, 10, 20, 40], 'bins')

#%%
#bins['Y_rate'].plot.line()
bins['PctTotal'].plot.hist()
ymin, ymax = plt.ylim()
plt.ylim(ymin=0, ymax = ceil(ymax*100)/100)
plt.xlabel("ln_mob_max")
plt.ylabel("Target_rate")
plt.title("Variable Binning - Target rate")

plt.show()

#%%
#Plotting
plt.figure(figsize = (12, 5))
plt.subplot(121)
bins['Y_rate'].plot.line(style='r')
ymin, ymax = plt.ylim()
plt.ylim(ymin=0, ymax = ceil(ymax*100)/100)
plt.title("Bin-Target Rate")

plt.subplot(122)
bins['PctTotal'].plot.bar(secondary_y=True,grid = False)
plt.ylim(ymin = 0, ymax = 1)
plt.title("Bin-Percentage")

plt.show()


#