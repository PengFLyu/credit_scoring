import pandas as pd
import numpy as np
from sklearn import tree



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


def dominantRate(df, keep_list = []):
    """
    Calculate dominant rate
    
    Input: DataFrame
    
    Output: Series indexed with column name 
    """
    if not isinstance(keep_list, list):
        raise ValueError("keep_list must be a list.")
    
    return df.drop(keep_list, 1).apply(lambda x: (x.value_counts()/df.shape[0]).max(), axis = 0)


def levels(df, keep_list = []):
    """
    """
    if not isinstance(keep_list, list):
        raise ValueError("keep_list must be a list.")
    
    obj_list = list(set(varTypeSplit(df, keep_list)[1].columns.values.tolist()) - set(keep_list))

    return df[obj_list].apply(lambda x: x.value_counts().count(), axis = 0)


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


def WOE_mapping(df, select_var, target, side = "left"):
    var_list = list(select_var.keys())
    output_df = pd.DataFrame()
    output_df[target] = df[target]

    for i in var_list:
        output_df[i] = var_bin_woe(df, target, i, select_var.get(i), side = side)

    return output_df


def binning_plot(df):
    '''
    plot Y_rate and bin_percent curves
    '''
    plt.figure(figsize=(9, 4))
    plt.subplot(121)
    ax1a = df['Y_rate'].plot.line(style='r')

    plt.subplot(122)
    ax1b = df['PctTotal'].plot.bar(secondary_y=True,alpha = 0.3, grid = False)

    plt.tight_layout()
    plt.show()
