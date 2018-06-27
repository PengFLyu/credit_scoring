import pandas as pd
import numpy as np
from sklearn import tree



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

    obj_list = varTypeSplit(df, keep_list)[1].columns.values.tolist() - keep_list

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



def numiv_eq_freq(col_series):
    """
    Rough IV calculation for numerical variables.
    """
    temp_df = pd.DataFrame()

    temp_df['var'] = col_series
    temp_df['target'] = input_target

    bin_index = dict(((temp_df['var'].value_counts().sort_index().cumsum()/temp_df.shape[0])*bin_num).apply(lambda x: ceil(x)))
    temp_df['bin_index'] = temp_df['var'].apply(lambda x: bin_index[x])

    return groupedresult(temp_df, 'bin_index', 'target')[1]


def numiv_opt(col_series):
    """
    """
    temp_df = pd.DataFrame()

    temp_df['var'] = col_series
    temp_df['target'] = input_target

    treeclf = tree.DecisionTreeClassifier(max_depth = 3).fit(temp_df.drop('target',1), input_target)
    temp_df['bin_index'] = treeclf.apply(temp_df.drop('target',1))

    return groupedresult(temp_df, 'bin_index', 'target')[1]



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
