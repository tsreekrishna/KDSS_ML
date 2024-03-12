import numpy as np
import geopandas as gp
import pandas as pd
from ..utils.helper import sowing_period, harvest_period, dip_impute, less_than_150_drop
from ..utils.constants import rabi_fns
import pickle

def rabi_data_preprocess(data_path: np.ndarray) -> (str, [pd.DataFrame, pd.DataFrame]):
    '''
    Preprocess the input data and filter non_crop data points

    Parameters
    -----------
    data : local data dict path in the form of a string

    Returns
    -------
    2 pandas dataframes -> preprocessed crop, non-crop

    '''
    # data import
    with open(data_path, 'rb') as file:
        raw_data_dict = pickle.load(file)
        
    # Removing redundant columns and reorganizing and formatting essential ones.
    raw_data = gp.GeoDataFrame(raw_data_dict)
    features = rabi_fns.copy()
    features.append('crop_type')
    data = raw_data[features]
    outliers = gp.GeoDataFrame()

    # Imputing the NDVI fornights with the averages if the dip is greater than 20 when compared to both adjs 
    data = data.apply(dip_impute, axis=1)

    # Determining sowing period(S.P). If the S.P is not found, then it is regarded as a non crop. 
    data['sowing_period'] = data.apply(sowing_period, axis=1)
    new_outliers = data[data.sowing_period == 'Unknown']
    outliers = pd.concat([outliers, new_outliers])
    data.drop(new_outliers.index, inplace=True)

    # Determining harvest period(H.P). If the H.P is not found, then it is regarded as a non crop.
    data['harvest_period'] = data.apply(harvest_period, axis=1)
    new_outliers = data[data.harvest_period == 'Unknown']
    outliers = pd.concat([outliers, new_outliers])
    data.drop(new_outliers.index, inplace=True)

    # Dropping the rows which have max of NDVI values less than 150 for all the values between sp and hp.
    new_outliers = data[data.apply(less_than_150_drop, axis=1) == False]
    outliers = pd.concat([outliers, new_outliers])
    data = data.drop(new_outliers.index)

    # Dropping the duplicates (if any)
    data = data.drop_duplicates()
    
    print('Data Preprocessing done!')

    return data, outliers

if __name__ == '__main__': 
    rabi_data_preprocess('data/master_pickle_raw')