import numpy as np
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings("ignore")
from ..utils.helper import seed_everything, grid_search

def early_crop_map_modelling(X_train, X_val, y_train, y_val, n_estimators, learning_rate, max_depth, raw_metric, tuning_metric_type, seed=42):
    # Seeding
    seed_everything(seed)
   
   # Initializing the variables
    crop_map_dict = {}

    fort_nights = list(X_train.columns)
    params = {'n_estimators':np.arange(n_estimators[0], n_estimators[1], n_estimators[2]), 
              'learning_rate':np.arange(learning_rate[0], learning_rate[1], learning_rate[2]), 
              'max_depth':np.arange(max_depth[0], max_depth[1], max_depth[2])}
    param_grid = list(ParameterGrid(params))
    
    i = 0
    while (i + 4) <= len(fort_nights):
        if i == 0:
            drop_cols = []
        else:
            drop_cols = fort_nights[-i:]
        temp_X_train = X_train.drop(drop_cols, axis=1)
        temp_X_val = X_val.drop(drop_cols, axis=1)
        metric, model = grid_search(temp_X_train, temp_X_val, y_train, y_val, param_grid, raw_metric, tuning_metric_type)
        fns_used = '-'.join(map(str, temp_X_train.columns[[0,-1]]))
        crop_map_dict[fns_used] = model
        print(f'---> Validation on {len(fort_nights) - i} fns done! Best {tuning_metric_type} of {raw_metric} on validation data is {metric}')
        i += 1
    print('---> Model training and tuning done!')
    return crop_map_dict