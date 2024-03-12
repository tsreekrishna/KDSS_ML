import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings("ignore")
from ..utils.helper import seed_everything, grid_search

def early_crop_map_modelling(X_train, X_val, y_train, y_val, n_estimators, learning_rate, max_depth, tuning_metric_type, seed=42):
    # Seeding
    seed_everything(seed) 
   
   # Initializing the variables
    crop_map_dict = {}
    i = 0

    fort_nights = list(X_train.columns)
    params = {'n_estimators':np.arange(n_estimators), 
              'learning_rate':np.arange(learning_rate), 
              'max_depth':np.arange(max_depth)}
    param_grid = list(ParameterGrid(params))

    while (i + 4) <= len(fort_nights):
        if i == 0:
            drop_cols = []
        else:
            drop_cols = fort_nights[-i:]
        temp_X_train = X_train.drop(drop_cols, axis=1)
        temp_X_val = X_val.drop(drop_cols, axis=1)
        metric, model = grid_search(temp_X_train, y_train, temp_X_val, y_val, param_grid, tuning_metric_type, seed)
        fns_used = '-'.join(map(str, temp_X_train.columns[[0,-1]]))
        fns_used.append('-'.join(map(str, temp_X_train.columns[[0,-1]])))
        crop_map_dict[fns_used] = [model, metric]
        i += 1
    return crop_map_dict