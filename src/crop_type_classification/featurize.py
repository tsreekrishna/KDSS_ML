import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ..utils.helper import seed_everything
import warnings
from ..utils.constants import rabi_fns
warnings.filterwarnings("ignore")

def train_val_split(data, val_size, stratify, seed=42):
    # Samples with less than 2 occurrences cannot be divided into two parts (Train & Validation). 
    # These are excluded before splitting and then reintroduced to the training set to enhance generalization.
    data['stratify_comb'] = data.apply(lambda row:'-'.join(map(lambda x: row[x], stratify)), axis=1)
    comb_under_two = data.stratify_comb.value_counts()[data.stratify_comb.value_counts() < 2].index
    samples_under_two = data[data['stratify_comb'].isin(comb_under_two)]
    data.drop(samples_under_two.index, inplace=True)
    # Train val split
    train, val = train_test_split(data, test_size=val_size, stratify=data[stratify], random_state=seed)
    # Reintroduce the samples to train set with less than 3 occurrences
    train = pd.concat([train, samples_under_two], axis=0)
    return train, val

def label_encode(data, target, predictors=rabi_fns):
    label_map = {'Mustard':0, 'Wheat':1, 'Potato':2}
    X, y = data[predictors], data[target]
    # Label encoding
    y = y.apply(lambda row:label_map[row])
    return X, y

def feature_scaling(X, scaler_type='StandardScaler'):
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    scaled_X = pd.DataFrame(scaler.fit_transform(X), columns=rabi_fns, index=X.index)
    return scaled_X, scaler



