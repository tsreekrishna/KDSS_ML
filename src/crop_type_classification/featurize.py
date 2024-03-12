import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ..utils.helper import seed_everything
import warnings
from ..utils.constants import rabi_fns
warnings.filterwarnings("ignore")

def train_val_test_split(data, test_size, val_size, stratify, seed=42):
    # Samples with less than 3 occurrences cannot be divided into three parts (Train, Validation, Test). 
    # These are excluded before splitting and then reintroduced to the training set to enhance generalization.
    data['stratify_comb'] = data.apply(lambda row:'-'.join(map(lambda x: row[x], stratify)), axis=1)
    comb_under_three = data.stratify_comb.value_counts()[data.stratify_comb.value_counts() < 3].index
    samples_under_three = data[data['stratify_comb'].isin(comb_under_three)]
    data.drop(samples_under_three.index, inplace=True)
    # Train test val split
    train_test, val = train_test_split(data, test_size=val_size, stratify=data[stratify], random_state=seed)
    actual_test_size = test_size/(1 - (test_size + val_size))
    train, test = train_test_split(train_test, test_size=actual_test_size, stratify=train_test[stratify], random_state=0)
    # Reintroduce the samples to train set with less than 3 occurrences
    train = pd.concat([train, samples_under_three], axis=0)
    return train, test, val

def label_encode(train, test, val, target, predictors=rabi_fns):
    label_map = {'Mustard':0, 'Wheat':1, 'Potato':2}
    X_train, y_train = train[predictors], train[target]
    X_val, y_val = val[predictors], val[target]
    X_test, y_test = test[predictors], test[target]
    for target in y_train, y_val, y_test:
        target = target.apply(lambda row:label_map[row])
    return X_train, X_test, X_val, y_train, y_test, y_val

def feature_scaling(X_train, X_test, X_val, scaler_type='StandardScaler'):
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=rabi_fns)
    for data in X_test, X_val:
        data = pd.DataFrame(scaler.transform(data), columns=rabi_fns)
    return X_train, X_test, X_val



