from copy import deepcopy
from typing import Dict, List
from itertools import combinations
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
import random
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support as score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def generate_label_set_map(label_map: Dict[str, str]) -> Dict[str, str]:
    set_map = {}
    for i in range(1,len(label_map)+1):
        for comb in combinations(label_map.keys(), i):
            key_comb = ''
            value_comb = ''
            for items in comb:
                key_comb =  key_comb + ' ' + items
                value_comb =  value_comb + ' ' + label_map[items]
            key_comb, value_comb = key_comb.lstrip(), value_comb.lstrip()
            set_map[key_comb] = value_comb
    return set_map

def dip_impute(row):
    lst = deepcopy(row)
    act_strt_idx = lst.index.get_loc('oct_2f')
    for i in range(len(lst.loc['oct_2f':'apr_1f'])):
        actual_idx = i + act_strt_idx
        if ((lst.iloc[actual_idx-1] - lst.iloc[actual_idx]) >= 20) and ((lst.iloc[actual_idx+1] - lst.iloc[actual_idx]) >= 20):
            lst.iloc[actual_idx] = (lst.iloc[actual_idx-1] + lst.iloc[actual_idx+1])/2
    return lst

def sowing_period(row):
    sowing_periods = row.loc['oct_1f':'dec_2f'].index
    sowing_periods_NDVI = row.loc['oct_1f':'dec_2f']
    minima = np.argmin(sowing_periods_NDVI)
    ndvi_values = row.loc['oct_1f':'apr_2f']
    i = minima
    while i < len(sowing_periods):
        if ndvi_values.iloc[i] in set(np.arange(110, 141)):
            if (ndvi_values.iloc[i+1] - ndvi_values.iloc[i]) > 5:
                if ((ndvi_values.iloc[i+1] - ndvi_values.iloc[i+4]) < 30):
                    return sowing_periods[i]
        i += 1 
    return 'Unknown'

def harvest_period(row):
    sowing_period_idx = row.index.get_loc(row['sowing_period'])
    i = sowing_period_idx + 6
    while i < len(row.loc[:'apr_2f']):
        if row.iloc[i] < 140:
            return row.index[i-1]
        i += 1
    return 'Unknown'

def less_than_150_drop(row):
    sp_loc = row.index.get_loc(row['sowing_period'])
    hp_loc = row.index.get_loc(row['harvest_period'])
    if max(row.iloc[sp_loc+1:hp_loc]) < 150:
        return False
    return True

def batch_prediction_prob(data, n_features, batch_size, trained_classifier):
    tensor = torch.Tensor(data.values)
    data_loader = DataLoader(tensor, batch_size=batch_size)
    with torch.no_grad():
        pred_prob = []
        for batch in data_loader:
            batch = batch.view([batch.shape[0], -1, n_features])
            trained_classifier.eval()
            pred_prob.append(trained_classifier.forward(batch))
    pred_prob = np.vstack([np.array(pred_prob[:-1]).reshape(-1,2), pred_prob[-1]])
    return pred_prob

def seed_everything(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    
def train_model(predictors, target, n_estimators, learning_rate, max_depth):
    classifier = XGBClassifier(n_estimators, learning_rate, max_depth)
    classifier.fit(predictors, target)
    return classifier

def target_metric(y_true, y_pred, raw_metric, tuning_metric_type):
    precision, recall, f1score, support = score(y_true, y_pred, labels=[0,1,2])
    metric_map = {'precision': precision,'recall': recall, 'f1score': f1score,'support': support}
    raw_metric = metric_map[raw_metric]
    if tuning_metric_type == 'harmonic_mean':
        tuning_metric = len(raw_metric) / np.sum(1.0 / raw_metric)
    return tuning_metric

def grid_search(X_train, y_train, X_val, y_val, param_grid, raw_metric, tuning_metric_type):
    val_metrics = []
    for p in param_grid:
        classifier = train_model(X_train, y_train, **p)
        val_pred = classifier.predict(X_val)
        score = target_metric(y_val, val_pred, raw_metric, tuning_metric_type)
        val_metrics.append(score)
    best_val_metric_idx = np.argmax(val_metrics)
    best_val_metric = val_metrics[best_val_metric_idx]
    best_model = train_model(X_train, y_train, **param_grid[best_val_metric_idx])
    return best_val_metric, best_model

def target_vs_fns_plot(test_scores, fns_used):
    sns.lineplot(y=np.array(test_scores)[:,0], x=fns_used, label='precision_min_class (Mustard)', marker="o")
    sns.lineplot(y=np.array(test_scores)[:,1], x=fns_used, label='precision_maj_class (Wheat)', marker="o")
    sns.lineplot(y=np.array(test_scores)[:,2], x=fns_used, label='precision_min_class (Potato)', marker="o")
    plt.xlabel('Fns Used')
    plt.ylabel('precision Scores')
    plt.xticks(rotation='vertical')
    plt.title('Wheat vs Mustard vs Potato')
    plt.ylim([0.5,1.2])
    plt.grid()
    fig = plt.gcf()
    plt.close()
    return fig

def confusion_matrix_plot(confusion_matrices, fns_used):
    plt.figure(figsize=(17,35))
    for i in range(len(confusion_matrices)):
        plt.subplot(6,2,i+1)
        sns.heatmap(confusion_matrices[i], annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class/GT')
        plt.title(f'Fns Used: [{fns_used[i]}]')
    fig = plt.gcf()
    plt.close()
    return fig

def pickling(task, object, path):
    if task == 'write':
        with open(path, 'wb') as handle:
            pickle.dump(object, handle)
        return
    elif task =='read':
        with open(path, 'rb') as handle:
            file = pickle.load(handle)
    return file