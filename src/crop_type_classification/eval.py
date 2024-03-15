# %matplotlib ipympl
import pandas as pd
from sklearn.metrics import classification_report
from ..utils.helper import target_vs_fns_plot, confusion_matrix_plot
import wandb

def rabi_models_testing(X_test, y_test, crop_map_dict, raw_metric):
    
    print('3. Evaluating the best models on test data....')
    
    label_crop_map = {0:'Mustard', 1:'Wheat',2:'Potato'}
    test_reports = []
    test_scores = []
    conf_mat_arrays = []
    fns_used = list(crop_map_dict.keys())
    
    for i in range(len(crop_map_dict)):
        strt_fn, end_fn = fns_used[i].split('-')
        temp_X_test = X_test.loc[:, strt_fn:end_fn]
        test_pred = crop_map_dict[fns_used[i]][0].predict(temp_X_test)
        report = classification_report(y_test, test_pred, labels=[0, 1, 2], target_names=['Mustard', 'Wheat','Potato'], output_dict=True)
        df = pd.DataFrame(report).transpose().loc[:'Potato',:'f1-score']
        test_reports.append(df)
        test_scores.append(df[raw_metric].values)
        y_test, test_pred = pd.Series(y_test.values, name='crop_type'), pd.Series(test_pred, name='pred')
        cf = pd.crosstab(y_test, test_pred).rename(label_crop_map).rename(label_crop_map, axis=1)
        conf_mat_arrays.append(cf)

        
    tar_vs_fns = target_vs_fns_plot(test_scores, fns_used)
    conf_mat_plots = confusion_matrix_plot(conf_mat_arrays, fns_used)
    
    return test_reports, tar_vs_fns, conf_mat_plots