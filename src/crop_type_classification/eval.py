import pandas as pd
from sklearn.metrics import classification_report
from ..utils.helper import target_vs_fns_plot, confusion_matrix_plot

def rabi_models_testing(X_test, y_test, crop_map_dict, raw_metric):
    test_reports = []
    test_scores = []
    confusion_matrices = []
    fns_used = list(crop_map_dict.keys())
    
    for i in range(len(crop_map_dict)):
        strt_fn, end_fn = fns_used[i].split('-')
        temp_X_test = X_test.loc[:, strt_fn:end_fn]
        test_pred = crop_map_dict[fns_used][0].predict(temp_X_test)
        report = classification_report(y_test, test_pred, target_names=['Mustard', 'Wheat','Potato'], output_dict=True)
        df = pd.DataFrame(report).transpose().loc[:'Potato',:'f1-score']
        test_reports.append(df)
        test_scores.append(df[raw_metric].values)
        cf = pd.crosstab(y_test, pd.Series(test_pred, name='pred')).rename({0:'Mustard', 1:'Wheat',2:'Potato'}).rename({0:'Mustard', 1:'Wheat',2:'Potato'}, axis=1)
        confusion_matrices.append(cf)
        
    tar_vs_fns = target_vs_fns_plot(test_scores, fns_used)
    conf_mat = confusion_matrix_plot(confusion_matrices, fns_used)
    
    return test_reports, tar_vs_fns, conf_mat