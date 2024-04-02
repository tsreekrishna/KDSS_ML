import wandb
import hydra
import omegaconf
import pandas as pd
from src.crop_type_classification.preprocess import rabi_NDVI_preprocess
from src.crop_type_classification.featurize import label_encode, feature_scaling
from src.crop_type_classification.train_model import early_crop_map_modelling
from src.crop_type_classification.eval import rabi_models_testing
from src.utils.helper import pickling
import yaml
import time


@hydra.main(config_path="configs/", config_name="evaluate", version_base='1.2')
def run_inference(cfg):
    config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    # Data Preprocessing
    preprocessed_data, _ = rabi_NDVI_preprocess(config['data_path'])
    
    # Label encoding
    print('2. Label encoding the target variable...')
    time.sleep(1)
    X_test, y_test = label_encode(preprocessed_data, target = config['target'],
                                                    predictors = config['predictors'])

    # Feature scaling
    print('3. Scaling the predictor features...')
    time.sleep(1)
    scaler = pickling(task='read', path=config['scaler_path'])
    X_test = pd.DataFrame(data=scaler.transform(X_test), columns=X_test.columns)
    
    #loading pretrained models
    print('4. Loading pretrained models as specified in the config file...')
    time.sleep(1)
    fns_models_map = {}
    models = config['classifiers']['models_to_load']
    model_dir_path = config['classifiers']['dir_path']
    for model in models:
        fns_used = model[0:13]
        fns_models_map[fns_used] = pickling(task='read', path=model_dir_path + model)
    
    # Logging metirc of train and validation sets on wandb
    print(f'7. Evaluating the best models on test data and logging appropriate metrics....')
    with wandb.init(project="KDSS", entity="wadhwani", config=config, name='test18') as run:
        set_type = 'test'
        metric_reports, tar_vs_fns, conf_mat_plots, roc_curves, pr_curves = rabi_models_testing(X_test, y_test, fns_models_map, config['raw_metric'])
        # Logging on wandb
        for i in range(len(metric_reports)):
            wandb.log({f'Classification_Metrics/{set_type}_{metric_reports[i].name}':wandb.Table(dataframe=metric_reports[i]),
                    f'Confusion Matrices/{set_type}_{i}_img':wandb.Image(conf_mat_plots[i]),
                    f'ROC Curves/{set_type}_{i}_img':wandb.Image(roc_curves[i]),
                    f'PR Curves/{set_type}_{i}_img':wandb.Image(pr_curves[i])})
        wandb.log({f'Misc Plots/{set_type}_Crop Precision vs Fornights Used':wandb.Image(tar_vs_fns)})
        run.finish()
        
if __name__ == "__main__":
    run_inference()