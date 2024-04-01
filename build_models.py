import wandb
import hydra
import omegaconf
import pandas as pd
from src.crop_type_classification.preprocess import rabi_NDVI_preprocess
from src.crop_type_classification.featurize import train_val_test_split, label_encode, feature_scaling
from src.crop_type_classification.train_model import early_crop_map_modelling
from src.crop_type_classification.eval import rabi_models_testing
from src.utils.helper import pickling
import yaml

@hydra.main(config_path="configs/", config_name="experiment", version_base='1.2')
def build_models(cfg):
    config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # Data Preprocessing
    preprocess_dict = config['data_cleaning']
    if preprocess_dict['bool'] == True:
        preprocessed_data, _ = rabi_NDVI_preprocess(preprocess_dict['input_path'])
        pickling(task='write', object=preprocessed_data.to_dict(), path=preprocess_dict['output_path'])
    else:
        print('1. Data preprocessing skipped. Loading already preprocessed data...')
        preprocessed_pkl = pickling(task='read', path=preprocess_dict['output_path'])
        preprocessed_data = pd.DataFrame(data=preprocessed_pkl)
    
    # Featurization
    featurize_dict = config['featurization']
    if featurize_dict['train_val_test_split']['bool'] == True:
        # Train-val-test split
        train, val, test =  train_val_test_split(data=preprocessed_data, 
                            test_size=featurize_dict['train_val_test_split']['test_size'],
                            val_size=featurize_dict['train_val_test_split']['val_size'],
                            stratify=featurize_dict['train_val_test_split']['stratify'],
                            seed=featurize_dict['train_val_test_split']['seed'])

        # Create a dictionary to store the splits
        splits = {'train': train.index.to_list(), 'val': val.index.to_list(), 'test': test.index.to_list()}
        config['featurization']['splits'] = splits
        # Write the dictionary to a YAML file
        with open('configs/featurize/splits.yaml', 'w') as file:
            yaml.dump(splits, file)
    else:
        print('2. Loading existing train-val-test splits...')
        train = preprocessed_data.loc[featurize_dict['train']]
        val = preprocessed_data.loc[featurize_dict['val']]
        test = preprocessed_data.loc[featurize_dict['test']]
    
    # Label encoding
    X_train, X_val, X_test, y_train, y_val, y_test = label_encode(train, val, test,
                                                    target = featurize_dict['label_encode']['target'],
                                                    predictors = featurize_dict['label_encode']['predictors'])
    
    # Feature scaling
    X_train, X_val, X_test  = feature_scaling(X_train, X_val, X_test, scaler_type=featurize_dict['scaling']['type'])
    
    # Training
    train_dict = config['train']
    crop_map_dict = early_crop_map_modelling(X_train, X_val, y_train, y_val,
                                                    n_estimators = train_dict['n_estimators'],
                                                    learning_rate = train_dict['learning_rate'],
                                                    max_depth = train_dict['max_depth'],
                                                    raw_metric = train_dict['raw_metric'],
                                                    tuning_metric_type = train_dict['tuning_metric_type'],
                                                    seed = train_dict['seed'])
    model_dir_path = config['artifact_paths']['trained_models']['output_dir_path']
    for tup in crop_map_dict.items():
        pickling(task='write', object=tup[1][0], path=model_dir_path+tup[0]+'.pkl') # Pickling trained models
    
    # Evaluation
    test_reports, tar_vs_fns, conf_mat_plots, roc_curves, pr_curves = rabi_models_testing(X_test, y_test, crop_map_dict, train_dict['raw_metric'])
    
    # Wandb Logging
    with wandb.init(project="KDSS", entity="wadhwani", config=config, name='test3', mode='disabled') as run:
        for i in range(len(test_reports)):
            wandb.log({f'Test Reports/{test_reports[i].name}':wandb.Table(dataframe=test_reports[i]),
                       f'Confusion Matrices/{i}_img':wandb.Image(conf_mat_plots[i]),
                       f'ROC Curves/{i}_img':wandb.Image(roc_curves[i]),
                       f'PR Curves/{i}_img':wandb.Image(pr_curves[i])})
        wandb.log({'Misc Plots/Crop Precision vs Fornights Used':wandb.Image(tar_vs_fns)})
        run.finish()
                                                                      
if __name__ == "__main__":
    build_models()

