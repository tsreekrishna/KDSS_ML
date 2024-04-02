import wandb
import hydra
import omegaconf
import pandas as pd
from src.crop_type_classification.preprocess import rabi_NDVI_preprocess
from src.crop_type_classification.featurize import train_val_split, label_encode, feature_scaling
from src.crop_type_classification.train_model import early_crop_map_modelling
from src.crop_type_classification.eval import rabi_models_testing
from src.utils.helper import pickling
import yaml

@hydra.main(config_path="configs/", config_name="modelling", version_base='1.2')
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
    if featurize_dict['train_val_split']['bool'] == True:
        print('2. Performing train-val split...')
        # Train-val-test split
        train, val =  train_val_split(data=preprocessed_data,
                            val_size=featurize_dict['train_val_split']['val_size'],
                            stratify=featurize_dict['train_val_split']['stratify'],
                            seed=featurize_dict['train_val_split']['seed'])

        # Create a dictionary to store the splits
        splits = {'train': train.index.to_list(), 'val': val.index.to_list()}
        config['featurization']['splits'] = splits
        # Write the dictionary to a YAML file
        with open('configs/featurization/splits.yaml', 'w') as file:
            yaml.dump(splits, file)
    else:
        print('2. Loading existing train-val splits...')
        train = preprocessed_data.loc[featurize_dict['train']]
        val = preprocessed_data.loc[featurize_dict['val']]
    
    # Label encoding
    print('3. Label encoding the target variable...')
    X_train, y_train = label_encode(train, target = featurize_dict['label_encode']['target'],
                                        predictors = featurize_dict['label_encode']['predictors'])
    X_val, y_val =  label_encode(val, target = featurize_dict['label_encode']['target'],
                                    predictors = featurize_dict['label_encode']['predictors'])
    
    # Feature scaling
    print('4. Feature scaling the predictors...')
    X_train, scaler = feature_scaling(X_train, scaler_type=featurize_dict['scaling']['type'])
    X_val = pd.DataFrame(data=scaler.transform(X_val), columns=X_val.columns)
    pickling(task='write', object=scaler, path=featurize_dict['scaling']['model_dir_path']+'scaler.pkl')
    
    # Training
    train_dict = config['train']
    print('5. Training and tuning crop classification models...')
    fns_models_map = early_crop_map_modelling(X_train, X_val, y_train, y_val,
                                                    n_estimators = train_dict['training_params']['n_estimators'],
                                                    learning_rate = train_dict['training_params']['learning_rate'],
                                                    max_depth = train_dict['training_params']['max_depth'],
                                                    raw_metric = train_dict['training_params']['raw_metric'],
                                                    tuning_metric_type = train_dict['training_params']['tuning_metric_type'],
                                                    seed = train_dict['training_params']['seed'])
    model_dir_path = train_dict['model_dir_path']
    for tup in fns_models_map.items():
        pickling(task='write', object=tup[1], path=model_dir_path+tup[0]+'.pkl') # Pickling trained models
    
    # Logging metirc of train and validation sets on wandb
    print(f'6. Evaluating the best models on train and val data and logging appropriate metrics....')
    with wandb.init(project="KDSS", entity="wadhwani", config=config, name='test15') as run:
        set_type = 'train'
        for sets in ((X_train, y_train), (X_val, y_val)):
            metric_reports, tar_vs_fns, conf_mat_plots, roc_curves, pr_curves = rabi_models_testing(*sets, fns_models_map, train_dict['training_params']['raw_metric'])
            # Logging on wandb
            for i in range(len(metric_reports)):
                wandb.log({f'Classification_Metrics/{set_type}_{metric_reports[i].name}':wandb.Table(dataframe=metric_reports[i]),
                        f'Confusion Matrices/{set_type}_{i}_img':wandb.Image(conf_mat_plots[i]),
                        f'ROC Curves/{set_type}_{i}_img':wandb.Image(roc_curves[i]),
                        f'PR Curves/{set_type}_{i}_img':wandb.Image(pr_curves[i])})
            wandb.log({f'Misc Plots/{set_type}_Crop Precision vs Fornights Used':wandb.Image(tar_vs_fns)})
            set_type = 'val'
        run.finish()
                                                                      
if __name__ == "__main__":
    build_models()

