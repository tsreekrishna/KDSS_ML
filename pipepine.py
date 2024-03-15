import wandb
import hydra
import omegaconf
from src.crop_type_classification.preprocess import rabi_data_preprocess
from src.crop_type_classification.featurize import train_val_test_split, label_encode, feature_scaling
from src.crop_type_classification.train_model import early_crop_map_modelling
from src.crop_type_classification.eval import rabi_models_testing
from src.utils.helper import pickling

@hydra.main(config_path="configs/", config_name="experiment", version_base='1.2')
def run_experiment(cfg):
    config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    with wandb.init(project="KDSS", entity="wadhwani", config=config, name='test', id='125') as run:
        
        # Data Preprocessing
        preprocess_dict = wandb.config['artifact_paths']['cleaning']
        preprocessed_data, _ = rabi_data_preprocess(preprocess_dict['input_path'])
        pickling(task='write', object=preprocessed_data.to_dict(), path=preprocess_dict['output_path'])
        
        # Featurization
        featurize_dict = wandb.config['featurize']
        # Train-val-test split
        train, val, test =  train_val_test_split(data=preprocessed_data, 
                            test_size=featurize_dict['train_test_split']['test_size'],
                            val_size=featurize_dict['train_test_split']['val_size'],
                            stratify=featurize_dict['train_test_split']['stratify'],
                            seed=featurize_dict['train_test_split']['seed'])
        # Label encoding
        pred_target_splits = label_encode(train, val, test,
                                        target = featurize_dict['label_encode']['target'],
                                        predictors = featurize_dict['label_encode']['predictors'])
        X_train, X_val, X_test, y_train, y_val, y_test = pred_target_splits
        # Feature scaling
        X_train, X_val, X_test  = feature_scaling(X_train, X_val, X_test, 
                                                    scaler_type=featurize_dict['scaler_type'])
        
        # Training
        train_dict = wandb.config['train']
        crop_map_dict = early_crop_map_modelling(X_train, X_val, y_train, y_val,
                                                 n_estimators = train_dict['n_estimators'],
                                                 learning_rate = train_dict['learning_rate'],
                                                 max_depth = train_dict['max_depth'],
                                                 raw_metric = train_dict['raw_metric'],
                                                 tuning_metric_type = train_dict['tuning_metric_type'],
                                                 seed = train_dict['seed'])
        
        # Evaluation
        test_reports, tar_vs_fns, conf_mat_plots = rabi_models_testing(X_test, y_test, crop_map_dict, 
                                                                 train_dict['raw_metric'])
        for i in range(len(test_reports)):
            wandb.log({f'{i}_report':wandb.Table(dataframe=test_reports[i])})
            wandb.log({f'{i}_img':wandb.Image(conf_mat_plots[i])})
        wandb.log({'Crop Precision vs Fornights Used':wandb.Image(tar_vs_fns)})
        run.finish()
                                                                      
if __name__ == "__main__":
    run_experiment()

