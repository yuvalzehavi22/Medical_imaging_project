from joblib import parallel_backend
import numpy as np
import torch
import argparse
import optuna
import wandb # pip install optuna
from models import DMGI
import os

def parse_args_opt():
    # input arguments
    parser = argparse.ArgumentParser(description='DMGI')

    parser.add_argument('--embedder', nargs='?', default='DMGI')
    parser.add_argument('--dataset', type=str, default= 'FairClip_resnet50_categ',
                    choices=['FairClip_resnet50', 'FairClip_biomedclip', 'abide','FairClip_resnet_text', 'FairClip_resnet50_text_without_PCA','FairClip_resnet50_categ'],
                    help='dataset type')
    parser.add_argument('--metapaths', nargs='?', default='type0,type1,type2,type3')

    parser.add_argument('--nb_epochs', type=int, default=70) #10000
    parser.add_argument('--hid_units', type=int, default=217)
    parser.add_argument('--lr', type = float, default = 0.05)
    parser.add_argument('--l2_coef', type=float, default=0.0001)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    
    parser.add_argument('--reg_coef', type=float, default=0.001) #
    parser.add_argument('--sup_coef', type=float, default=0.2) #0.1
    parser.add_argument('--contrastive_coef', type=float, default=0.001) #
    
    parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--nheads', type=int, default=1)
    parser.add_argument('--activation', nargs='?', default='relu')
    parser.add_argument('--isSemi', action='store_true', default=True)
    parser.add_argument('--isBias', action='store_true', default=False)
    parser.add_argument('--isAttn', action='store_true', default=True)
    # Contrastive loss parameters
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature parameter for contrastive loss')
    parser.add_argument('--sched_step', type=int, default=10, help='Step size for learning rate scheduler')


    return parser.parse_known_args()
def save_best_params(trial, value, params):
    file_path = "best_hyperparameters.txt"
    # Check if file exists and read the current best value
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                saved_best_value = float(lines[1].strip().split(":")[1])
            else:
                saved_best_value = -np.inf
    else:
        saved_best_value = -np.inf
    # If current trial value is better than the saved one, update the file
    if value > saved_best_value:
        with open(file_path, "w") as f:
            f.write(f"Trial {trial.number} - Best hyperparameters so far: {params}\n")
            f.write(f"Best value so far: {value}\n")
            
# Define the objective function for Optuna
def objective(trial):
    args, _ = parse_args_opt()

    # Define the hyperparameters to tune
    args.lr = trial.suggest_loguniform('lr', 1e-3, 1e-2)
    args.l2_coef = trial.suggest_loguniform('l2_coef', 1e-6, 1e-5)
    # args.hid_units = trial.suggest_int('hid_units', 32, 256)
    # args.nb_epochs = trial.suggest_int('nb_epochs', 80, 500)
    args.drop_prob = trial.suggest_float('drop_prob', 0.2, 0.8)
    args.reg_coef = trial.suggest_loguniform('reg_coef', 1e-4, 5e-4)
    args.sup_coef = trial.suggest_float('sup_coef', 5e-2, 1)
    #args.contrastive_coef = trial.suggest_loguniform('contrastive_coef', 1e-4, 1e-1)

    args.nheads = trial.suggest_int('nheads', 1, 4)
    args.sc = trial.suggest_float('sc', 1, 3)
    args.sc = trial.suggest_float('--temperature', 0.05, 0.5)

    args.margin = trial.suggest_float('margin', 2, 4)
    # args.sched_step = trial.suggest_int('sched_step', 5, 30)
    
    # Print the trial's hyperparameters
    print(f"\nTrial {trial.number} parameters: lr={args.lr}, "
          f"reg_coef={args.reg_coef}, sup_coef={args.sup_coef}, contrastive_coef={args.contrastive_coef}, "
          f"drop_prob={args.drop_prob}, sc={args.sc}, l2_coef={args.l2_coef}, sched_step={args.sched_step},"
          f"hid_units={args.hid_units}, nb_epochs={args.nb_epochs}, drop_prob={args.drop_prob}, "
          f"nheads={args.nheads}, margin={args.margin}, temperature-{args.temperature}")

    # Initialize W&B with the updated config
    wandb.init(project="Multimodal-medicle-DMGI-optuna-noreen", config=args)  

    # Initialize and train the model
    embedder = DMGI(args)
    mean_auc_val = embedder.training()
    # Save the best parameters so far after each trial
    save_best_params(trial, mean_auc_val, {
        'lr': args.lr,
        'reg_coef': args.reg_coef,
        'sup_coef': args.sup_coef,
        'l2_coef': args.l2_coef,
        'drop_prob': args.drop_prob,
        'nheads':args.nheads,
        'sc':args.sc,
        'margin': args.margin,
        'temperature':args.temperature
    })
    
    wandb.finish()
    return mean_auc_val

def main():
    '''
    Run Optuna's study to optimize the hyperparameters
    '''
    # Parallelize trials across GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1

    # Parallel execution
    with parallel_backend("multiprocessing", n_jobs=num_gpus):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

    print("Best hyperparameters:", study.best_params)
    print("Best loss:", study.best_value)

    # # Print the best trial based on the minimum loss
    # best_trial = study.best_trial
    # print(f"\nBest trial: Trial {best_trial.number}, Accuracy: {best_trial.value}")

    # Save the best model for later use
    # best_model_path = f"best_model_trial_{best_trial.number}.pth"
    # torch.save(best_trial, best_model_path)

if __name__ == "__main__":
    main()
    
# if __name__ == '__main__':
#     # Create Optuna study and run optimization
#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=50)

#     print("Best hyperparameters:", study.best_params)
#     print("Best loss:", study.best_value)

# to avoid memory constrains:
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# python MultiplexNetwork/Hyperparams_opt_DMGI.py --reg_coef 0.0001 --sup_coef 0.01 --dataset 'FairClip_resnet50'

