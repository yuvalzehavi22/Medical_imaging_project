import optuna
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
from simclr_micle import SimCLR_micle
from models.resnet_simclr import ResNetSimCLR
from data_load import FundusDataset
import wandb
import numpy as np
import argparse
import os
import warnings
from joblib import parallel_backend

warnings.filterwarnings("ignore")

# Argument parser
parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-a', '--arch', metavar='ARCH', default='biomedclip',
                    choices=['resnet18', 'resnet50', 'biomedclip'],
                    help='model architecture: resnet18 | resnet50 | biomedclip (default: biomedclip)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--use', default='training', type=str, help='options: training, validation, test')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=512, type=int, help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int, help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N', help='Number of views for contrastive learning training.')
parser.add_argument('--gpu_index', default=0, type=int, help='Gpu index.')

args = parser.parse_args()

def objective(trial):
    # Set PYTORCH_CUDA_ALLOC_CONF to avoid memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Define hyperparameter search space and update `args`
    setattr(args, 'arch', trial.suggest_categorical("arch", ['biomedclip']))
    setattr(args, 'lr', trial.suggest_loguniform("lr", 1e-5, 1e-3))
    setattr(args, 'weight_decay', trial.suggest_loguniform("weight_decay", 1e-6, 1e-4))
    setattr(args, 'seed', trial.suggest_categorical("seed", [42, 0]))  # Suggesting categorical seed values

    # Set the chosen seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Add normalization stats as hyperparameters
    mean = [
        trial.suggest_uniform("mean_r", 0.3, 0.6),
        trial.suggest_uniform("mean_g", 0.3, 0.6),
        trial.suggest_uniform("mean_b", 0.3, 0.6)
    ]
    std = [
        trial.suggest_uniform("std_r", 0.2, 0.4),
        trial.suggest_uniform("std_g", 0.2, 0.4),
        trial.suggest_uniform("std_b", 0.2, 0.4)
    ]   

    # Include mean and std in the W&B configuration
    config = {
        'arch': args.arch,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'mean_r': mean[0],
        'mean_g': mean[1],
        'mean_b': mean[2],
        'std_r': std[0],
        'std_g': std[1],
        'std_b': std[2]
    }

    # Initialize W&B with the updated config
    wandb.init(project="SimCLR_hyperparam_optimization", config=config)

    # Print current trial parameters
    print(f"\nTrial {trial.number}: arch={args.arch}, lr={args.lr}, weight_decay={args.weight_decay}, \n"
          f"batch_size={args.batch_size}, epochs={args.epochs}, seed={args.seed}, mean={mean}, std={std}")
    
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(args.gpu_index)
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    print(f"Using device: {args.device}")


    # Dataset loading and transformations 
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load training datasets
    train_image_dir = 'dataset/training_extracted_images'
    train_dataset = FundusDataset(use='training', image_dir=train_image_dir, transform=transform)
    train_loader = train_dataset.create_dataloader(batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # Load validation dataset
    val_image_dir = 'dataset/validation_extracted_images'
    val_dataset = FundusDataset(use='validation', image_dir=val_image_dir, transform=transform)
    val_loader = val_dataset.create_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Initialize model, optimizer, and scheduler
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, pretrained=True).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    # Initialize SimCLR with MICLE
    simclr_micle = SimCLR_micle(model=model, args=args, device=args.device)

    # Train the model
    simclr_micle.train(train_loader, optimizer, scheduler, val_loader=val_loader)

    # Evaluate using Linear Evaluation Protocol
    linear_eval_accuracy = simclr_micle.linear_evaluation(train_loader, val_loader)

    # Return final linear evaluation accuracy for Optuna to maximize
    wandb.finish()  # Finish WandB logging for this trial
    return linear_eval_accuracy

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

    # Print the best trial based on the minimum loss
    best_trial = study.best_trial
    print(f"\nBest trial: Trial {best_trial.number}, Accuracy: {best_trial.value}")

    # Save the best model for later use
    best_model_path = f"best_model_trial_{best_trial.number}.pth"
    torch.save(best_trial, best_model_path)

if __name__ == "__main__":
    main()


# python Image_embedder/SimCLR/optuna_SimCLR.py --out_dim 512 --epochs 30