import argparse
import os
from sklearn.model_selection import KFold
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
from torchvision import transforms
from data_load import FundusDataset
from models.resnet_simclr import ResNetSimCLR
from simclr_micle import SimCLR_micle
import wandb
import warnings

warnings.filterwarnings("ignore")

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-a', '--arch', metavar='ARCH', default='biomedclip',
                    choices=['resnet18', 'resnet50', 'biomedclip'],
                    help='model architecture: resnet18 | resnet50 | biomedclip (default: biomedclip)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--use', default='train', type=str, help='options: train, valid, test')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning_rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true', help='Use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=1024, type=int, help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int, help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N', help='Number of views for contrastive learning training.')
parser.add_argument('--gpu_index', default=0, type=int, help='GPU index.')
parser.add_argument('--save_path', default='./models', type=str, help='Directory to save the trained models')

parser.add_argument('--mean_r', default=0, type=float, help='Transform normalization')
parser.add_argument('--mean_g', default=0, type=float, help='Transform normalization')
parser.add_argument('--mean_b', default=0, type=float, help='Transform normalization')
parser.add_argument('--std_r', default=0, type=float, help='Transform normalization')
parser.add_argument('--std_g', default=0, type=float, help='Transform normalization')
parser.add_argument('--std_b', default=0, type=float, help='Transform normalization')

#args = parser.parse_args()

# Define function to save the model after training
def save_model(model, optimizer, scheduler, epochs, filename):
    checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, filename)

# Define function to load a saved model
def load_model(filename, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return model, optimizer, scheduler

# Define function to extract embeddings
def extract_embeddings(dataloader, model, args):
    model.eval()  # Set model to evaluation mode
    all_features = []
    all_user_ids = []

    for images, _, user_ids in dataloader:
        images = images.to(args.device)
    
        with torch.no_grad():
            features = model(images)  # Extract features from the model
        
        all_features.append(features.cpu())
        all_user_ids.extend(user_ids)

    all_features = torch.cat(all_features, dim=0)
    return all_features, all_user_ids

def main():

    # Set PYTORCH_CUDA_ALLOC_CONF to avoid memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."

    # Initialize W&B run and log the parameters
    wandb.init(project="Multimodal-medicle-Image embedder", config=args)        

    # Set CUDA settings if applicable
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    print(f"Using device: {args.device}")

    # Set the chosen seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ----------------------------------------------------------------------
    # ----------------- loading and preprocessing the data -----------------
    # ----------------------------------------------------------------------
    print("Preparing images")

    # Define transformations
    mean = [args.mean_r, args.mean_g, args.mean_b]
    std = [args.std_r, args.std_g, args.std_b]

    transform = transforms.Compose([
        transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load datasets
    train_image_dir = 'dataset/training_extracted_images' #'training_extracted_images'
    train_dataset = FundusDataset(use='training', image_dir=train_image_dir, transform=transform)
    train_loader = train_dataset.create_dataloader(batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # Load validation dataset
    val_image_dir = 'dataset/validation_extracted_images' #'validation_extracted_images'
    val_dataset = FundusDataset(use='validation', image_dir=val_image_dir, transform=transform)
    val_loader = val_dataset.create_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # test_image_dir = 'test_extracted_images'
    # test_dataset = FundusDataset(use='test', image_dir=test_image_dir, transform=transform)
    # test_loader = test_dataset.create_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    print('start training...')

    # Initialize model, optimizer, and scheduler
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    # Initialize SimCLR with MICLE
    simclr_micle = SimCLR_micle(model=model, args=args, device=args.device)

    # Train the model on this fold
    simclr_micle.train(train_loader, optimizer, scheduler, val_loader=val_loader)
    
    # # Evaluate using Linear Evaluation Protocol
    # linear_eval_accuracy  = simclr_micle.linear_evaluation(train_loader, val_loader)
    # print(f"Linear Evaluation Accuracy: {linear_eval_accuracy}")

    # Save the model for this fold
    save_model(model, optimizer, scheduler, args.epochs, f'best_model_{args.arch}.pth')

    # ----------------------------------------------------------------------
    # ------------------- Load Model and Extract Embeddings ----------------
    # ----------------------------------------------------------------------
    
    print('Extracting embeddings from test set...')

    test_image_dir = 'dataset/test_extracted_images'
    test_dataset = FundusDataset(use='test', image_dir=test_image_dir, transform=transform)
    test_loader = test_dataset.create_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Load the trained model for embedding extraction
    model, _, _ = load_model(f'best_model_{args.arch}.pth', model)
    simclr_micle = SimCLR_micle(model=model, args=args, device=args.device)

    # Evaluate using Linear Evaluation Protocol
    linear_test_accuracy  = simclr_micle.linear_evaluation(train_loader, test_loader)
    print(f"Linear Test Accuracy: {linear_test_accuracy}")

    # Extract embeddings for training, validation and test set
    groups = ['train', 'val', 'test']
    for group_type in groups:
        if group_type == 'train':
            data_loader = train_loader
        elif group_type == 'val':
            data_loader = val_loader
        else:
            data_loader = test_loader

        dataset_features, dataset_user_ids = extract_embeddings(data_loader, model, args)

        # Save test embeddings
        output_dir = f'./extracted_feature_{args.arch}_new/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        np.savetxt(os.path.join(output_dir, f'{group_type}_feature.csv'), np.array(dataset_features.numpy(), dtype=float), delimiter=',')
        np.savetxt(os.path.join(output_dir, f'{group_type}_id.csv'), np.array(dataset_user_ids, dtype=str), fmt='%s', delimiter=',')

if __name__ == "__main__":
    main()


# python Image_embedder/SimCLR/run_model.py --arch "resnet50" --epochs 100 --batch_size 32 --lr 0.00007720667856219949 --weight_decay 0.00000805047459559454 --seed 42 --mean_r 0.4965942943023609 --mean_g 0.34832272153877525 --mean_b 0.34631267195331167 --std_r 0.24671120698077464 --std_g 0.3316894433365245 --std_b 0.3981696270606668
# python Image_embedder/SimCLR/run_model.py --arch "biomedclip" --epochs 100 --batch_size 32 --out_dim 512 --lr 0.00001076258001714231 --weight_decay 0.00001261554499309077 --seed 0 --mean_r 0.4697076487547538 --mean_g 0.49192886761307025 --mean_b 0.3109853660307996 --std_r 0.3928844016884331 --std_g 0.2355402551428965 --std_b 0.3767927298736668