# Code based on https://github.com/pcy1302/DMGI
import numpy as np
from torchinfo import summary
import wandb
np.random.seed(0)
import torch
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import argparse

def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='DMGI')

    parser.add_argument('--embedder', nargs='?', default='DMGI')
    #parser.add_argument('--dataset', nargs='?', default='FairClip') # abide
    parser.add_argument('--dataset', type=str, default='FairClip_resnet50_text',
                    choices=['FairClip','FairClip_resnet50', 'FairClip_biomedclip', 'abide','FairClip_resnet_text','FairClip_resnet50_text','FairClip_resnet50_categ'],
                    help='dataset type')
    parser.add_argument('--metapaths', nargs='?', default='type0,type1,type2') #,type3

    parser.add_argument('--nb_epochs', type=int, default=30) #10000
    parser.add_argument('--hid_units', type=int, default=217)
    parser.add_argument('--lr', type = float, default = 0.05)
    
    parser.add_argument('--l2_coef', type=float, default=0.0001)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    
    parser.add_argument('--reg_coef', type=float, default=0.001) #
    parser.add_argument('--sup_coef', type=float, default=0.2) #0.1
    parser.add_argument('--contrastive_coef', type=float, default=0.1) #
    
    parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--nheads', type=int, default=1)
    parser.add_argument('--activation', nargs='?', default='relu')
    parser.add_argument('--isSemi', action='store_true', default=True)
    parser.add_argument('--isBias', action='store_true', default=False)
    parser.add_argument('--isAttn', action='store_true', default=True)
    parser.add_argument('--use_const_loss', action='store_true', default=True)    
    # Contrastive loss parameters
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature parameter for contrastive loss')
    parser.add_argument('--sched_step', type=int, default=10, help='Step size for learning rate scheduler')

    return parser.parse_known_args()


def main():
    args, unknown = parse_args()

    # Initialize W&B run and log the parameters
    wandb.init(project="Multimodal-medicle-DMGI", config=args)    
    
    from models import DMGI
    embedder = DMGI(args)
    embedder.training()

if __name__ == '__main__':
    main()


# to avoid memory constrains:
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------- AUC 0.758 -------------------
# python MultiplexNetwork/main.py --reg_coef 0.000435 --sup_coef 0.2 --dataset 'FairClip_resnet_text' --lr 0.00118 --nb_epochs 200 --l2_coef 1.9046e-06 --drop_prob 0.3842 --hid_units 217


# ---------------- Best so far : 0.774:  [Classification] Accuracy: 0.7225 (0.0036) | AUC: 0.7747 (0.0016) | Sensitivity (Recall): 0.7101 (0.0104) | Macro-F1: 0.7225 (0.0036) -------------------
# python MultiplexNetwork/main.py --reg_coef 0.000435 --sup_coef 0.2  --lr 0.00118 --nb_epochs 200 --l2_coef 1.9046e-06 --drop_prob 0.3842 --hid_units 217 --dataset 'FairClip_resnet50_text'

#  python MultiplexNetwork/main.py --reg_coef 0.000435 --sup_coef 0.2  --lr 0.00118 --nb_epochs 200 --l2_coef 2e-06 --drop_prob 0.3842 --hid_units 217 --dataset 'FairClip_resnet50_text' --contrastive_coef 0.1