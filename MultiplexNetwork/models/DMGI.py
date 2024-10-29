# Code based on https://github.com/pcy1302/DMGI/blob/master/models/DMGI.py

import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from embedder import embedder
from layers import GCN, Discriminator, Attention
import numpy as np
np.random.seed(0)
import pandas as pd
import wandb
from evaluate import evaluate
from models import LogReg
import pickle as pkl
from tqdm import trange
import matplotlib.pyplot as plt

import os
# Set the environment variable for PyTorch CUDA memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

class DMGI(embedder):
    def __init__(self, args):
        super(DMGI, self).__init__(args)
        self.args = args
        self.temperature = args.temperature  # Temperature parameter for contrastive loss

    def training(self):
        torch.cuda.empty_cache()

        print(f'\nTraining on {self.args.device} device')
        features = [feature.to(self.args.device) for feature in self.features]

        adj = [adj_.to(self.args.device) for adj_ in self.adj]
        if self.args.use_const_loss:
            model_name = f'{self.args.dataset}_{self.args.embedder}_cont_loss_'
 
        else:
            model_name = f'{self.args.dataset}_{self.args.embedder}'

        model = modeler(self.args).to(self.args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
        
        # # Define a scheduler learning rate 
        T_max = self.args.nb_epochs // 2     
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0, last_epoch=-1)

        b_xent = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for discriminator
        xent = nn.CrossEntropyLoss()     # Cross-entropy for supervised semi-learning

        contrastive_losses = []          # Track contrastive loss during training
        contrastive_losses_v =[]
        train_losses = []
        val_losses = []

        # Initialize variables for early stopping
        best = float('inf')
        cnt_wait = 0
        # print(torch.cuda.memory_summary())

        grad_norms = {}
        for epoch in trange(self.args.nb_epochs):
            model.train()
            optimizer.zero_grad()

            # Shuffle node indices for negative sampling -and apply these shuffled features (shuf) as inputs for the model, which provides negative samples for contrastive learning. This helps create the positive and negative pairs necessary for contrastive loss calculation.
            idx = np.random.permutation(self.args.nb_nodes)
            shuf = [feature[:, idx, :] for feature in features]
            shuf = [shuf_ft.to(self.args.device) for shuf_ft in shuf]

            lbl_1 = torch.ones(self.args.batch_size, self.args.nb_nodes)
            lbl_2 = torch.zeros(self.args.batch_size, self.args.nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.args.device)

            result = model(features, adj, shuf, self.args.sparse, None, None, None)
            logits = result['logits']
            
            # Compute Binary Cross-Entropy Loss for each view
            xent_loss = sum([b_xent(logit, lbl) for logit in logits])
            total_loss = xent_loss

            # Add Regularization Loss
            reg_loss = result['reg_loss'] #is meant to optimize the embeddings' consistency across different graphs, indirectly enforcing consensus.
            total_loss += self.args.reg_coef * reg_loss
            
            # ----------------------------------------------------------------------------------
            # --------------------------- Compute Contrastive Loss -----------------------------
            # ----------------------------------------------------------------------------------
            if self.args.use_const_loss:
                contrastive_loss_value = result['contrastive_loss']
                #print('\ncontrastive_loss_value:\n', contrastive_loss_value)           
                total_loss += self.args.contrastive_coef *contrastive_loss_value 
                contrastive_losses.append(contrastive_loss_value.item())

            # Add Semi-Supervised Loss if applicable
            if self.args.isSemi:
                """ 
                In semi-supervised learning, the model is primarily learning from both labeled and unlabeled data,
                but the labeled data (used for supervised loss) only comes from the training set.
                """
                sup = result['semi']
                semi_loss = xent(sup[self.idx_train], self.train_lbls)
                total_loss += self.args.sup_coef * semi_loss
            
            if self.args.use_const_loss:
                print(f'Epoch {epoch + 1}/{self.args.nb_epochs}, Semi Loss: {semi_loss.item()} , Reg Loss: {reg_loss.item()}, Cross Entropy Loss: {xent_loss.item()}, Const Loss:{contrastive_loss_value.item()}')
            else:
                print(f'Epoch {epoch + 1}/{self.args.nb_epochs}, Semi Loss: {semi_loss.item()} , Reg Loss: {reg_loss.item()}, Cross Entropy Loss: {xent_loss.item()}')
            # -------------------- validation ----------------
            # Validation loss
            model.eval()
            with torch.no_grad():
                val_loss = 0
                val_result = model(features, adj, shuf, self.args.sparse, None, None, None)
                val_logits = val_result['logits']
                val_xent_loss = sum([b_xent(logit, lbl) for logit in val_logits])
                val_reg_loss = val_result['reg_loss']
                val_loss += val_xent_loss + self.args.reg_coef * val_reg_loss
                if self.args.isSemi:
                    val_sup = val_result['semi']
                    val_semi_loss = xent(val_sup[self.idx_val], self.val_lbls)
                    val_loss += self.args.sup_coef * val_semi_loss
                
                if self.args.use_const_loss:
                    val_cont_loss_value = val_result['contrastive_loss'] 
                    val_loss += self.args.contrastive_coef *val_cont_loss_value 
                    contrastive_losses_v.append(val_cont_loss_value.item())

                val_losses.append(val_loss.item())             
            
            # --------------- train step ------------------    
            # Backpropagation and optimizer step
            total_loss.backward()
            
            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=8)
            
            optimizer.step()
            # Update learning rate scheduler
            # scheduler.step()
            
            # --------------- logging --------------------

                  
            # Log losses and learning rate to W&B
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": total_loss.item(),
                "train_xent_loss": xent_loss.item(),
                "train_reg_loss": reg_loss.item(),
                "train_sup_loss": semi_loss.item(),
                "val_loss": val_loss.item(),
                # "val_xent_loss": val_xent_loss.item(),
                # "val_reg_loss": val_reg_loss.item(),
                "learning_rate": scheduler.get_last_lr()[0]  # Log current learning rate
            }, step=epoch + 1)

            if self.args.use_const_loss:
                wandb.log({
                    "train_contrastive_loss": contrastive_loss_value.item(),
                    "val_contrastive_loss": val_cont_loss_value.item(),
                }, step=epoch + 1)
            total_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)  # Calculate the L2 norm of gradients
                    grad_norms[f'grad_norm/{name}'] = param_norm.item()
                    total_norm += param_norm.item() ** 2

            wandb.log(grad_norms)

            # Logging
            if (epoch + 1) % 5 == 0:
                #print(f'Epoch {epoch + 1}/{self.args.nb_epochs}, Loss: {total_loss.item()}, Val Loss: {val_loss.item()}.\nTrain contrastive loss: {contrastive_loss_value.item()}, Val contrastive loss: {contrastive_loss_value_v.item()}')
                print(f'Epoch {epoch + 1}/{self.args.nb_epochs}, Loss: {total_loss.item()}, Val Loss: {val_loss.item()}')

                total_norm = total_norm ** 0.5
                print(f'Total gradient norm: {total_norm}')             

            # Early stopping: Save best model
            if total_loss.item() < best:
                best = total_loss.item()
                cnt_wait = 0
                torch.save(model.state_dict(), 'MultiplexNetwork/saved_model/best_{}{}{}.pkl'.format(self.args.dataset, self.args.embedder, self.args.metapaths))
            else:
                cnt_wait += 1

            # Stop training if patience limit is reached
            if cnt_wait == self.args.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            train_losses.append(total_loss.item())
            
            torch.cuda.empty_cache()
                # Create a DataFrame
        train_losses_data = pd.DataFrame({'train_loss': train_losses})

        # Save to CSV 
        train_losses_data.to_csv(f'MultiplexNetwork/saved_ROC/train_losses_{model_name}.csv', index=False)
           
        self.plot_loss_curve(train_losses, val_losses, 'Total Loss')
        
        if self.args.use_const_loss:
            self.plot_loss_curve(contrastive_losses, contrastive_losses_v, 'Contrastive Loss')
        
        # Load the best model after training
        model.load_state_dict(torch.load(f'MultiplexNetwork/saved_model/best_{self.args.dataset}{self.args.embedder}{self.args.metapaths}.pkl'))
        print("\nEvaluation mode on ....\n")

        # Evaluation
        model.eval()
        mean_auc_val = evaluate(model.H.data.detach(), self.idx_train, self.idx_val, self.idx_test, self.labels, self.args.device, model_name,isTest=False)
        evaluate(model.H.data.detach(), self.idx_train, self.idx_val, self.idx_test, self.labels, self.args.device,model_name)
        
        return  mean_auc_val
    
    def plot_loss_curve(self, train_loss, val_loss, loss_title):
        # Plotting loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label=f"Training - {loss_title}")
        plt.plot(val_loss, label=f"Validation - {loss_title}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"DMGI {loss_title} Curve")
        plt.savefig(f'DMGI {loss_title} Curve.png')
        return
class modeler(nn.Module):
    '''
     GCN transforms the input features into embeddings, which are continuous, high-dimensional representations of the data.
    '''
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
        if self.args.use_const_loss:
            cluster_labels = pd.read_csv('/home/noreena/repos/Multimodal-Medical/MultiplexNetwork/data/cluster_labels_resnet50.csv', header=None)
            self.cluster_labels = cluster_labels
        self.gcn = nn.ModuleList([GCN(args.ft_size, args.hid_units, args.activation, args.drop_prob, args.isBias) for _ in range(args.nb_graphs)])

        self.disc = Discriminator(args.hid_units)

        # Initialize self.H as a trainable parameter
        self.H = nn.Parameter(torch.FloatTensor(1, args.nb_nodes, args.hid_units))
        nn.init.xavier_normal_(self.H)
        #nn.init.xavier_uniform_(self.H)
        #nn.init.kaiming_normal_(self.H, mode='fan_in', nonlinearity='relu')

        self.readout_func = self.args.readout_func
        if args.isAttn:
            self.attn = nn.ModuleList([Attention(args) for _ in range(args.nheads)])

        if args.isSemi:
            self.logistic = LogReg(args.hid_units, args.nb_classes).to(args.device)
    
    def contrastive_loss(self, embeddings, cluster_labels):
        '''
        Calculates a similarity matrix based on the provided embeddings and supervised cluster labels. 
        Positive pairs are those that share the same cluster label, and negative pairs are those with different labels.
        '''
        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix and convert to a 0-1 range
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        #similarity_matrix = (similarity_matrix + 1) / 2  # Convert cosine similarity range from [-1, 1] to [0, 1]

        # Convert cluster labels to PyTorch tensor
        cluster_labels_ = cluster_labels.values.squeeze()
        cluster_labels_tensor = torch.tensor(cluster_labels_, dtype=torch.long).to(self.args.device)

        # Create label matrix and mask for the upper triangle, excluding diagonal
        label_matrix = (cluster_labels_tensor.unsqueeze(1) == cluster_labels_tensor.unsqueeze(0)).float()
        upper_tri_mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()

        # Extract positive and negative pairs from the upper triangle
        positives = similarity_matrix[label_matrix.bool() & upper_tri_mask]
        negatives = similarity_matrix[~label_matrix.bool() & upper_tri_mask]

        # Combine positive and negative logits, scale by temperature
        logits = torch.cat([positives, negatives], dim=0) / self.args.temperature

        # Create contrastive labels: 1 for positive pairs, 0 for negative pairs
        contrastive_labels = torch.cat([
            torch.ones(positives.size(0), dtype=torch.long).to(self.args.device),   # Label 1 for positive pairs
            torch.zeros(negatives.size(0), dtype=torch.long).to(self.args.device)   # Label 0 for negative pairs
        ])

        # Use Binary Cross-Entropy for contrastive loss on normalized logits
        contrastive_loss = F.binary_cross_entropy_with_logits(logits, contrastive_labels.float())

        return contrastive_loss

    def forward(self, feature, adj, shuf, sparse, msk, samp_bias1, samp_bias2):
        h_1_all = []; h_2_all = []; c_all = []; logits = []
        result = {}

        for i in range(self.args.nb_graphs):
            # True node embeddings
            h_1 = self.gcn[i](feature[i], adj[i], sparse) #  processes the input feature and adjacency matrix for each graph
            
            # Graph Sammery
            c = self.readout_func(h_1)  # the readout function aggregates the node embeddings, which can be considered a form of generating a consensus representation
            c = self.args.readout_act_func(c)
            
            # Corrupted node embeddings
            h_2 = self.gcn[i](shuf[i], adj[i], sparse)

            logit = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
            h_1_all.append(h_1)
            h_2_all.append(h_2)
            c_all.append(c)
            logits.append(logit)
            
        result['logits'] = logits

        if self.args.isAttn:
            h_1_all_lst = []; h_2_all_lst = []; c_all_lst = []
            for h_idx in range(self.args.nheads):
                h_1_all_, h_2_all_, c_all_, p = self.attn[h_idx](h_1_all, h_2_all, c_all)
                h_1_all_lst.append(h_1_all_); h_2_all_lst.append(h_2_all_); c_all_lst.append(c_all_)
            h_1_all = torch.mean(torch.cat(h_1_all_lst, 0), 0).unsqueeze(0)
            h_2_all = torch.mean(torch.cat(h_2_all_lst, 0), 0).unsqueeze(0)
        else:
            h_1_all = torch.mean(torch.cat(h_1_all), 0).unsqueeze(0)
            h_2_all = torch.mean(torch.cat(h_2_all), 0).unsqueeze(0)
        
        # normalize for the calculation
        H_normalized = F.normalize(self.H, p=2, dim=2)
        h_1_all_normalized = F.normalize(h_1_all, p=2, dim=2)
        h_2_all_normalized = F.normalize(h_2_all, p=2, dim=2)

        # Calculate the positive and negative regularization losses using the normalized versions
        pos_reg_loss = ((H_normalized - h_1_all_normalized) ** 2).sum()
        neg_reg_loss = ((H_normalized - h_2_all_normalized) ** 2).sum()
        
        # Calculate the positive and negative regularization losses
        # pos_reg_loss = ((self.H - h_1_all) ** 2).sum()
        # neg_reg_loss = ((self.H - h_2_all) ** 2).sum()

        if 0:        
            print("self.H mean:", self.H.mean().item(), "std:", self.H.std().item())
            print("h_1_all mean:", h_1_all.mean().item(), "std:", h_1_all.std().item())
            print("h_2_all mean:", h_2_all.mean().item(), "std:", h_2_all.std().item())
            # print('concensus:', self.H)
            # print('pos:', h_1_all)
            # print('neg:', h_2_all)
            print('pos_reg_loss:', pos_reg_loss)
            print('neg_reg_loss:', neg_reg_loss)

        reg_loss = pos_reg_loss - neg_reg_loss
        result['reg_loss'] = reg_loss

        if self.args.isSemi:
            semi = self.logistic(self.H).squeeze(0)
            # semi = self.logistic(h_1_all).squeeze(0)
            result['h1'] = h_1_all
            result['semi'] = semi
        
        if self.args.use_const_loss:
            contrastive_loss = self.contrastive_loss(h_1_all.squeeze(0), self.cluster_labels)
            #contrastive_loss = self.contrastive_loss(self.H.squeeze(0), self.cluster_labels)
            result['contrastive_loss'] = contrastive_loss

        return result
