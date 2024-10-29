import logging
import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
import wandb
from utils import save_config_file, accuracy, save_checkpoint
import numpy as np

from medical_aug import medical_aug 
torch.manual_seed(0)

# class SimCLR_micle(object): # takes model, train it with contrastive loss

class SimCLR_micle(torch.nn.Module):

    def __init__(self, model, args, device):
        super(SimCLR_micle, self).__init__()
        self.model = model.to(device)
        self.args = args
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss().to(device)
        self.device = device
        # Initialize lists to track loss values
        self.epoch_losses = []
        self.micle_losses = []
        
    def info_nce_loss(self, features):
        '''
        This method implements the contrastive loss as per the SimCLR framework. 
        Input:
            features: These are the feature representations of the images output by the model, after they have been encoded into latent space.
        '''
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1) #------------------ modify for biomedCLIP
        
        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.args.temperature
        return logits, labels
    
    def micle(self, features, indx): #additional loss component, - needs to be compatible with the output features of BioMedCLIP.
        index = indx.detach().numpy()
        unique, counts = np.unique(index, return_counts=True)
        count_dict = dict(zip(unique, counts))
        loss = torch.zeros(1).to(self.device)
        loss = torch.zeros(1).to(self.device)
        count = 0
        len_dict = 0
        for key in count_dict:
            len_dict +=1
            if count_dict[key]>2:
                which = np.where(index == key)[0]
                mask = torch.tensor(which).to(self.device)
                mask = torch.tensor(which).to(self.device)

                features_ = features[mask]
                features_ = F.normalize(features_, dim=1)

                similarity_matrix = torch.matmul(features_, features_.T)
                similarity_matrix = F.normalize(similarity_matrix)
                mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool).to(self.device)
                mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool).to(self.device)
                positive = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

                labels = torch.ones(positive.shape,dtype=torch.long).to(self.device)
                labels = torch.ones(positive.shape,dtype=torch.long).to(self.device)

                loss += self.mse(positive.to(torch.float32),labels.to(torch.float32)) #The loss is accumulated across all image instances, and the total loss is normalized by the number of unique images with more than 2 instances.
                count +=1
        if not (count==0):
            loss =loss/count
        return loss

    def train(self, train_loader, optimizer, scheduler, val_loader=None):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        n_iter = 0

        best_val_loss = float('inf')  # Track the best validation loss
        patience_counter = 0  # Counts how long since the last improvement

        early_stop_patience = 10  # Number of epochs with no improvement before stopping
        early_stop_delta = 0  # Minimum change in the monitored quantity to qualify as improvement

        for epoch_counter in range(self.args.epochs):
            self.model.train()
            for images, labels, _ in tqdm(train_loader):
                # Apply medical_aug batch-wise instead of per image
                augmented_images = torch.cat([medical_aug(image).unsqueeze(0) for image in images], dim=0)
                images = torch.cat((images, augmented_images), dim=0)  # Concatenate augmented images
                labels = torch.cat((labels, labels), dim=0)  # Concatenate labels for augmented images
                images = images.to(self.device)

                # Forward pass once to avoid redundant computations
                with autocast(device_type=self.device.type, enabled=self.args.fp16_precision):
                    features = self.model(images)
                    
                    # Compute both SimCLR and MICLe losses
                    logits, simclr_labels = self.info_nce_loss(features)
                    simclr_loss = self.criterion(logits, simclr_labels)
                    
                    micle_loss = self.micle(features, labels)

                # Total loss: SimCLR loss + MICLe loss
                train_loss = simclr_loss + micle_loss

                # Zero gradients, backpropagate, and update the optimizer
                optimizer.zero_grad()

                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                n_iter += 1

            # Scheduler step after each epoch
            if epoch_counter >= 10:
                scheduler.step()
            
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                #if epoch_counter % 10 == 0:
                print(f"Epoch [{epoch_counter}], Training Loss: {train_loss.item()}, Validation Loss: {val_loss}")
                wandb.log({
                    "epoch": epoch_counter,
                    "Training loss": train_loss.item(),
                    "Micle Training loss": micle_loss.item(),
                    "SimCLR Training loss": simclr_loss.item(),
                    "SimCLR Validation loss": val_loss,
                }, step=epoch_counter)
                #wandb.log({"Epoch": epoch_counter, "Total Training loss": train_loss.item(), "Micle Training loss": micle_loss.item(), "SimCLR Training loss": simclr_loss.item(), "SimCLR Validation loss": val_loss})

                # Check if the validation loss improved
                if val_loss < best_val_loss - early_stop_delta:
                    best_val_loss = val_loss  # Update the best loss
                    patience_counter = 0  # Reset patience counter
                    # Optionally, save the best model
                    torch.save(self.model.state_dict(), 'best_model_.pth')
                else:
                    patience_counter += 1

                # Early stopping if no improvement in validation loss for 'early_stop_patience' epochs
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch_counter}. Best validation loss: {best_val_loss}")
                    break

            else:
                val_loss = None
                #if epoch_counter % 10 == 0:
                print(f"Epoch [{epoch_counter}], Loss: {train_loss.item()}")
                # Log losses and learning rate to W&B
                wandb.log({
                    "epoch": epoch_counter,
                    "Training loss": train_loss.item(),
                    "Micle Training loss": micle_loss.item(),
                    "SimCLR Training loss": simclr_loss.item(),
                }, step=epoch_counter)

        return train_loss, val_loss

    def evaluate(self, val_loader):
        self.model.eval()  #  Set model to evaluation mode
        total_loss = 0.0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                # Apply medical_aug batch-wise instead of per image for validation
                augmented_images = torch.cat([medical_aug(image).unsqueeze(0) for image in images], dim=0)
                images = torch.cat((images, augmented_images), dim=0)  # Concatenate augmented images
                labels = torch.cat((labels, labels), dim=0)  # Concatenate labels for augmented images
                images = images.to(self.device)

                # Forward pass to get embeddings
                features = self.model(images)

                # Calculate InfoNCE loss (contrastive loss)
                logits, simclr_labels = self.info_nce_loss(features)
                loss = self.criterion(logits, simclr_labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def linear_evaluation(self, train_loader, val_loader):
        self.model.eval()  # Freeze the model and only train the linear classifier
        train_embeddings = []
        train_labels = []
        val_embeddings = []
        val_labels = []

        # Collect embeddings and labels from training data
        with torch.no_grad():
            for images, labels, _ in train_loader:
                images = images.to(self.device)
                embeddings = self.model(images)
                train_embeddings.append(embeddings.cpu().numpy())
                train_labels.append(labels.cpu().numpy())

        # Collect embeddings and labels from validation data
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                embeddings = self.model(images)
                val_embeddings.append(embeddings.cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        # Convert to numpy arrays
        train_embeddings = np.concatenate(train_embeddings, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        val_embeddings = np.concatenate(val_embeddings, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)

        # Train a logistic regression classifier on the frozen embeddings
        clf = LogisticRegression(max_iter=1000)
        clf.fit(train_embeddings, train_labels)

        # Evaluate the classifier on the validation embeddings
        val_predictions = clf.predict(val_embeddings)
        val_probabilities = clf.predict_proba(val_embeddings)[:, 1]  # Get probabilities for the positive class if binary
        
        accuracy = accuracy_score(val_labels, val_predictions)
        auc = roc_auc_score(val_labels, val_probabilities)

        print(f"Linear Evaluation Accuracy: {accuracy * 100:.2f}%")
        print(f"Linear Evaluation AUC: {auc:.4f}")

        wandb.log({"Accuracy": accuracy, "AUC": auc})
        
        return accuracy