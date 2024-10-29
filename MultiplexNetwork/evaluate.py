import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score
from models import LogReg
import torch.nn as nn


def evaluate(embeds, idx_train, idx_val, idx_test, labels, device,model_name, isTest=True):
    hid_units = embeds.shape[2]
    nb_classes = labels.shape[2]
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    accs = []
    aucs = []
    recalls = []
    macro_f1s = []
    test_logits_all = []
    test_lbls_all = []

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.to(device)

        val_accs = []
        val_aucs = []
        val_recalls = []
        val_macro_f1s = []

        test_accs = []
        test_aucs = []
        test_recalls = []
        test_macro_f1s = []

        for iter_ in range(50):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            loss.backward()
            opt.step()

            # val
            log.eval()
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = accuracy_score(val_lbls.cpu(), preds.cpu())
            val_recall = recall_score(val_lbls.cpu(), preds.cpu(), average='binary')
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')

            # Calculate AUC-ROC for the validation set
            logits_proba = torch.softmax(logits, dim=1)[:, 1]
            val_auc = roc_auc_score(val_lbls.detach().cpu().numpy(), logits_proba.detach().cpu().numpy())

            val_accs.append(val_acc)
            val_macro_f1s.append(val_f1_macro)
            val_recalls.append(val_recall)
            val_aucs.append(val_auc)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = accuracy_score(test_lbls.cpu(), preds.cpu())
            test_recall = recall_score(test_lbls.cpu(), preds.cpu(), average='binary')
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')

            # Calculate AUC-ROC for the test set
            logits_proba_test = torch.softmax(logits, dim=1)[:, 1]
            test_auc = roc_auc_score(test_lbls.detach().cpu().numpy(), logits_proba_test.detach().cpu().numpy())

            test_accs.append(test_acc)
            test_macro_f1s.append(test_f1_macro)
            test_recalls.append(test_recall)
            test_aucs.append(test_auc)

            test_logits_all.append(logits_proba_test.detach().cpu().numpy())
            test_lbls_all.append(test_lbls.detach().cpu().numpy())

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        aucs.append(test_aucs[max_iter])
        recalls.append(test_recalls[max_iter])
        macro_f1s.append(test_macro_f1s[max_iter])

    if isTest:
        print("\t[Classification] Accuracy: {:.4f} ({:.4f}) | AUC: {:.4f} ({:.4f}) | Sensitivity (Recall): {:.4f} ({:.4f}) | Macro-F1: {:.4f} ({:.4f})".format(
            np.mean(accs), np.std(accs),
            np.mean(aucs), np.std(aucs),
            np.mean(recalls), np.std(recalls),
            np.mean(macro_f1s), np.std(macro_f1s)
        ))
        print("\t[Maximums] Accuracy: {:.4f} | AUC: {:.4f} | Sensitivity: {:.4f} | Macro-F1: {:.4f}".format(
            np.max(accs), np.max(aucs), np.max(recalls), np.max(macro_f1s)
        ))

        # Plot the ROC curve
        test_lbls_flat = np.concatenate(test_lbls_all)
        test_logits_flat = np.concatenate(test_logits_all)

        fpr, tpr, _ = roc_curve(test_lbls_flat, test_logits_flat)
        # Create a DataFrame with fpr and tpr values
        roc_data = pd.DataFrame({'fpr': fpr, 'tpr': tpr})

        # Save to CSV
        csv_path = f'MultiplexNetwork/saved_ROC/roc_data_{model_name}.csv'
        roc_data.to_csv(csv_path, index=False)
        
        # plt.figure()
        # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {np.mean(aucs):.4f})')
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic')
        # plt.legend(loc="lower right")

        # # Save the ROC plot
        # plt.savefig('roc_curve.png')
        # plt.show()

    else:
        return np.mean(val_aucs)

    test_embs = np.array(test_embs.cpu())
    test_lbls = np.array(test_lbls.cpu())
    return np.mean(aucs)
