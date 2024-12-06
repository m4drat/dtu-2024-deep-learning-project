import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


def compute_confusion_matrix(target, pred, normalize=None):
    return metrics.confusion_matrix(
        target.detach().cpu().numpy(),
        pred.detach().cpu().numpy(),
        normalize=normalize
    )

def plot_confusion_matrix(cm, classes):
    def normalize(matrix, axis):
        axis = {'true': 1, 'pred': 0}[axis]
        return matrix / matrix.sum(axis=axis, keepdims=True)

    x_labels = [classes[i] for i in classes]
    y_labels = x_labels
    plt.figure(figsize=(6, 6))
    sns.heatmap(ax=plt.gca(), data=normalize(cm, 'true'), annot=True, linewidths=0.5, cmap="Reds", cbar=False, fmt=".2f", xticklabels=x_labels, yticklabels=y_labels,)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.tight_layout()
    plt.show()
    
def accuracy(target, pred):
    return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())

def evaluate_model(model, dataset_loader, device, classes):
    confusion_matrix = np.zeros((len(classes), len(classes)))
    test_accuracy = 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        model.eval()
        test_accuracies = []
        for inputs, targets in dataset_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            
            predictions = output.max(1)[1]
            test_accuracies.append(accuracy(targets, predictions) * len(inputs))
                
            confusion_matrix += compute_confusion_matrix(targets, predictions)
            
            # AUC
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(output[:, 1].cpu().numpy())  # assume class 1 is the positive class

        test_accuracy = 100* np.sum(test_accuracies) / len(dataset_loader.dataset)
        print(f"Test accuracy: {test_accuracy:.3f}")
        
        auc = metrics.roc_auc_score(all_targets, all_predictions)
        print(f"AUC: {auc:.3f}")
    
    return confusion_matrix