import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

from sklearn import metrics
from PIL import Image


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
            all_predictions.extend(predictions.cpu().numpy())

        test_accuracy = 100* np.sum(test_accuracies) / len(dataset_loader.dataset)
        print(f"Test accuracy: {test_accuracy:.3f}")
        
        auc = metrics.roc_auc_score(all_targets, all_predictions)
        print(f"AUC: {auc:.3f}")
    
    return confusion_matrix


def get_image(path, transform):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, transform(Image.fromarray(img)).unsqueeze(0)


def generate_gradcam(model, img, img_tensor, target_layer, target_class = 0):

    activations = []
    gradients = []
    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)


    # Perform the forward pass
    model.eval() # Set the model to evaluation mode
    output = model(img_tensor)
    pred_class = output.max(1)[1]
    print(pred_class)

    # Zero the gradients
    model.zero_grad()

    # Backward pass to compute gradients
    output[0][target_class].backward()

    weights = torch.mean(gradients[0], dim=[2, 3], keepdim=True)

    heatmap = torch.sum(weights * activations[0], dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap = heatmap.cpu().detach().numpy()
    
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / heatmap.max()

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

