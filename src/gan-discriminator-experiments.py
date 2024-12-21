import os
import kagglehub
import warnings
import torch

import sys
sys.path.append('../')

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# from сnn.tools import evaluate_model, plot_confusion_matrix
from сnn.datasets import get_manjilkarki_deep_fake_real_dataset, get_xhlulu_140k_real_and_fake_dataset, get_alaaeddineayadi_real_vs_fake_dataset

# train-disriminate-last-layer.py
from train_disriminate_last_layer import DiscriminatorFromPretrainedBigGAN, evaluate_model, save_confusion_matrix

def evaluate():
    HEIGHT, WIDTH = 256, 256

    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]

    def convert_to_rgb(image):
        return image.convert("RGB")

    preprocess = transforms.Compose(
        [
            transforms.Resize((HEIGHT, WIDTH)),
            transforms.Lambda(convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
    # Suppress all warnings
    warnings.filterwarnings("ignore")

    print("Evaluating the model on manjilkarki/deepfake-and-real-images")

    _, _, test_dataset = get_manjilkarki_deep_fake_real_dataset(preprocess)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    classes = {index: name for name, index in test_dataset.class_to_idx.items()}

    # Instantiate the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiscriminatorFromPretrainedBigGAN().to(device)

    state_dict = torch.load("best_model_params1733195785.9681547-16-0.8005", map_location=device)
    model.load_state_dict(state_dict)

    print("Model loaded successfully")

    save_confusion_matrix(evaluate_model(model, test_loader, device, classes), classes, "confusion_matrix-manjilkarki.png")

    ### Style Gan Dataset 140k

    _, _, test_dataset = get_xhlulu_140k_real_and_fake_dataset(preprocess)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    save_confusion_matrix(evaluate_model(model, test_loader, device, classes), classes, "confusion_matrix-140k.png")

    ### Some random dataset
    _, _, test_dataset = get_alaaeddineayadi_real_vs_fake_dataset(preprocess)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    save_confusion_matrix(evaluate_model(model, test_loader, device, classes), classes, "confusion_matrix-alaaeddineayadi.png")

if __name__ == "__main__":
    evaluate()
