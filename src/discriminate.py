#!/usr/bin/python3

from typing import Dict
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from BigGAN import Discriminator
from utils import Distribution
from PIL import Image


def load():
    path = "weights/138k/"
    d_state_dict = torch.load(path + "D.pth")
    D = Discriminator(D_ch=96, skip_init=True)
    D.load_state_dict(d_state_dict)
    return D


def hook_specific_layers(model: Discriminator, activations: Dict[str, torch.Tensor]):
    """
    Hook the second-to-last and last layers to capture their activations.
    """

    def hook_fn(module, input, output):
        activations[module.__class__.__name__] = output

    # Assuming BigGAN's discriminator has a known architecture
    layer_names = ["$linear", "$embed"]  # Replace with actual layer names or indices
    for name, layer in model.named_modules():
        print(f"Layer name: ${name}")
        if name in layer_names:
            layer.register_forward_hook(hook_fn)


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(x)


def preprocess_image(image_path):
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]

    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )

    img = Image.open(image_path)
    input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    return input_tensor.cuda()


def main():
    D = load()
    activations = {}

    # Hook specific layers
    hook_specific_layers(D, activations)

    D.eval()
    D.cuda()

    # Load and preprocess the image
    img_path = "../generated-images/image_0.png"
    input_tensor = preprocess_image(img_path)

    y_ = Distribution(torch.zeros(1, requires_grad=False))
    y_.init_distribution("categorical", num_categories=1000)
    y_ = y_.to("cuda", torch.int64, non_blocking=False, copy=False)
    y_.sample_()

    # Forward pass through the discriminator
    with torch.no_grad():
        D(input_tensor, y_[:1])

    # Combine activations from the second-to-last and last layers
    second_to_last = activations["Linear"].flatten(start_dim=1)
    last = activations["Embedding"].flatten(start_dim=1)
    combined_features = torch.cat((second_to_last, last), dim=1)

    # Define the simple classifier
    input_dim = combined_features.shape[1]
    classifier = SimpleClassifier(input_dim).cuda()

    # Example prediction
    prediction = classifier(combined_features)
    print("Prediction (Real=1, Fake=0):", prediction.item())


if __name__ == "__main__":
    main()
