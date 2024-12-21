#!/usr/bin/python3

from typing import Dict
from matplotlib import pyplot as plt
import torch
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


def hook_conv_layers(model: Discriminator, activations=None):
    """
    Hook the forward pass of all convolutional layers in the model to extract their activations.
    """

    if activations is None:
        activations = {}

    def hook_fn(module, input, output):
        activations[module] = output

    def hook_conv_layers_recursive(module):
        for layer in module.children():
            if isinstance(layer, torch.nn.Conv2d):
                print(layer)
                layer.register_forward_hook(hook_fn)
            elif isinstance(layer, torch.nn.Module):
                hook_conv_layers_recursive(layer)

    hook_conv_layers_recursive(model)


def main():
    D = load()
    activations = {}

    hook_conv_layers(D, activations)

    D.eval()
    D.cuda()

    # Load the image we want to discriminate
    img = Image.open("../generated-images/image_0.png")

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

    input_tensor = preprocess(img).cuda()

    # Convert to batch
    input_tensor = input_tensor.unsqueeze(0)

    y_ = Distribution(torch.zeros(1, requires_grad=False))
    y_.init_distribution("categorical", num_categories=1000)
    y_ = y_.to("cuda", torch.int64, non_blocking=False, copy=False)
    y_.sample_()

    # Discriminate the image
    with torch.no_grad():
        result = D(input_tensor, y_[:1])
        # result = D(input_tensor)
        print(result)

    num_layers = len(activations)
    fig, axes = plt.subplots(
        num_layers, 8, figsize=(10, num_layers * 2)
    )  # Create a grid of subplots

    for row_idx, (layer, activation) in enumerate(activations.items()):
        activation = activation[0].cpu().numpy()  # Get first example in batch
        num_channels = activation.shape[0]
        num_to_plot = min(num_channels, 8)  # Ensure at most 8 activations per layer

        # Plot 8 activations as a single row
        for col_idx in range(8):
            if col_idx < num_to_plot:
                ax = axes[row_idx, col_idx]
                ax.imshow(activation[col_idx], cmap="viridis")
                ax.axis("off")
            else:
                axes[row_idx, col_idx].axis("off")  # Hide unused subplots in the row

        # Add a title to the first subplot in each row
        axes[row_idx, 0].set_title(f"Layer {row_idx + 1}")

    plt.tight_layout()
    plt.savefig("activations.png", dpi=700)
    plt.show()


if __name__ == "__main__":
    main()