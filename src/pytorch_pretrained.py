import torch
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, one_hot_from_int
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Dict


def hook_conv_layers(
    model: BigGAN, activations: Dict[torch.nn.Module, torch.Tensor] | None = None
):
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
    # Step 1: Load the pretrained BigGAN model
    model = BigGAN.from_pretrained("biggan-deep-256")
    activations = {}

    hook_conv_layers(model, activations)

    # Step 2: Generate a class vector
    class_vector = one_hot_from_int([207], batch_size=1)
    class_vector = torch.from_numpy(class_vector)

    # Step 3: Generate random noise vector
    truncation = 0.4
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)
    noise_vector = torch.from_numpy(noise_vector)

    # Step 4: Generate an image using the model
    class_vector = class_vector.to(torch.float32)
    noise_vector = noise_vector.to(torch.float32)

    with torch.no_grad():
        generated_images = model(noise_vector, class_vector, truncation)

    # Step 5: Visualize activations for some convolutional layers
    indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 52]
    activations = [list(activations.items())[idx] for idx in indices]

    num_layers = len(activations)
    fig, axes = plt.subplots(
        num_layers, 8, figsize=(10, num_layers * 2)
    )  # Create a grid of subplots

    for row_idx, (layer, activation) in enumerate(activations):
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
