import torch
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, one_hot_from_int
import matplotlib.pyplot as plt
import numpy as np

from typing import List


def extract_conv_layers(model: BigGAN) -> List[torch.nn.Conv2d]:
    """
    Extracts all convolutional layers from the model recursively.
    """
    conv_layers = []

    def extract_conv_layers_recursive(module):
        for layer in module.children():
            if isinstance(layer, torch.nn.Conv2d):
                conv_layers.append(layer)
            elif isinstance(layer, torch.nn.Module):
                extract_conv_layers_recursive(layer)

    extract_conv_layers_recursive(model)

    return conv_layers


def main():
    # Step 1: Load the pretrained BigGAN model
    model = BigGAN.from_pretrained("biggan-deep-256")

    print(extract_conv_layers(model))

    # Step 2: Generate a class vector
    # Example: Generate a one-hot vector for the "dog" class (class 207 in ImageNet)
    class_vector = one_hot_from_int(
        [207], batch_size=1
    )  # Class 207 corresponds to 'Labrador retriever'
    class_vector = torch.from_numpy(class_vector)

    # Step 3: Generate random noise vector
    # Use truncated normal noise to ensure better image quality
    truncation = 0.4  # Controls image quality and diversity
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)
    noise_vector = torch.from_numpy(noise_vector)

    # Step 4: Generate an image using the model
    # Ensure inputs are in the correct device and data type
    class_vector = class_vector.to(torch.float32)
    noise_vector = noise_vector.to(torch.float32)

    with torch.no_grad():  # No gradient computation needed for inference
        generated_images = model(noise_vector, class_vector, truncation)

    # Step 5: Post-process the generated image
    # BigGAN outputs a tensor with values in the range [-1, 1]. Rescale to [0, 1].
    generated_images = (generated_images + 1) / 2.0
    generated_images = generated_images.clamp(0, 1)  # Ensure values are in [0, 1]

    # Convert tensor to a NumPy array for visualization
    image_array = generated_images[0].permute(1, 2, 0).cpu().numpy()

    # Step 6: Display the generated image
    plt.imshow(image_array)
    plt.axis("off")  # Remove axis for better visualization
    plt.title("Generated Image (Class: Dog)")
    plt.show()


if __name__ == "__main__":
    main()
