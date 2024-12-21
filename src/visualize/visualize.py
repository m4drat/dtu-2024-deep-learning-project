import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from BigGAN import Discriminator  # Make sure to import the BigGAN Discriminator class
from matplotlib.colors import Normalize

# Global variable for storing activations
activations = None

# Define the model architecture (same as used during training)
class DiscriminatorFromPretrainedBigGAN(nn.Module):
    def __init__(self):
        super(DiscriminatorFromPretrainedBigGAN, self).__init__()
        discriminator = load_biggan_discriminator()

        # Freeze all layers.
        for param in discriminator.parameters():
            param.requires_grad = False

        # Remove the last layer.
        self.discriminator_layers = nn.Sequential(
            *recursively_flatten_modules(discriminator)[:-2]
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.bottleneck = nn.Linear(1536, 256)

        # Add a classifier layer.
        self.classifier = nn.Linear(256, 1)

        # Register hook for Grad-CAM
        self.target_layer = self.discriminator_layers[-1]  # Last layer in the model
        self.target_layer.register_forward_hook(hook_fn)

    def forward(self, x):
        for layer in self.discriminator_layers:
            x = layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.classifier(x)
        return x

# Hook function to capture activations
def hook_fn(module, input, output):
    global activations
    activations = output

# Helper to load BigGAN discriminator
def load_biggan_discriminator():
    path = "weights/138k/"  # Replace with the path where your weights are stored
    d_state_dict = torch.load(path + "D.pth")
    D = Discriminator(D_ch=96, skip_init=True)
    D.load_state_dict(d_state_dict)
    return D

# Helper to recursively flatten the model layers
def recursively_flatten_modules(module):
    flat_modules = []
    for submodule in module.children():
        if isinstance(submodule, (nn.ModuleList, nn.Sequential)):
            flat_modules.extend(recursively_flatten_modules(submodule))
        else:
            flat_modules.append(submodule)
    return flat_modules

# Generate Grad-CAM heatmap
def generate_gradcam(model, image_tensor, target_class=None):
    global activations
    
    # Ensure the model and image are on the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = image_tensor.to(device)
    image_tensor = Variable(image_tensor, requires_grad=True)
    model = model.to(device)
    
    # Forward pass
    output = model(image_tensor)
    label = torch.sigmoid(output)  # If you have a binary classification, or use softmax for multi-class.
    
    print(f"Probability: {label}")
    
    # If no target_class provided, take the class with max probability
    if target_class is None:
        target_class = torch.argmax(label, dim=1)

    # Backward pass to get gradients for the target class
    model.zero_grad()
    output[0][target_class].backward(retain_graph=True)

    # Get the gradients and activations
    gradients = torch.autograd.grad(output[0][target_class], activations, retain_graph=True)[0]

    # Pool the gradients over all the channels (take the mean of gradients)
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Get the activations from the last convolutional layer
    activation_map = activations[0]

    # Multiply the activations by the pooled gradients
    for i in range(activation_map.shape[0]):
        activation_map[i, :, :] *= pooled_gradients[i]

    # Average the activations across the channels
    heatmap = torch.mean(activation_map, dim=0)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
    heatmap /= np.max(heatmap)

    # Resize the heatmap to match the input image size
    heatmap = cv2.resize(heatmap, (image_tensor.shape[2], image_tensor.shape[3]))
    
    return heatmap

# Visualization function
def show_heatmap(image, heatmap):
    # Convert image tensor to numpy array
    image = image.cpu().detach().numpy().transpose(1, 2, 0)
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    # Apply heatmap to the image
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    superimposed_img = heatmap + np.float32(image) / 255
    superimposed_img = superimposed_img / np.max(superimposed_img)
        # Create figure and axis for better control
    fig, ax = plt.subplots()
    # Plot the image and heatmap
    plt.imshow(superimposed_img)
    plt.axis('off')
        # Manually create a new axis for the colorbar
    # The [0.8, 0.1, 0.05, 0.8] is the position of the colorbar [left, bottom, width, height]
    cax = fig.add_axes([0.85, 0.1, 0.05, 0.8])  # Adjust these values to change the position

    # Create a ScalarMappable object and use it for the colorbar
    norm = Normalize(vmin=0, vmax=1)  # Normalize the heatmap values to [0, 1] for color mapping
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])  # Empty array for ScalarMappable

    # Add colorbar to the new custom axis
    cbar = fig.colorbar(sm, cax=cax)  # Use the custom cax (colorbar axis)
    cbar.set_label('Heatmap Intensity')  # Label for the colorba
    plt.savefig('./outputs/output.png')
    plt.show()

# Load your model's state_dict and initialize the model
def load_model(model_path):
    model = DiscriminatorFromPretrainedBigGAN()  # Define your model
    model.load_state_dict(torch.load(model_path),strict=False)  # Load the trained state_dict
    model.eval()  # Set to evaluation mode
    return model

# Main function to load the image and generate the Grad-CAM heatmap
def main():
    # Load the model with its weights
    model = load_model('./model_weights/best_model_weights.8005')  # Provide path to your trained model state_dict

    # Load the image and apply necessary transformations
    image_path = './imgs/fake_1006.jpg'  # Provide the path to the image for which you want to generate Grad-CAM
    image = Image.open(image_path)

    # Apply the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Generate Grad-CAM heatmap
    heatmap = generate_gradcam(model, image_tensor)

    # Show the image with the heatmap overlay
    show_heatmap(image_tensor[0], heatmap)

if __name__ == "__main__":
    main()
