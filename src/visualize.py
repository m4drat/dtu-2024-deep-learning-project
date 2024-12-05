import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
import cv2
import numpy as np
from train_disriminate_last_layer import *


# Function to load the model
def load_model(model_path):
    model = torch.load(model_path,weights_only=False)
    model.eval()
    return model

# Function to preprocess input image (adjust according to your use case)
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

# Hook the feature extractor to get the activations
def hook_fn(module, input, output):
    global activations
    activations = output

# Grad-CAM Implementation
def generate_gradcam(model, image_tensor, target_class=None):
    # Hook the feature extractor of the model to get activations
    target_layer = model.discriminator_layers[6].conv2  # Change depending on your model architecture
    target_layer.register_forward_hook(hook_fn)

    # Forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = image_tensor.to(device) 
    image_tensor = Variable(image_tensor, requires_grad=True)
    model = model.to(device)
    output = model(image_tensor)

    if target_class is None:
        target_class = torch.argmax(output)

    # Backward pass to get gradients
    model.zero_grad()
    output[0][target_class].backward(retain_graph=True)

    # Get the gradients from the last convolutional layer
    gradients = torch.autograd.grad(output[0][target_class], activations,retain_graph=True)[0]

    # Pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)

    # Weight the channels by the gradients
    weighted_activations = activations * pooled_gradients

    # Generate heatmap by averaging over the channels
    heatmap = weighted_activations.mean(dim=1).squeeze()

    # Apply ReLU to the heatmap (to remove negative values)
    heatmap = F.relu(heatmap)

    # Normalize the heatmap
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / heatmap.max()

    return heatmap

# Function to superimpose the heatmap on the image
def overlay_heatmap(image_path, heatmap):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize heatmap to the size of the input image
    heatmap = cv2.resize(heatmap.detach().cpu().numpy(), (img.shape[1], img.shape[0]))

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Plot the original image and heatmap
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()
    plt.savefig('./sam-vis-weights/output.png')

# Main function to load the model, generate Grad-CAM, and visualize the results
def main(model_path, image_path):
    model = load_model(model_path)
    print(model)
    image_tensor = preprocess_image(image_path)

    heatmap = generate_gradcam(model, image_tensor)

    overlay_heatmap(image_path, heatmap)

# Example Usage
model_path = './sam-vis-weights/best_model_params_whole.pth'
image_path = './imgs/fake-face.jpg'

main(model_path, image_path)
