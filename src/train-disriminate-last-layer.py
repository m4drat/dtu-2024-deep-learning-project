#!/usr/bin/python3

from typing import Dict
from matplotlib import pyplot as plt
import torch
import torch.utils
import torch.utils.data
import time
from torchvision import transforms
from BigGAN import Discriminator
from utils import Distribution
from torchvision import datasets, models, transforms
from PIL import Image
import os
import torch
from tempfile import TemporaryDirectory
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


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

    def forward(self, x):
        for layer in self.discriminator_layers:
            x = layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.classifier(x)
        return x


def recursively_flatten_modules(module):
    """
    Recursively flattens nested nn.Module containers like nn.ModuleList or nn.Sequential.
    Args:
        module (nn.Module): The root module to flatten.
    Returns:
        List[nn.Module]: A flat list of all submodules.
    """
    flat_modules = []
    for submodule in module.children():
        if isinstance(submodule, (nn.ModuleList, nn.Sequential)):
            # If the submodule is a container, recursively flatten it
            flat_modules.extend(recursively_flatten_modules(submodule))
        else:
            # If it's a regular layer, add it to the list
            flat_modules.append(submodule)
    return flat_modules


def load_biggan_discriminator():
    path = "weights/138k/"
    d_state_dict = torch.load(path + "D.pth")
    D = Discriminator(D_ch=96, skip_init=True)
    D.load_state_dict(d_state_dict)
    return D


def train_model(
    model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=16
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        preds = (outputs > 0).long()
                        labels = labels.view(-1, 1).float()
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val Acc: {best_acc:4f}")

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

    # Save the model
    torch.save(
        model.state_dict(),
        f"best_model_params{time.time()}-{num_epochs}-{best_acc:.4f}",
    )
    return model


def convert_to_rgb(image):
    return image.convert("RGB")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]

    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.Lambda(convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )

    # Define data directory paths
    train_dir = "../datasets/2/real_vs_fake/real-vs-fake/train"
    val_dir = "../datasets/2/real_vs_fake/real-vs-fake/valid"

    # Load train and validation datasets using ImageFolder
    train_dataset = datasets.ImageFolder(root=train_dir, transform=preprocess)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=preprocess)

    subset_ratio = 1
    indices = {
        "train": torch.randperm(len(train_dataset))[
            : int(subset_ratio * len(train_dataset))
        ],
        "val": torch.randperm(len(val_dataset))[: int(subset_ratio * len(val_dataset))],
    }
    train = torch.utils.data.Subset(train_dataset, indices["train"])
    val = torch.utils.data.Subset(val_dataset, indices["val"])

    # Create dataloaders for train and validation
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=16, shuffle=True, num_workers=8
    )
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=16, shuffle=False, num_workers=8
    )

    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": len(train), "val": len(val)}

    print("Dataset sizes: ", dataset_sizes)

    model = DiscriminatorFromPretrainedBigGAN().to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(
        model,
        criterion,
        optimizer,
        exp_lr_scheduler,
        dataloaders,
        dataset_sizes,
        num_epochs=16,
    )


if __name__ == "__main__":
    main()
