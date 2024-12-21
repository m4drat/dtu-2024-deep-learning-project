import pandas as pd
import os
import kagglehub
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

manjilkarki_path = "manjilkarki/deepfake-and-real-images"
path_140k_real_fake = "xhlulu/140k-real-and-fake-faces"
alaaeddineayadi_path = "alaaeddineayadi/real-vs-ai-generated-faces"

class ImageDataset(Dataset):
    """
    A class to represent dataset:
    https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
    """
    
    def __init__(self, path, csv_file, transform=None):
        self.data = pd.read_csv(os.path.join(path, csv_file))
        self.transform = transform
        self.path = path
        self._labels = {'real': 1, 'fake': 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.get_image_path(idx)
        label = self._labels[self.data.iloc[idx, 4]] # column 4 corresponds to the label (fake or real)


        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def get_image_path(self, idx):
        i_path = os.path.join(self.path, 'real_vs_fake')
        i_path = os.path.join(i_path, 'real-vs-fake')
        return os.path.join(i_path, self.data.iloc[idx, 5]) # column 5 corresponds to the path



def get_manjilkarki_deep_fake_real_dataset(transform):
    """
    Returns train, validation and test datasets from https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
    """
    path = kagglehub.dataset_download(manjilkarki_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(path, 'Dataset', 'Train'), transform=transform)
    valid_dataset = datasets.ImageFolder(root=os.path.join(path, 'Dataset', 'Validation'), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(path, 'Dataset', 'Test'), transform=transform)

    return train_dataset, valid_dataset, test_dataset


def get_xhlulu_140k_real_and_fake_dataset(transform):
    """
    Returns train, validation and test datasets from https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
    """
    path = kagglehub.dataset_download(path_140k_real_fake)
    train_dataset = ImageDataset(path, 'train.csv', transform=transform)
    valid_dataset = ImageDataset(path, 'valid.csv', transform=transform)
    test_dataset = ImageDataset(path, 'test.csv', transform=transform)

    return train_dataset, valid_dataset, test_dataset


def get_alaaeddineayadi_real_vs_fake_dataset(transform):
    """
    Returns train, validation and test datasets from https://www.kaggle.com/datasets/alaaeddineayadi/real-vs-ai-generated-faces
    """ 
    path = kagglehub.dataset_download(alaaeddineayadi_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(path, 'dataset', 'train'), transform=transform)
    valid_dataset = datasets.ImageFolder(root=os.path.join(path, 'dataset', 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(path, 'dataset', 'test'), transform=transform)

    return train_dataset, valid_dataset, test_dataset
