import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def load_mnist():
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def subset(train_data, test_data):
    train_subset = Subset(train_data, range(2000))
    test_subset = Subset(test_data, range(500))

    # Load the full subset into memory
    train_images, train_labels = next(iter(DataLoader(train_subset, batch_size=len(train_subset))))
    test_images, test_labels = next(iter(DataLoader(test_subset, batch_size=len(test_subset))))

    return train_images, train_labels, test_images, test_labels

def flatten(train_images, train_labels, test_images, test_labels):
    X_train = train_images.view(len(train_images), -1).numpy()
    X_test = test_images.view(len(test_images), -1).numpy()

    y_train = train_labels.numpy()
    y_test = test_labels.numpy()

    return X_train, X_test, y_train, y_test


def pca_data(n_features):
    train_data, test_data = load_mnist()

    train_images, train_labels, test_images, test_labels = subset(train_data, test_data)

    X_train, X_test, y_train, y_test = flatten(train_images, train_labels, test_images, test_labels)

    pca = PCA(n_components=n_features)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    explained_variance = np.sum(pca.explained_variance_ratio_)    
    print(f"Explained variance: {explained_variance:.2%}")


pca_data(784)











