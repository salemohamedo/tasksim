import torch
import numpy as np
from scipy.sparse.linalg import eigsh as largest_eigsh


def subspace_similarity(X, Y, k=10, centered=True):
    if centered:
        X = X - np.mean(X, 0, keepdims=True)
        Y = Y - np.mean(Y, 0, keepdims=True)
        XtX = (X.T).dot(X) / (X.shape[0]-1)
        YtY = (Y.T).dot(Y) / (Y.shape[0]-1)
    else:
        XtX = (X.T).dot(X)
        YtY = (Y.T).dot(Y)

    _, X_eig = largest_eigsh(XtX, k, which='LM')
    _, Y_eig = largest_eigsh(YtY, k, which='LM')

    
    X_Y = (X_eig.T).dot(Y_eig)
    subspace_overlap = (1/k)*(np.linalg.norm(X_Y, ord='fro') ** 2)

    return subspace_overlap



def get_features(model, dataloader, num_samples=10000):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for x, y, _ in dataloader:
            x = x.to(model.device)
            _x = model(x)
            if len(_x.shape)==2:
                features.append(_x.detach().cpu())
            else:
                features.append(_x.squeeze().detach().cpu())
            labels.append(y)

    features = torch.cat(features)
    labels = torch.cat(labels)

    unique_labels = torch.unique(labels)
    num_classes = len(unique_labels)
    num_samples_per_class = num_samples // num_classes
    features = []
    for y in unique_labels:
        X_ = features[labels == y]
        idx = torch.randperm(X_.shape[0])
        X_ = X_[idx]

        X_ = X_ if X_.shape[0] <= num_samples_per_class else X_[:num_samples_per_class]
        features.append(X_)
    features = torch.cat(features)

    return features.numpy()


def trace_similarity(X, Y):
    XY = X.dot(Y.T)
    XX = X.dot(X.T)
    YY = Y.dot(Y.T)
    
    XY_ = XY.dot(XY.T)
    XX_ = XX.dot(XX.T)
    YY_ = YY.dot(YY.T)

    trace_sim = np.trace(XY_) / np.sqrt(np.trace(XX_)*np.trace(YY_))

    return trace_sim



