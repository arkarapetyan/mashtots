import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_class_distributions(labels: torch.Tensor) -> plt.Figure:
    """Creates Matplotlib bar plot of class distributions.

    Args:
        labels: Labels of the Training Set samples as torch Tensor.

    Returns:
        matplotlib figure of the class distributions.
    """

    classes, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(classes.astype(str), counts)
    ax.set_title("Training Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    return fig


def plot_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor) -> plt.Figure:
    """Creates Matplotlib plot of a Confusion Matrix.

    Args:
        y_true: True Labels as torch Tensor.
        y_pred: Predicted Labels as torch Tensor.

    Returns:
        matplotlib figure of the class distributions.
    """

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Validation Confusion Matrix")
    return fig
