"""OpenCV-based image preprocessing and augmentation utilities."""

import torch
import cv2
import numpy as np
import random


class RandomDilation:
    """Randomly apply a small dilation operation to an image tensor.

    Args:
        p: Probability of applying dilation.
    """

    def __init__(self, p: float = 0.3):
        """Initialize the random dilation transform.

        Args:
            p: Probability of applying dilation.
        """

        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply dilation with probability ``p``.

        Args:
            img: Image tensor accepted by OpenCV after conversion to NumPy.

        Returns:
            Original image tensor or dilated image tensor.
        """

        if random.random() > self.p:
            return img

        img = img.numpy()
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        return torch.Tensor(img)


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        # img is expected to be a float tensor in [0, 1]
        noise = torch.randn_like(img) * self.std + self.mean
        return torch.clamp(img + noise, 0.0, 1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def denoise(img, method="median"):
    """Denoise an image with a selected OpenCV blur method.

    Args:
        img: Image array to denoise.
        method: Denoising method. Supported values are ``"median"`` and
            ``"gaussian"``.

    Returns:
        Denoised image array, or the original image when the method is unknown.
    """

    if method == "median":
        return cv2.medianBlur(img, 3)
    elif method == "gaussian":
        return cv2.GaussianBlur(img, (3, 3), 0)
    return img


def deskew(img):
    """Deskew an image by estimating the foreground orientation.

    Args:
        img: RGB image array.

    Returns:
        Deskewed image array. If too few foreground pixels are found, the
        original image is returned.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    coords = np.column_stack(np.where(gray > 0))

    if len(coords) < 10:
        return img

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    return cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def binarize(img, threshold=127):
    """Convert an RGB image to a binary grayscale image.

    Args:
        img: RGB image array.
        threshold: Pixel threshold used for binary conversion.

    Returns:
        Binary grayscale image array.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def thinning(img):
    """Skeletonize foreground strokes in an RGB image.

    Args:
        img: RGB image array.

    Returns:
        Thinned binary image array.

    Raises:
        ImportError: If OpenCV was installed without the ``ximgproc`` module.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    if hasattr(cv2, "ximgproc"):
        return cv2.ximgproc.thinning(binary)

    raise ImportError("cv2.ximgproc not available. Install opencv-contrib-python.")
