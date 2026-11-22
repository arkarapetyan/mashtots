"""Image augmentation transforms and OpenCV preprocessing helpers."""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import random

from torchvision.transforms import InterpolationMode


class FastElasticTransform:
    """Elastic warp, a faster drop-in for ``torchvision.transforms.ElasticTransform``.

    Like torchvision, it draws a uniform random displacement field, smooths it
    with a Gaussian of width ``sigma`` pixels, scales it by ``alpha``, and
    resamples the image along it. torchvision smooths with a full 2-D kernel
    (41 x 41 taps for ``sigma=5``); a Gaussian is separable, so this class
    uses a horizontal and a vertical 1-D pass instead. The result is the same
    up to float rounding, at a fraction of the CPU cost, which matters because
    the elastic warp was the slowest step of the data loader.

    Args:
        alpha: Strength of the displacement.
        sigma: Smoothness of the displacement field in pixels.
        interpolation: Resampling mode.
        fill: Value for pixels sampled from outside the image.
    """

    def __init__(
        self,
        alpha: float = 50.0,
        sigma: float = 5.0,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: float = 0.0,
    ):
        """Initialize the transform. See the class docstring for the arguments."""

        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.interpolation = interpolation
        self.fill = [float(fill)]

        # Same kernel size and weights as torchvision's gaussian_blur.
        kernel_size = int(8 * self.sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        half = (kernel_size - 1) * 0.5
        x = torch.linspace(-half, half, steps=kernel_size)
        kernel = torch.exp(-0.5 * (x / self.sigma) ** 2)
        kernel = kernel / kernel.sum()
        self._kernel_x = kernel.view(1, 1, 1, kernel_size)
        self._kernel_y = kernel.view(1, 1, kernel_size, 1)
        self._pad = kernel_size // 2

    def get_displacement(self, height: int, width: int) -> torch.Tensor:
        """Draw a smooth random displacement field.

        Args:
            height: Image height in pixels.
            width: Image width in pixels.

        Returns:
            Displacement of shape ``(1, height, width, 2)`` in normalized
            coordinates, as expected by ``F.elastic_transform``.
        """

        field = torch.rand(2, 1, height, width) * 2 - 1  # dx and dy
        pad = self._pad
        field = F.conv2d(F.pad(field, (pad, pad, 0, 0), mode="reflect"), self._kernel_x)
        field = F.conv2d(F.pad(field, (0, 0, pad, pad), mode="reflect"), self._kernel_y)
        dx = field[0] * self.alpha / height  # torchvision scales dx by height
        dy = field[1] * self.alpha / width
        return torch.cat([dx, dy], dim=0).permute(1, 2, 0).unsqueeze(0)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Warp an image tensor.

        Args:
            img: Image tensor with shape ``(C, H, W)``.

        Returns:
            Warped image with the same shape.
        """

        height, width = img.shape[-2:]
        return TF.elastic_transform(
            img,
            self.get_displacement(height, width),
            self.interpolation,
            self.fill,
        )

    def __repr__(self) -> str:
        """Return a readable description for logged transform pipelines."""

        return f"{self.__class__.__name__}(alpha={self.alpha}, sigma={self.sigma})"


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
