"""Dataset and Lightning data module for Mashtots character recognition."""

import torch
import pandas as pd


from typing import Optional, Any, Callable
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image


class MashtotsDataset(Dataset):
    """PyTorch dataset for the Mashtots Armenian character dataset.

    The dataset reads either training images from class-named directories or
    test/prediction samples from the CSV file included with the downloaded
    dataset. Image values are scaled to the ``[0, 1]`` range before transforms
    are applied.

    Args:
        data_dir: Root directory containing the extracted Mashtots dataset.
        train: If ``True``, load images from ``Train/Train``. If ``False``,
            load flattened image rows from ``new_test/new_test.csv``.
        transform: Optional transform pipeline applied to each sample. When no
            transform is provided, a normalization-only transform is used.
    """

    def __init__(
        self,
        data_dir: Path,
        train: bool,
        transform: Optional[Any] = None,
    ):
        """Initialize the dataset and load sample metadata.

        Args:
            data_dir: Root directory containing the extracted Mashtots dataset.
            train: If ``True``, load training image paths. If ``False``, load
                samples from the test CSV.
            transform: Optional transform pipeline applied in ``__getitem__``.
        """

        super(MashtotsDataset, self).__init__()
        self.data_dir = data_dir
        self.train = train

        self.transform = transform

        if self.train:
            self.load_data: Callable[..., tuple[list[Path], list[torch.Tensor]]] = (
                self._load_train_data
            )
            self.get_item_input: Callable[[int], torch.Tensor] = self._get_train_item
        else:
            self.load_data = self._load_test_data
            self.get_item_input = self._get_test_item

        self.inputs, self.labels = self.load_data()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one transformed sample and its label.

        Args:
            idx: Integer index of the sample to retrieve.

        Returns:
            A tuple containing the transformed image tensor and its label.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.get_item_input(idx)
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of available labels, matching the number of samples.
        """

        return len(self.labels)

    def _load_train_data(self) -> tuple[list[Path], list[torch.Tensor]]:
        """Load training image paths and class labels.

        Training data is expected under ``data_dir/Train/Train`` with one
        subdirectory per class. Each subdirectory name must be convertible to an
        integer label.

        Returns:
            A tuple of image paths and labels.
        """

        train_dir = self.data_dir / "Train" / "Train"

        inputs = []
        labels = []
        for cls_dir in sorted(train_dir.iterdir()):
            cls_inputs = sorted(cls_dir.iterdir())
            cls_labels = torch.LongTensor([int(cls_dir.name)] * len(cls_inputs))
            inputs.extend(cls_inputs)
            labels.extend(cls_labels)

        return inputs, labels

    def _load_test_data(self) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Load test or prediction samples from the dataset CSV.

        The CSV is expected to contain flattened ``64 x 64`` grayscale images.
        If the final column can be interpreted as labels, it is returned as the
        target tensor. Otherwise, zero labels are created for unlabeled
        prediction data.

        Returns:
            A tuple containing image tensors with shape ``(N, 1, 64, 64)`` and
            labels.
        """

        test_path = self.data_dir / "new_test" / "new_test.csv"
        raw_data = torch.Tensor(pd.read_csv(test_path).to_numpy())

        try:
            inputs = raw_data[:, :-1].reshape(-1, 1, 64, 64).div_(255)
            labels = torch.LongTensor(raw_data[:, -1].flatten())
        except RuntimeError:
            inputs = raw_data.reshape(-1, 1, 64, 64).div_(255)
            labels = torch.zeros(len(inputs))
        return inputs, labels

    def _get_train_item(self, idx: int) -> torch.Tensor:
        """Read and scale a training image.

        Args:
            idx: Index of the image path in ``self.inputs``.

        Returns:
            Image tensor scaled to the ``[0, 1]`` range.
        """

        img_path = self.inputs[idx]
        img = read_image(img_path).float().div_(255)
        return img

    def _get_test_item(self, idx: int) -> torch.Tensor:
        """Return a preloaded test or prediction image tensor.

        Args:
            idx: Index of the tensor in ``self.inputs``.

        Returns:
            Image tensor for the requested index.
        """

        return self.inputs[idx]
