"""Datasets for Mashtots character recognition."""

import numpy as np
import pandas as pd
import torch


from typing import Optional, Any, Sequence
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image

IMAGE_SIZE = 64
"""Height and width of every Mashtots image."""

LABEL_COLUMNS = ("label", "Label", "category", "Category", "target")
"""Column names recognized as class labels in the CSV split."""

ID_COLUMNS = ("id", "Id", "ID")
"""Column names recognized as sample ids in the CSV split."""


class MashtotsDataset(Dataset):
    """PyTorch dataset for the Mashtots Armenian character dataset.

    The dataset reads either training images from class-named directories or
    test/prediction samples from the CSV file included with the downloaded
    dataset. Image values are scaled to the ``[0, 1]`` range before transforms
    are applied.

    The CSV is read by column name: an id column (one of ``ID_COLUMNS``) and a
    label column (one of ``LABEL_COLUMNS``) are used when present, and every
    remaining column must be one of the ``64 x 64`` pixels. Unlabeled samples
    get the label ``-1``.

    Args:
        data_dir: Root directory containing the extracted Mashtots dataset.
        train: If ``True``, load images from ``Train/Train``. If ``False``,
            load flattened image rows from ``new_test/new_test.csv``.
        transform: Optional transform pipeline applied to each sample.

    Attributes:
        ids: Sample ids from the CSV id column, or ``None`` when absent.
        has_labels: Whether real labels are available.
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
        self.data_dir = Path(data_dir)
        self.train = train
        self.transform = transform
        self.ids: Optional[list] = None
        self.has_labels = True

        if self.train:
            self.inputs, self.labels = self._load_train_data()
        else:
            self.inputs, self.labels = self._load_test_data()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return one transformed sample and its label.

        Args:
            idx: Integer index of the sample to retrieve.

        Returns:
            A tuple containing the transformed image tensor and its label.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.load_image(idx)
        if self.transform:
            x = self.transform(x)
        return x, self.labels[idx]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""

        return len(self.labels)

    def load_image(self, idx: int) -> torch.Tensor:
        """Return the untransformed image for ``idx`` scaled to ``[0, 1]``.

        Args:
            idx: Index of the sample.

        Returns:
            Image tensor with shape ``(1, 64, 64)``.
        """

        if self.train:
            return read_image(str(self.inputs[idx])).float().div_(255)
        return self.inputs[idx]

    def _load_train_data(self) -> tuple[list[Path], list[int]]:
        """Load training image paths and class labels.

        Training data is expected under ``data_dir/Train/Train`` with one
        subdirectory per class. Each subdirectory name must be convertible to an
        integer label.

        Returns:
            A tuple of image paths and integer labels.
        """

        train_dir = self.data_dir / "Train" / "Train"

        inputs: list[Path] = []
        labels: list[int] = []
        for cls_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
            cls_inputs = sorted(p for p in cls_dir.iterdir() if p.is_file())
            inputs.extend(cls_inputs)
            labels.extend([int(cls_dir.name)] * len(cls_inputs))

        return inputs, labels

    def _load_test_data(self) -> tuple[torch.Tensor, list[int]]:
        """Load test or prediction samples from the dataset CSV.

        Returns:
            Image tensors with shape ``(N, 1, 64, 64)`` and labels (``-1``
            when the CSV has no label column).

        Raises:
            ValueError: If the pixel columns do not form ``64 x 64`` images.
        """

        csv_path = self.data_dir / "new_test" / "new_test.csv"
        frame = pd.read_csv(csv_path)

        id_column = next((c for c in ID_COLUMNS if c in frame.columns), None)
        label_column = next((c for c in LABEL_COLUMNS if c in frame.columns), None)

        if id_column is not None:
            self.ids = frame[id_column].tolist()
        if label_column is not None:
            labels = frame[label_column].astype(int).tolist()
        else:
            labels = [-1] * len(frame)
            self.has_labels = False

        pixels = frame.drop(columns=[c for c in (id_column, label_column) if c])
        if pixels.shape[1] != IMAGE_SIZE * IMAGE_SIZE:
            raise ValueError(
                f"{csv_path} has {pixels.shape[1]} pixel columns after removing "
                f"id/label columns; expected {IMAGE_SIZE * IMAGE_SIZE}. Recognized "
                f"id columns: {ID_COLUMNS}, label columns: {LABEL_COLUMNS}."
            )

        inputs = torch.from_numpy(pixels.to_numpy(dtype=np.float32))
        inputs = inputs.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE).div_(255)
        return inputs, labels


class TransformedSubset(Dataset):
    """A subset of a dataset with its own transform.

    Lets the train, validation and test splits share one underlying
    :class:`MashtotsDataset` (and one directory scan) while applying
    different transforms.

    Args:
        dataset: Underlying dataset; its own transform is ignored.
        indices: Indices of the samples in the subset.
        transform: Transform applied to each image, or ``None``.
    """

    def __init__(
        self,
        dataset: MashtotsDataset,
        indices: Sequence[int],
        transform: Optional[Any] = None,
    ):
        """Initialize the subset. See the class docstring for the arguments."""

        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return one transformed sample and its label."""

        source_idx = self.indices[idx]
        x = self.dataset.load_image(source_idx)
        if self.transform:
            x = self.transform(x)
        return x, self.dataset.labels[source_idx]

    def __len__(self) -> int:
        """Return the number of samples in the subset."""

        return len(self.indices)
