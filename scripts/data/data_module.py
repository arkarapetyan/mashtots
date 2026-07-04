import torch
import numpy as np
import lightning as L
import torchvision.transforms as T

from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .kaggle_download import download_and_extract_dataset
from .dataset import MashtotsDataset
from ..utils.reproducibility import make_torch_generator, seed_worker
from ..utils.transforms import AddGaussianNoise


class MashtotsDataModule(L.LightningDataModule):
    """Lightning data module for Mashtots training, validation, and inference.

    The module downloads the dataset when needed, creates train/validation
    splits for the ``fit`` stage, and exposes data loaders for Lightning's
    ``fit``, ``test``, and ``predict`` workflows.

    Args:
        data_dir: Directory where the dataset is or will be stored.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes used by each data loader.
        seed: Seed used for splitting, undersampling, shuffling, and worker RNGs.
        pin_memory: Whether data loaders should pin CPU memory before transfer
            to accelerator devices.
        **kwargs: Additional configuration values. Currently supports
            ``val_ratio`` for the validation split ratio.
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 128,
        num_workers: int = 8,
        seed: int = 42,
        pin_memory: bool = False,
        **kwargs,
    ):
        """Initialize the data module.

        Args:
            data_dir: Directory where the dataset is or will be stored.
            batch_size: Number of samples per batch.
            num_workers: Number of worker processes used by data loaders.
            seed: Seed used for reproducible data operations.
            pin_memory: Whether data loaders should pin CPU memory.
            **kwargs: Additional data options, including ``val_ratio``.
        """

        super(MashtotsDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.train_transform = T.Compose(
            [
                T.RandomAffine(
                    degrees=10,  # slight rotation
                    translate=(0.1, 0.1),  # shift
                    scale=(0.9, 1.1),  # zoom
                    shear=5,  # slight shear
                ),
                T.ElasticTransform(alpha=1.0, sigma=0.1),  # distorts strokes naturally
                AddGaussianNoise(mean=0, std=0.02),
                T.Normalize((0.0334,), (0.1035,)),
            ]
        )
        self.inference_transform = T.Compose(
            [
                T.Normalize((0.0334,), (0.1035,)),
            ]
        )
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self._kwargs = kwargs

    def _dataloader_generator(self) -> torch.Generator:
        """Return a fresh seeded generator for deterministic loader behavior."""

        return make_torch_generator(self.seed)

    def prepare_data(self) -> None:
        """Download and extract the dataset if ``data_dir`` does not exist."""

        if self.data_dir.exists():
            return

        self.data_dir.mkdir()
        download_and_extract_dataset(
            "mashtots-dataset-v2",
            self.data_dir,
        )

    def train_dataloader(self) -> DataLoader:
        """Create the training data loader.

        Returns:
            Data loader over the augmented training subset.
        """

        return DataLoader(
            self.mashtots_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=self.pin_memory,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=self._dataloader_generator(),
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation data loader.

        Returns:
            Data loader over the normalization-only validation subset.
        """

        return DataLoader(
            self.mashtots_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=self.pin_memory,
            worker_init_fn=seed_worker,
            generator=self._dataloader_generator(),
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test data loader.

        Returns:
            Data loader over samples loaded from the test CSV.
        """

        return DataLoader(
            self.mashtots_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=self.pin_memory,
            worker_init_fn=seed_worker,
            generator=self._dataloader_generator(),
        )

    def predict_dataloader(self) -> DataLoader:
        """Create the prediction data loader.

        Returns:
            Data loader over samples loaded from the prediction CSV.
        """

        return DataLoader(
            self.mashtots_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=seed_worker,
            generator=self._dataloader_generator(),
        )

    def setup(self, stage: str) -> None:
        """Set up datasets for a Lightning stage.

        Args:
            stage: Lightning stage name. Supported values are ``"fit"``,
                ``"test"``, and ``"predict"``.

        Raises:
            ValueError: If ``stage`` is not one of the supported stage names.
        """

        match stage:
            case "fit":
                self._setup_train()
            case "test":
                self._setup_test()
            case "predict":
                self._setup_predict()
            case _:
                raise ValueError(f"Invalid DataModule Stage Setup Argument: {stage}")

    def _setup_train(self) -> None:
        """Create training and validation datasets with a shared random split."""
        val_ratio = self._kwargs.get("val_ratio", 0.2)
        full = MashtotsDataset(self.data_dir, train=True, transform=None)
        labels = np.asarray(full.labels)
        all_indices = np.arange(len(labels))
        train_indices, val_indices = train_test_split(
            all_indices,
            test_size=val_ratio,
            stratify=labels,
            random_state=self.seed,
        )
        self.mashtots_train = MashtotsDataset(
            self.data_dir,
            train=True,
            transform=self.train_transform,
        )
        self.mashtots_val = MashtotsDataset(
            self.data_dir,
            train=True,
            transform=self.inference_transform,
        )
        self.mashtots_train = torch.utils.data.Subset(
            self.mashtots_train,
            train_indices,
        )
        self.mashtots_val = torch.utils.data.Subset(
            self.mashtots_val,
            val_indices,
        )

    def _setup_test(self) -> None:
        """Create the test dataset."""

        self.mashtots_test = MashtotsDataset(
            self.data_dir,
            train=False,
            transform=self.inference_transform,
        )

    def _setup_predict(self) -> None:
        """Create the prediction dataset."""

        self.mashtots_predict = MashtotsDataset(
            self.data_dir,
            train=False,
            transform=self.inference_transform,
        )
