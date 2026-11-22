"""Lightning data module for the Mashtots dataset."""

from typing import Any, Optional

import numpy as np
import lightning as L
import torchvision.transforms as T

from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .kaggle_download import download_and_extract_dataset
from .dataset import MashtotsDataset, TransformedSubset
from ..utils.reproducibility import make_torch_generator, seed_worker
from ..utils.transforms import AddGaussianNoise, FastElasticTransform

PIXEL_MEAN = 0.0334
"""Mean pixel value of the training images (see EDA.ipynb)."""

PIXEL_STD = 0.1035
"""Pixel standard deviation of the training images (see EDA.ipynb)."""

DEFAULT_AUGMENTATION: dict[str, Any] = {
    "degrees": 10.0,
    "translate": (0.1, 0.1),
    "scale": (0.95, 1.05),
    "shear": 10.0,
    "elastic_alpha": 40.0,
    "elastic_sigma": 5.0,
    "elastic_p": 0.5,
    "noise_std": 0.02,
}
"""Training augmentation defaults; see ``build_train_transform``."""


COMPETITION_NAME = "mashtots-dataset-v2"
"""Kaggle competition the dataset is downloaded from."""


def is_dataset_ready(data_dir: Path) -> bool:
    """Check that ``data_dir`` holds the extracted dataset, not just the folder.

    Args:
        data_dir: Dataset root directory.

    Returns:
        ``True`` when the training images and the prediction CSV are present.
    """

    train_dir = data_dir / "Train" / "Train"
    test_csv = data_dir / "new_test" / "new_test.csv"
    return (
        train_dir.is_dir()
        and any(p.is_dir() for p in train_dir.iterdir())
        and test_csv.is_file()
    )


def build_train_transform(**augmentation: Any) -> T.Compose:
    """Build the training augmentation pipeline.

    Args:
        **augmentation: Overrides for ``DEFAULT_AUGMENTATION``:

            * ``degrees``, ``translate``, ``scale``, ``shear``: passed to
              ``RandomAffine``. The scale range is kept narrow by default
              because letter size is one of the few cues that separates some
              upper/lowercase pairs.
            * ``elastic_alpha``, ``elastic_sigma``, ``elastic_p``: smooth
              elastic warp of the strokes, applied with probability
              ``elastic_p``. torchvision blurs a random displacement field with
              a Gaussian of width ``elastic_sigma`` (pixels) and scales it by
              ``elastic_alpha``; ``alpha=40, sigma=5`` moves strokes by roughly
              2-3 pixels at 64x64. ``elastic_p=0`` disables it.
            * ``noise_std``: additive Gaussian pixel noise. ``0`` disables it.

    Returns:
        Transform for ``[0, 1]`` image tensors ending with normalization.
    """

    params = {**DEFAULT_AUGMENTATION, **augmentation}
    transforms: list[Any] = [
        T.RandomAffine(
            degrees=params["degrees"],
            translate=tuple(params["translate"]),
            scale=tuple(params["scale"]),
            shear=params["shear"],
        )
    ]
    if params["elastic_p"] > 0 and params["elastic_alpha"] > 0:
        transforms.append(
            T.RandomApply(
                [
                    FastElasticTransform(
                        alpha=float(params["elastic_alpha"]),
                        sigma=float(params["elastic_sigma"]),
                    )
                ],
                p=params["elastic_p"],
            )
        )
    if params["noise_std"] > 0:
        transforms.append(AddGaussianNoise(mean=0.0, std=params["noise_std"]))
    transforms.append(T.Normalize((PIXEL_MEAN,), (PIXEL_STD,)))
    return T.Compose(transforms)


class MashtotsDataModule(L.LightningDataModule):
    """Lightning data module for Mashtots training, validation, and inference.

    The module downloads the dataset when needed, splits the labeled images
    into train/validation (and optionally test) subsets with a stratified,
    seeded split, and exposes data loaders for Lightning's ``fit``,
    ``validate``, ``test``, and ``predict`` workflows.

    * ``test`` uses a labeled hold-out split of the training images, so it
      needs ``test_ratio > 0``.
    * ``predict`` uses the unlabeled Kaggle CSV (``new_test/new_test.csv``).

    Args:
        data_dir: Directory where the dataset is or will be stored.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes used by each data loader.
        seed: Seed used for splitting, shuffling, and worker RNGs.
        pin_memory: Whether data loaders should pin CPU memory before transfer
            to accelerator devices.
        val_ratio: Fraction of the labeled images used for validation.
        test_ratio: Fraction of the labeled images held out for testing.
        augmentation: Overrides for the training augmentation, see
            ``build_train_transform``.
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 128,
        num_workers: int = 8,
        seed: int = 42,
        pin_memory: bool = False,
        val_ratio: float = 0.1,
        test_ratio: float = 0.0,
        augmentation: Optional[dict[str, Any]] = None,
    ):
        """Initialize the data module. See the class docstring for arguments.

        Raises:
            ValueError: If ``val_ratio + test_ratio >= 1``.
        """

        super(MashtotsDataModule, self).__init__()
        if val_ratio + test_ratio >= 1:
            raise ValueError("val_ratio + test_ratio must be below 1")

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.train_transform = build_train_transform(**(augmentation or {}))
        self.inference_transform = T.Compose(
            [T.Normalize((PIXEL_MEAN,), (PIXEL_STD,))]
        )

    def _dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        """Create a data loader with the module's shared settings.

        Args:
            dataset: Dataset to load.
            shuffle: Whether to reshuffle every epoch.

        Returns:
            Configured data loader.
        """

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # Persistent workers need at least one worker process.
            persistent_workers=self.num_workers > 0,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=make_torch_generator(self.seed),
        )

    def prepare_data(self) -> None:
        """Download and extract the dataset unless it is already complete.

        Checks for the extracted files rather than the folder, so an empty or
        partial ``data_dir`` left by an interrupted download is filled in.

        Raises:
            RuntimeError: If the download finishes but the expected files are
                still missing.
        """

        if is_dataset_ready(self.data_dir):
            return

        download_and_extract_dataset(COMPETITION_NAME, self.data_dir)

        if not is_dataset_ready(self.data_dir):
            found = sorted(p.name for p in self.data_dir.iterdir())
            raise RuntimeError(
                f"Downloaded '{COMPETITION_NAME}' into {self.data_dir}, but "
                "Train/Train/<class>/ and new_test/new_test.csv were not found. "
                f"Top-level contents: {found}"
            )

    def _require_dataset(self) -> None:
        """Fail with a clear message when the dataset has not been downloaded.

        Raises:
            FileNotFoundError: If the dataset files are missing.
        """

        if not is_dataset_ready(self.data_dir):
            raise FileNotFoundError(
                f"No Mashtots dataset in {self.data_dir}. Call prepare_data() "
                "(Lightning does this in trainer.fit) to download it from Kaggle."
            )

    def train_dataloader(self) -> DataLoader:
        """Create the training data loader over the augmented training split."""

        return self._dataloader(self.mashtots_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Create the validation data loader (normalization only)."""

        return self._dataloader(self.mashtots_val)

    def test_dataloader(self) -> DataLoader:
        """Create the data loader over the labeled hold-out test split."""

        return self._dataloader(self.mashtots_test)

    def predict_dataloader(self) -> DataLoader:
        """Create the data loader over the unlabeled Kaggle prediction CSV."""

        return self._dataloader(self.mashtots_predict)

    def setup(self, stage: str) -> None:
        """Set up datasets for a Lightning stage.

        Args:
            stage: Lightning stage name: ``"fit"``, ``"validate"``,
                ``"test"``, or ``"predict"``.

        Raises:
            ValueError: If ``stage`` is unknown, or ``stage="test"`` while
                ``test_ratio`` is 0.
        """

        match stage:
            case "fit" | "validate":
                self._setup_labeled_splits()
            case "test":
                if self.test_ratio <= 0:
                    raise ValueError(
                        "The test stage evaluates a labeled hold-out split; set "
                        "data.test_ratio > 0. The Kaggle CSV has no labels and "
                        "is used by the predict stage."
                    )
                self._setup_labeled_splits()
            case "predict":
                self._setup_predict()
            case _:
                raise ValueError(f"Invalid DataModule Stage Setup Argument: {stage}")

    def split_indices(
        self, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split sample indices into stratified train/validation/test parts.

        With ``test_ratio == 0`` the train/validation split is identical to
        the one used before the test split existed, so results stay comparable.

        Args:
            labels: Class label of every sample.

        Returns:
            Train, validation, and test index arrays (test may be empty).
        """

        indices = np.arange(len(labels))
        test_indices = np.array([], dtype=int)
        if self.test_ratio > 0:
            indices, test_indices = train_test_split(
                indices,
                test_size=self.test_ratio,
                stratify=labels,
                random_state=self.seed,
            )
        train_indices, val_indices = train_test_split(
            indices,
            test_size=self.val_ratio / (1 - self.test_ratio),
            stratify=labels[indices],
            random_state=self.seed,
        )
        return train_indices, val_indices, test_indices

    def _setup_labeled_splits(self) -> None:
        """Create the train, validation and (optional) test subsets."""

        if hasattr(self, "mashtots_val"):
            return
        self._require_dataset()
        full = MashtotsDataset(self.data_dir, train=True, transform=None)
        train_idx, val_idx, test_idx = self.split_indices(np.asarray(full.labels))
        self.mashtots_train = TransformedSubset(full, train_idx, self.train_transform)
        self.mashtots_val = TransformedSubset(full, val_idx, self.inference_transform)
        self.mashtots_test = TransformedSubset(full, test_idx, self.inference_transform)

    def _setup_predict(self) -> None:
        """Create the prediction dataset from the Kaggle CSV."""

        self._require_dataset()
        self.mashtots_predict = MashtotsDataset(
            self.data_dir,
            train=False,
            transform=self.inference_transform,
        )
