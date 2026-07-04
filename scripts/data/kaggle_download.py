"""Utilities for downloading and extracting Kaggle competition datasets."""

import zipfile

from pathlib import Path
from dotenv import load_dotenv

# 1. Load the .env file explicitly first
load_dotenv()


def download_and_extract_dataset(
    competition_name: str,
    download_dir: Path,
) -> None:
    """Download a Kaggle competition archive and extract it locally.

    The Kaggle API downloads ``<competition_name>.zip`` into ``download_dir``.
    The archive is then extracted into the same directory and removed after a
    successful extraction.

    Args:
        competition_name: Kaggle competition slug to download.
        download_dir: Directory where the archive should be downloaded and
            extracted.

    Raises:
        OSError: If the archive cannot be read, extracted, or deleted.
        zipfile.BadZipFile: If the downloaded file is not a valid zip archive.
        kaggle.rest.ApiException: If the Kaggle API request fails.
    """

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.competition_download_files(
        competition_name,
        path=download_dir,
        quiet=False,
    )

    zip_path = download_dir / f"{competition_name}.zip"
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(download_dir)

    zip_path.unlink()
