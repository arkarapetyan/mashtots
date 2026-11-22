"""Utilities for downloading and extracting Kaggle competition datasets."""

import zipfile

from pathlib import Path
from dotenv import load_dotenv

# Make credentials from the repository's .env file visible to the Kaggle client.
load_dotenv()

_AUTH_HELP = (
    "Kaggle authentication failed, so the dataset could not be downloaded.\n"
    "Put your credentials in the .env file at the repository root (see env.example):\n"
    "  KAGGLE_API_TOKEN=<API token from https://www.kaggle.com/settings/api>\n"
    "or, with a legacy kaggle.json key:\n"
    "  KAGGLE_USERNAME=<your username>\n"
    "  KAGGLE_KEY=<key from kaggle.json>\n"
    "You must also have joined the competition and accepted its rules on Kaggle."
)


def download_and_extract_dataset(
    competition_name: str,
    download_dir: Path,
) -> None:
    """Download a Kaggle competition archive and extract it locally.

    The Kaggle API downloads ``<competition_name>.zip`` into ``download_dir``.
    The archive is then extracted into the same directory and removed after a
    successful extraction. A corrupt archive is deleted so the next attempt
    downloads it again.

    Args:
        competition_name: Kaggle competition slug to download.
        download_dir: Directory where the archive should be downloaded and
            extracted.

    Raises:
        RuntimeError: If Kaggle authentication fails or Kaggle refuses the
            download.
        OSError: If the archive cannot be read, extracted, or deleted.
        zipfile.BadZipFile: If the downloaded file is not a valid zip archive.
    """

    try:
        # Importing the package authenticates ``kaggle.api``; on failure the
        # Kaggle client prints help and calls ``exit(1)``.
        import kaggle
    except SystemExit:
        raise RuntimeError(_AUTH_HELP) from None

    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle competition '{competition_name}' → {download_dir}")
    try:
        kaggle.api.competition_download_files(
            competition_name,
            path=str(download_dir),
            quiet=False,
        )
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise RuntimeError(
                f"Kaggle refused the download (HTTP {status}).\n{_AUTH_HELP}"
            ) from exc
        raise

    zip_path = download_dir / f"{competition_name}.zip"
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(download_dir)
    except zipfile.BadZipFile:
        zip_path.unlink(missing_ok=True)
        raise

    zip_path.unlink()
