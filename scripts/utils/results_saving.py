import pandas as pd
from pathlib import Path


def save_preds_to_csv(preds: list[int], save_path: Path) -> None:
    """Save class predictions in Kaggle submission format.

    Args:
        preds: Predicted class indices ordered by sample id.
        save_path: Destination CSV path.
    """

    save_dir = save_path.parent
    if not save_dir.exists():
        save_dir.mkdir()
    df = pd.DataFrame(
        {
            "Id": range(1, len(preds) + 1),
            "Category": [p for p in preds],
        }
    )
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} predictions → {save_path}")
