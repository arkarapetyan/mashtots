"""Writing predictions in Kaggle submission format."""

from typing import Optional, Sequence

import pandas as pd
from pathlib import Path


def save_preds_to_csv(
    preds: Sequence[int],
    save_path: Path,
    ids: Optional[Sequence] = None,
) -> None:
    """Save class predictions in Kaggle submission format.

    Args:
        preds: Predicted class indices ordered like the prediction CSV.
        save_path: Destination CSV path.
        ids: Sample ids from the prediction CSV. When ``None``, ids are
            ``1..N`` in row order.

    Raises:
        ValueError: If ``ids`` and ``preds`` have different lengths.
    """

    if ids is None:
        ids = range(1, len(preds) + 1)
    elif len(ids) != len(preds):
        raise ValueError(f"Got {len(ids)} ids for {len(preds)} predictions")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"Id": list(ids), "Category": list(preds)})
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} predictions → {save_path}")
