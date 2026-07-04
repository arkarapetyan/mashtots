"""Prediction entry point for exported Mashtots classification models."""

import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from scripts.data.data_module import MashtotsDataModule
from scripts.serialized_model import get_extension_format
from scripts.test_config import TestConfig
from scripts.utils.reproducibility import seed_everything
from scripts.utils.results_saving import save_preds_to_csv


@hydra.main(config_path="configs", config_name="test_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Parse Hydra configuration and run prediction.

    Args:
        cfg: Hydra configuration object resolved from the configured prediction
            config files.
    """

    container = OmegaConf.to_container(cfg, resolve=True)
    config = TestConfig(**container)
    predict(config)


def predict(config: TestConfig) -> None:
    """Generate predictions with a serialized model and write them to CSV.

    Args:
        config: Validated prediction configuration containing model, data, and
            output settings.
    """

    seed_everything(config.seed)

    data_module = MashtotsDataModule(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        seed=config.seed,
        pin_memory=config.data.pin_memory,
    )
    data_module.setup(stage="predict")
    predict_loader = data_module.predict_dataloader()

    extension_format_type = get_extension_format(config.model.model_path.suffix)
    serialized_model = extension_format_type(model_path=config.model.model_path)

    all_preds: list[int] = []
    for batch in tqdm(predict_loader):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        logits = serialized_model.predict(images)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)

    out_path = config.output.output_dir / config.output.filename
    save_preds_to_csv(all_preds, out_path)


if __name__ == "__main__":
    main()
