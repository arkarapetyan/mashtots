"""Prediction entry point for exported Mashtots classification models.

Runs one model, or an ensemble that averages the class probabilities of
several exported models (``.pt2`` and/or ``.onnx``), on the Kaggle test CSV.
"""

import hydra
import torch
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from scripts.data.data_module import MashtotsDataModule
from scripts.ensemble import EnsemblePredictor
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
    """Generate predictions with one or more serialized models and write a CSV.

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
    data_module.prepare_data()  # downloads the dataset if it is missing
    data_module.setup(stage="predict")
    predict_loader = data_module.predict_dataloader()

    predictor = EnsemblePredictor(
        model_paths=config.model.model_paths,
        weights=config.model.weights,
        tta=config.model.tta,
        tta_shift_px=config.model.tta_shift_px,
    )
    for path, weight in zip(predictor.model_paths, predictor.weights):
        print(f"Model {path.name}: weight {weight:.3f}")

    ensemble_preds: list[torch.Tensor] = []
    model_preds: list[list[torch.Tensor]] = [[] for _ in predictor.models]
    for batch in tqdm(predict_loader):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        probs, per_model = predictor.predict_proba(images)
        ensemble_preds.append(probs.argmax(dim=1))
        for preds, model_probs in zip(model_preds, per_model):
            preds.append(model_probs.argmax(dim=1))

    final = torch.cat(ensemble_preds)
    if len(predictor.models) > 1:
        for path, preds in zip(predictor.model_paths, model_preds):
            agreement = (torch.cat(preds) == final).float().mean().item()
            print(f"{path.name} agrees with the ensemble on {agreement:.2%} of samples")

    out_path = config.output.output_dir / config.output.filename
    save_preds_to_csv(
        final.tolist(),
        out_path,
        ids=data_module.mashtots_predict.ids,
    )


if __name__ == "__main__":
    main()
