"""Training entry point for Mashtots image classification models."""

import mlflow
import hydra
import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from scripts.model import get_model_type
from scripts.data.data_module import MashtotsDataModule
from scripts.serialized_model import get_export_format
from scripts.train_config import TrainConfig
from scripts.utils.callbacks import ConfusionMatrixCallback


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig) -> None:
    """Parse Hydra configuration and launch training.

    Args:
        cfg: Hydra configuration object resolved from the configured training
            config files.
    """

    container = OmegaConf.to_container(cfg, resolve=True)
    config = TrainConfig(**container)
    train(config)


def train(config: TrainConfig, save_model: bool = True) -> float:
    """Train a Mashtots classification model and optionally export it.

    Args:
        config: Validated training configuration containing model, data,
            trainer, callback, logger, and export settings.
        save_model: Whether to export and log the best model artifact after
            fitting. Disable this for hyperparameter search runs.

    Returns:
        Best validation accuracy recorded by the checkpoint callback.
    """

    model_type = get_model_type(config.model.model_name)
    print(model_type)

    mashtots = MashtotsDataModule(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        val_ratio=config.data.val_ratio,
        num_workers=config.data.num_workers,
        seed=config.seed,
        pin_memory=config.data.pin_memory,
    )

    model = model_type(
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        dropout_rate=config.model.dropout_rate,
        optimizer_lr=config.model.optimizer_lr,
        optimizer_weight_decay=config.model.optimizer_weight_decay,
        max_epochs=config.trainer.max_epochs,
    )

    confusion_matrix_callback = ConfusionMatrixCallback(
        num_classes=config.model.num_classes,
        every_n_epochs=5,
        class_names=config.callback.class_names,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.callback.checkpoints_dir,
        monitor="val_acc",
        filename="mashtots-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
        save_top_k=1,
        mode="max",
    )

    early_stopping = EarlyStopping(
        monitor="val_acc",
        patience=config.callback.early_stop_patience,
        mode="min",
        verbose=False,
    )

    lr_monitor = LearningRateMonitor(
        logging_interval="step",
    )

    logger = MLFlowLogger(
        experiment_name=config.logger.experiment_name,
        tracking_uri=config.logger.tracking_uri,
    )

    trainer = L.Trainer(
        max_epochs=config.trainer.max_epochs,
        callbacks=[
            checkpoint_callback,
            early_stopping,
            lr_monitor,
            confusion_matrix_callback,
        ],
        logger=logger,
        log_every_n_steps=200,
        precision=config.trainer.precision,
        deterministic=True,
    )
    trainer.fit(model, datamodule=mashtots)

    best_model_score = checkpoint_callback.best_model_score
    if best_model_score is None:
        best_model_score = trainer.callback_metrics["val_acc"]
    best_val_acc = float(best_model_score.item())

    if not save_model:
        return best_val_acc

    best_model_path = checkpoint_callback.best_model_path
    best_model = model.__class__.load_from_checkpoint(
        best_model_path,
        model_name=config.model.model_name,
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
    )

    example_input = next(iter(mashtots.train_dataloader()))[0]
    export_format_type = get_export_format(config.export.export_format)
    save_path = export_format_type.export_model(
        model=best_model,
        example_input=example_input,
        save_dir=config.export.model_dir,
        filename=f"mashtots-efficientnet-best-{best_val_acc:.2f}"
        if not config.export.filename
        else config.export.filename,
    )

    with mlflow.start_run(run_id=logger.run_id):
        mlflow.log_artifact(save_path, artifact_path="models")

    return best_val_acc


if __name__ == "__main__":
    main()
