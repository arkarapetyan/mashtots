"""Training entry point for Mashtots image classification models."""

import hydra
import torch
import lightning as L

from lightning.pytorch.callbacks import (
    EMAWeightAveraging,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from scripts.model import get_model_type
from scripts.data.data_module import MashtotsDataModule
from scripts.serialized_model import get_export_format
from scripts.train_config import TrainConfig
from scripts.utils.callbacks import ConfusionMatrixCallback
from scripts.utils.reproducibility import seed_everything


def resolve_precision(precision: str) -> str:
    """Turn ``"auto"`` into the fastest precision the hardware runs natively.

    Mixed precision keeps fp32 master weights and does the heavy math in 16
    bits, so it is faster without a meaningful accuracy cost.

    Args:
        precision: Configured precision, possibly ``"auto"``.

    Returns:
        A precision string accepted by Lightning.
    """

    if precision != "auto":
        return precision
    if not torch.cuda.is_available():
        return "32"
    if torch.cuda.is_bf16_supported(including_emulation=False):
        return "bf16-mixed"  # Ampere and newer: no loss scaling needed
    return "16-mixed"  # Volta/Turing: fp16 with dynamic loss scaling


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
            loss, trainer, callback, logger, and export settings.
        save_model: Whether to export and log the best model artifact after
            fitting. Disable this for hyperparameter search runs.

    Returns:
        Best validation accuracy recorded by the checkpoint callback.
    """

    seed_everything(config.seed)
    # Let remaining fp32 matmuls use TF32 tensor cores on Ampere and newer GPUs.
    torch.set_float32_matmul_precision("high")
    precision = resolve_precision(config.trainer.precision)
    print(f"Training precision: {precision}")
    model_name = config.model.model_name
    model_type = get_model_type(model_name)

    mashtots = MashtotsDataModule(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        seed=config.seed,
        pin_memory=config.data.pin_memory,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        augmentation=config.data.augmentation.model_dump(),
    )

    model = model_type(
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        dropout_rate=config.model.dropout_rate,
        optimizer_lr=config.model.optimizer_lr,
        optimizer_weight_decay=config.model.optimizer_weight_decay,
        max_epochs=config.trainer.max_epochs,
        loss=config.loss.model_dump(),
        **config.model.model_kwargs,
    )

    # "resnet+arcface" -> "resnet_arcface" for file names.
    file_tag = model_name.replace("+", "_")

    callbacks: list[L.Callback] = [
        ModelCheckpoint(
            dirpath=config.callback.checkpoints_dir,
            monitor="val_acc",
            filename=f"mashtots-{file_tag}-{{epoch:02d}}-{{val_acc:.4f}}",
            save_top_k=1,
            mode="max",
        ),
        EarlyStopping(
            monitor="val_acc",
            patience=config.callback.early_stop_patience,
            mode="max",
            verbose=False,
        ),
        LearningRateMonitor(logging_interval="step"),
        ConfusionMatrixCallback(
            num_classes=config.model.num_classes,
            every_n_epochs=config.callback.confusion_matrix_every_n_epochs,
            class_names=config.callback.class_names,
        ),
    ]
    if config.callback.weight_averaging.enabled:
        # Validation, checkpoints and the export all use the averaged weights.
        callbacks.append(
            EMAWeightAveraging(decay=config.callback.weight_averaging.decay)
        )
    checkpoint_callback = callbacks[0]

    logger = MLFlowLogger(
        experiment_name=config.logger.experiment_name,
        tracking_uri=config.logger.tracking_uri,
        run_name=config.logger.run_name or model_name,
    )

    trainer = L.Trainer(
        max_epochs=config.trainer.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=200,
        precision=precision,
        deterministic=config.trainer.deterministic,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
    )
    trainer.fit(model, datamodule=mashtots)

    best_model_score = checkpoint_callback.best_model_score
    if best_model_score is None:
        best_model_score = trainer.callback_metrics["val_acc"]
    best_val_acc = float(best_model_score.item())

    if config.data.test_ratio > 0:
        trainer.test(model, datamodule=mashtots, ckpt_path="best")

    if not save_model:
        return best_val_acc

    # Export from CPU: a .pt2 stores its weights on the device they were exported
    # from, and a CPU file loads on any machine (and is moved to a GPU if present).
    best_model = model_type.load_from_checkpoint(
        checkpoint_callback.best_model_path, map_location="cpu"
    )

    mashtots.setup(stage="validate")
    example_input = next(iter(mashtots.val_dataloader()))[0].cpu()
    export_format_type = get_export_format(config.export.export_format)
    save_path = export_format_type.export_model(
        model=best_model,
        example_input=example_input,
        save_dir=config.export.model_dir,
        filename=config.export.filename or f"mashtots-{file_tag}-{best_val_acc:.4f}",
    )
    print(f"Exported best model (val_acc={best_val_acc:.4f}) → {save_path}")

    logger.experiment.log_artifact(
        logger.run_id, str(save_path), artifact_path="models"
    )

    return best_val_acc


if __name__ == "__main__":
    main()
