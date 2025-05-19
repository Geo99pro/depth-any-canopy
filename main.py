import time
import torch
import hydra
import comet_ml
import lightning as pl

from omegaconf import DictConfig
from dataset import PrepareDataset
from lightning.pytorch.loggers import CometLogger
from lightning_model import DepthAnythingV2Module
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(args: DictConfig):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    
    custom_dataset = PrepareDataset(**args.dataset)
    train_dataset, val_dataset = custom_dataset.get_train_val_loaders()

    #data_module = EarthViewNEONDatamodule(**args.dataset)
    model = DepthAnythingV2Module(**args.model)

    experiment_id = time.strftime("%Y%m%d-%H%M%S")
    logger = False
    if args.logger:
        logger = CometLogger(
            api_key="9gB1lNby6NfLqBASYqyTh7oaD",
            workspace="depth-any-canopy-test-1",
            project="depth-any-canopy",
            experiment_name=f"depth-any-canopy-{experiment_id}",
            mode="create",
            online=True,
        )
        experiment_id = logger.experiment.id

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/{experiment_id}",
        filename="depth-any-canopy-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callback = [checkpoint_callback, early_stopping]
    if logger:
        callback.append(lr_monitor)

    trainer = pl.Trainer(
        **args.trainer,
        logger=logger,
        callbacks=callback,
        log_every_n_steps=50,
        precision="32-true" if args.model.encoder == "vitl" else "32-true",
        limit_val_batches=50,
        val_check_interval=1.0,
    )

    trainer.fit(model, 
                train_dataloaders=train_dataset, 
                val_dataloaders=val_dataset)


if __name__ == "__main__":
    main()
