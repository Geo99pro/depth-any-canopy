import os
import time
import shutil as sh
import comet_ml
import torch
import hydra
import lightning as pl

from omegaconf import DictConfig
from dataset import ProcessData
from lightning.pytorch.loggers import CometLogger
from lightning_model import DepthAnythingV2Module
from safetensors.torch import load_file as safe_load
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(args: DictConfig):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    logger = None
    experiment_id = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    model_name = os.path.splitext(os.path.basename(args.model.safetensor_path))[0]
    dataset_name = "rgb_training"
    
    experiment_id = f"train-FT-{dataset_name}-{model_name}-{experiment_id}"
    output_path = os.path.join(os.getcwd(), experiment_id)
    
    if args.logger:
        logger = CometLogger(api_key="9gB1lNby6NfLqBASYqyTh7oaD",
                            workspace="NEW_depth-any-canopy-pretraining-models",
                            project_name="depth-any-canopy-model-small",
                            experiment_name=experiment_id,
                            offline=False)
        
    print("🟢 Training mode: loading training, validation....")
    checkpoint_path, scatter_plot_path = manage_folder(output_path)

    process_data_instance = ProcessData(images_folder_path=args.dataset.images_folder_path,
                                    chm_folder_path=args.dataset.chm_folder_path,
                                    split_ratio=args.dataset.split_ratio,
                                    mean=args.dataset.mean,
                                    std=args.dataset.std,
                                    batch_size=args.dataset.batch_size, 
                                    num_workers=args.dataset.num_workers,
                                    patch_size=args.dataset.patch_size,
                                    overlap_ratio=args.dataset.overlap_ratio,
                                    visualize_patches=args.dataset.visualize_patches,
                                    output_path=output_path)

    train_loader, val_loader, min_depth, max_depth = process_data_instance.get_loader_and_depth()
    
    model = DepthAnythingV2Module(encoder=args.model.encoder,
                                min_depth=min_depth,
                                max_depth=max_depth,
                                lr=args.model.lr,
                                use_huggingface=args.model.use_huggingface,
                                safetensor_path=args.model.safetensor_path,
                                freeze_parts=args.model.freeze_parts,
                                scatter_plot_path=scatter_plot_path)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=checkpoint_path,
                                    save_last=True, filename="depth-any-canopy-{epoch:02d}-{val_loss:.2f}",
                                    save_top_k=5, mode="min")

    early_stopping = EarlyStopping(monitor="val_loss",
                                    min_delta=0.00001,
                                    patience=3, mode="min", verbose=True)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callback = [checkpoint_callback, early_stopping]
    if logger:
        callback.append(lr_monitor)

    trainer = pl.Trainer(**args.trainer,
                        logger=logger,
                        callbacks=callback,
                        log_every_n_steps=50,
                        precision="32-true" if args.model.encoder == "vitl" else "32-true",
                        limit_val_batches=50,
                        val_check_interval=1.0)
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("🟢 Training completed successfully.")

def manage_folder(output_path):
    checkpoint_path = os.path.join(output_path, "checkpoints")
    scatter_plot_path = os.path.join(output_path, "scatter_plots")
    os.makedirs(output_path)
    os.makedirs(checkpoint_path)
    os.makedirs(scatter_plot_path)

    print(f"🟢 Train Output directory created at {output_path}.")
    print(f"🟢 Train Checkpoint directory created at {checkpoint_path}.")
    print(f"🟢 Train Scatter plot directory created at {scatter_plot_path}.")
    return checkpoint_path, scatter_plot_path

if __name__ == "__main__":
    main()
