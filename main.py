import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import comet_ml
import time
import torch
import hydra
import lightning as pl

from omegaconf import DictConfig
from dataset import ProcessData
from lightning.pytorch.loggers import CometLogger
from lightning_model import DepthAnythingV2Module
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)

def safe_timestamp():
    """Get the current timestamp as a string."""
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

def build_experiment_id(is_train: bool, dataset_name: str, model_name: str) -> str:
    """Build a unique experiment ID based on training mode, dataset name, model name, and timestamp."""
    mode = "train" if is_train else "test"
    return f"{mode}-FT-{dataset_name}-{model_name}-{safe_timestamp()}"

def get_logger(args: DictConfig, experiment_id: str):
    """Initialize and return a CometLogger if logging is enabled in args.
    
    Args:
        args (DictConfig): The configuration arguments.
        experiment_id (str): The unique experiment ID.

    Returns:
        CometLogger: The initialized CometLogger or None if logging is disabled.
    """
    if not args.logger:
        return None
    
    return CometLogger(api_key=args.comet.api_key,
                       workspace=args.comet.workspace,
                       project=args.comet.project_name,
                       name=experiment_id,
                       online=args.comet.online,
                       tags=list(args.comet.get("tags", [])))

def ensure_dirs(*paths):
    """Ensure that the specified directories exist, creating them if necessary.
    
    Args:
        *paths: The directory paths to ensure exist.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)

def manage_folder(base_output: str):
    """
    Manage the output folder structure for the experiment.

    Args:
        base_output (str): The base output directory path.
    """
    ckpt = os.path.join(base_output, "checkpoints")
    plots = os.path.join(base_output, "scatter_plots")
    ensure_dirs(base_output, ckpt, plots)
    print(f"🟢 Created output directories at {base_output}.")
    print(f"🟢 Checkpoints at {ckpt}.")
    print(f"🟢 Scatter plots at {plots}.")
    return ckpt, plots

def get_mode(is_train: bool) -> str:
    """Return the mode string based on the is_train flag."""
    return "Training" if is_train else "Testing"

@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(args: DictConfig):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    model_name = os.path.splitext(os.path.basename(args.model.safetensor_path))[0]
    dataset_name = getattr(args.dataset, "name", "custom_dataset")
    
    experiment_id = build_experiment_id(args.dataset.is_train, dataset_name, model_name)
    output_path = os.path.join(os.getcwd(), experiment_id)

    logger = get_logger(args, experiment_id)

    if args.dataset.is_train:
        print("🟢 Training mode: loading training, validation....")
        checkpoint_path, scatter_plot_path = manage_folder(base_output=output_path)

        process_data_instance = ProcessData(images_folder_path=args.dataset.images_folder_path,
                                    chm_folder_path=args.dataset.chm_folder_path,
                                    split_ratio=args.dataset.split_ratio,
                                    mean=args.dataset.mean,
                                    std=args.dataset.std,
                                    batch_size=args.dataset.batch_size, 
                                    num_workers=args.dataset.num_workers,
                                    patch_size=args.dataset.patch_size,
                                    overlap_ratio=args.dataset.overlap_ratio,
                                    split_type=args.dataset.split_type,
                                    sam_mask=args.dataset.sam_mask,
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
                                scatter_plot_path=scatter_plot_path,
                                depth_bounds=process_data_instance.depth_bounds)

        callbacks = [ModelCheckpoint(monitor="val_loss", 
                                    dirpath=checkpoint_path,
                                    save_last=True, 
                                    filename="depth-any-canopy-{epoch:02d}-{val_loss:.2f}",
                                    save_top_k=5, 
                                    mode="min"),

                    EarlyStopping(monitor="val_loss",
                                min_delta=0.00001,
                                patience=3, 
                                mode="min", 
                                verbose=True),
                                
                    LearningRateMonitor(logging_interval="step")]

        trainer = pl.Trainer(max_epochs=args.trainer.max_epochs,
                        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
                        devices=1,
                        logger=logger,
                        callbacks=callbacks,
                        log_every_n_steps=args.comet.log_every_n_steps,
                        val_check_interval=1.0,
                        enable_progress_bar=True)

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print("🔴 Training completed successfully.")

    else:
        print("🟢 Testing mode: loading testing....")
        checkpoint_path, scatter_plot_path = manage_folder(output_path)
        test_loader, depth_bounds = ProcessData.get_test_loader(images_folder_path=args.dataset.images_folder_path,
                                                            chm_folder_path=args.dataset.chm_folder_path,
                                                            sam_binary_mask_path=args.dataset.sam_mask,
                                                            mean=args.dataset.mean,
                                                            std=args.dataset.std,
                                                            batch_size=args.dataset.batch_size, 
                                                            num_workers=args.dataset.num_workers,
                                                            patch_size=args.dataset.patch_size,
                                                            overlap_ratio=args.dataset.overlap_ratio,
                                                            global_chm_min=0.000,
                                                            global_chm_max=110.615,
                                                            visualize_patches=args.dataset.visualize_patches,
                                                            output_path=output_path)

        safepath = args.model.safetensor_path
        if safepath.endswith(".ckpt"):
            print(f"📦 Loading model from checkpoint {safepath}.")
            model = DepthAnythingV2Module.load_from_checkpoint(checkpoint_path=safepath,
                                                            map_location="cpu",
                                                            strict=False,
                                                            scatter_plot_path=scatter_plot_path,
                                                            freeze_parts=None,
                                                            depth_bounds=[depth_bounds],
                                                            output_path=output_path)

        elif args.model.safetensor_path.endswith(".safetensors"):
            print(f"📦 Loading model from safetensor {args.model.safetensor_path}.")
            model = DepthAnythingV2Module(encoder=args.model.encoder,
                                        min_depth=args.model.min_depth,
                                        max_depth=args.model.max_depth,
                                        safetensor_path=args.model.safetensor_path,
                                        scatter_plot_path=scatter_plot_path,
                                        freeze_parts=None,
                                        depth_bounds=[depth_bounds],
                                        output_path=output_path)

        else:
            raise ValueError("Model path must be a .ckpt or .safetensors file.")

        trainer = pl.Trainer(logger=logger,
                            log_every_n_steps=50,
                            accelerator="gpu" if torch.cuda.is_available() else "cpu",
                            devices=1)

        trainer.test(model, dataloaders=test_loader)
        print("🟡 Testing completed successfully.")


if __name__ == "__main__":
    main()
