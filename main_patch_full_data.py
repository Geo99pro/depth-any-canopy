import os
import time
import shutil as sh
import comet_ml
import torch
import hydra
import lightning as pl

from omegaconf import DictConfig
from dataset import PrepareDatasetFullData
from lightning.pytorch.loggers import CometLogger
from lightning_model import DepthAnythingV2Module
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(args: DictConfig):
    #pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    

    logger = None
    experiment_id = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    model_name = os.path.splitext(os.path.basename(args.model.safetensor_path))[0]
    dataset_name = os.path.splitext(os.path.basename(args.dataset.reference_tiff_path))[0]
    if args.dataset.is_test:
        experiment_id = f"test-FD-{dataset_name}-{model_name}-{experiment_id}"
    else:
        experiment_id = f"train-FT-FD-{dataset_name}-{model_name}-{experiment_id}"
    if args.logger:
        logger = CometLogger(api_key="9gB1lNby6NfLqBASYqyTh7oaD",
                            workspace="depth-any-canopy-pretraining-models",
                            project_name="depth-any-canopy-model-small",
                            experiment_name=experiment_id,
                            offline=False)

        #experiment_id = logger.experiment.id
        
    if not args.dataset.is_test:
        print("🟢 Training mode: loading train and val datasets...")
        output_path = os.path.join(os.getcwd(), experiment_id)
        os.makedirs(output_path)
        print(f"🟢 Output directory created at {output_path}.")
        scatter_plot_path = os.path.join(output_path, "scatter_plots")
        os.makedirs(scatter_plot_path)
        print(f"🟢 Scatter plot directory created at {scatter_plot_path}.")

        train_dataset, val_dataset = PrepareDatasetFullData(**args.dataset, output_path=output_path).get_train_val_loader()

        #data_module = EarthViewNEONDatamodule(**args.dataset)
        model = DepthAnythingV2Module(encoder=args.model.encoder,
                                        min_depth=args.model.min_depth,
                                        max_depth=args.model.max_depth,
                                        lr=args.model.lr,
                                        use_huggingface=args.model.use_huggingface,
                                        safetensor_path=args.model.safetensor_path,
                                        freeze_parts=args.model.freeze_parts,
                                        scatter_plot_path=scatter_plot_path)

        checkpoint_path = os.path.join(output_path, "checkpoints")
        os.makedirs(checkpoint_path)
        print(f"🟢 Checkpoint directory created at {checkpoint_path}.")

        checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=checkpoint_path,
                                        save_last=True, filename="depth-any-canopy-{epoch:02d}-{val_loss:.2f}",
                                        save_top_k=5, mode="min")

        early_stopping = EarlyStopping(monitor="val_loss", 
                                       patience=3, mode="min", verbose=True)

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
            val_check_interval=1.0)
        
        trainer.fit(model, train_dataloaders=train_dataset, val_dataloaders=val_dataset)

    elif args.dataset.is_test:
        print("🟡 Test mode: loading test dataset...")
        output_path = os.path.join(os.getcwd(), experiment_id)
        os.makedirs(output_path)
        print(f"🟡 Output directory created at {output_path}.")

        scatter_plot_path = os.path.join(output_path, "scatter_plots")
        os.makedirs(scatter_plot_path)
        print(f"🟡 Scatter plot directory created at {scatter_plot_path}.")
        
        test_dataset = PrepareDatasetFullData.get_test_loader(source_tiff_path=args.dataset.source_tiff_path,
                                                      reference_tiff_path=args.dataset.reference_tiff_path,
                                                      batch_size=args.dataset.batch_size,
                                                      num_workers=args.dataset.num_workers,
                                                      patch_size=args.dataset.patch_size,
                                                      overlap_ratio=args.dataset.overlap_ratio,
                                                      mean=args.dataset.mean,
                                                      std=args.dataset.std)
        #GET TEST DATASET LEN
        print(f"Test dataset length: {len(test_dataset)}")
        if args.model.safetensor_path.endswith(".ckpt"):
            print(f"📦 Loading model from checkpoint {args.model.safetensor_path}.")
            model = DepthAnythingV2Module.load_from_checkpoint(checkpoint_path=args.model.safetensor_path,
                                                            map_location="cpu",
                                                            strict=False,
                                                            scatter_plot_path=scatter_plot_path)
        
        elif args.model.safetensor_path.endswith(".safetensors"):
            print(f"📦 Loading model from safetensor {args.model.safetensor_path}.")
            model = DepthAnythingV2Module(encoder=args.model.encoder,
                                          min_depth=args.model.min_depth,
                                          max_depth=args.model.max_depth,
                                          safetensor_path=args.model.safetensor_path,
                                          freeze_parts=None)
            
        else:
            raise ValueError("Model path must be a .ckpt or .safetensors file.")

        trainer = pl.Trainer(logger=logger,
                             log_every_n_steps=50,
                             accelerator="auto")
        trainer.test(model, dataloaders=test_dataset)
if __name__ == "__main__":
    main()
