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

    if args.dataset.is_inference_only:
        model_name = os.path.splitext(os.path.basename(args.model.safetensor_path))[0]
        dataset_name = os.path.splitext(os.path.basename(args.dataset.reference_tiff_path))[0]
        test_experiment_id = f"Inference-{dataset_name}-{model_name}-{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
        logger = CometLogger(
            api_key="9gB1lNby6NfLqBASYqyTh7oaD",
            workspace="depth-any-canopy-pretraining-models",
            project_name="depth-any-canopy-model-small",
            experiment_name=test_experiment_id,
            offline=False)
        
        output_path = os.path.join(os.getcwd(), test_experiment_id)
        scatter_plot_path = os.path.join(output_path, "scatter_plots")
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(scatter_plot_path, exist_ok=True)

        process_data = ProcessData(
            image_path=args.dataset.isolate_test_source_tiff_path,
            chm_path=args.dataset.isolate_test_reference_tiff_path,
            batch_size=args.dataset.batch_size,
            num_workers=args.dataset.num_workers,
            patch_size=args.dataset.patch_size,
            overlap_ratio=args.dataset.overlap_ratio,
            output_path=output_path
        )
        test_loader, min_depth, max_depth = process_data.get_test_loader(
            args.dataset.isolate_test_source_tiff_path,
            args.dataset.isolate_test_reference_tiff_path
        )
        print(f"📦 Loading model from checkpoint {args.model.safetensor_path}")
        model = DepthAnythingV2Module(encoder=args.model.encoder,
                                    min_depth=min_depth,
                                    max_depth=max_depth,
                                    lr=args.model.lr,
                                    use_huggingface=args.model.use_huggingface,
                                    safetensor_path=args.model.safetensor_path,
                                    freeze_parts=args.model.freeze_parts,
                                    scatter_plot_path=scatter_plot_path)
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=50,
            accelerator="auto"
        )
        trainer.test(model, dataloaders=test_loader)

    else:
        logger = None
        experiment_id = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        model_name = os.path.splitext(os.path.basename(args.model.safetensor_path))[0]
        dataset_name = os.path.splitext(os.path.basename(args.dataset.reference_tiff_path))[0]

        experiment_id = f"train-FT-{dataset_name}-{model_name}-{experiment_id}"
        if args.logger:
            logger = CometLogger(api_key="9gB1lNby6NfLqBASYqyTh7oaD",
                                workspace="depth-any-canopy-pretraining-models",
                                project_name="depth-any-canopy-model-small",
                                experiment_name=experiment_id,
                                offline=False)
            
        print("🟢 Training mode: loading training, validation and test dataset....")
        output_path = os.path.join(os.getcwd(), experiment_id)
        checkpoint_path = os.path.join(output_path, "checkpoints")
        scatter_plot_path = os.path.join(output_path, "scatter_plots")
        os.makedirs(output_path)
        os.makedirs(checkpoint_path)
        os.makedirs(scatter_plot_path)

        print(f"🟢 Train Output directory created at {output_path}.")
        print(f"🟢 Train Checkpoint directory created at {checkpoint_path}.")
        print(f"🟢 Train Scatter plot directory created at {scatter_plot_path}.")

        process_data_instance = ProcessData(image_path=args.dataset.source_tiff_path,
                                        chm_path=args.dataset.reference_tiff_path,
                                        split_ratio=args.dataset.split_ratio,
                                        mean=args.dataset.mean,
                                        std=args.dataset.std,
                                        batch_size=args.dataset.batch_size, 
                                        num_workers=args.dataset.num_workers,
                                        patch_size=args.dataset.patch_size,
                                        overlap_ratio=args.dataset.overlap_ratio,
                                        ndvi_thr=args.dataset.ndvi_threshold,
                                        split_type=args.dataset.split_type,
                                        top_side_as_train=args.dataset.top_side_as_train,
                                        visualize_patches=args.dataset.visualize_patches,
                                        output_path=output_path)
        train_dataset, val_dataset, test_dataset, min_depth, max_depth = process_data_instance.get_loader_and_depth()
        
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
                                        min_delta=0.0001,
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
        
        trainer.fit(model, train_dataloaders=train_dataset, val_dataloaders=val_dataset)
        
        #prepare for testing
        best_ckpt_path = checkpoint_callback.best_model_path
        test_experiment_id = f"test-{dataset_name}-{model_name}-{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
        test_logger = CometLogger(
            api_key="9gB1lNby6NfLqBASYqyTh7oaD",
            workspace="depth-any-canopy-pretraining-models",
            project_name="depth-any-canopy-model-small",
            experiment_name=test_experiment_id,
            offline=False)
        output_path = os.path.join(os.getcwd(), test_experiment_id)
        scatter_plot_path = os.path.join(output_path, "scatter_plots")
        os.makedirs(output_path)
        os.makedirs(scatter_plot_path)
        print(f"🟡 Test Output directory created at {output_path}.")
        print(f"🟡 Test Scatter plot directory created at {scatter_plot_path}.")

        if test_dataset is not None:
            print("🟡 Test mode....")
            print(f"📦 Loading best model from checkpoint {best_ckpt_path}.")

            model = DepthAnythingV2Module.load_from_checkpoint(checkpoint_path=best_ckpt_path,
                                                                map_location="cpu",
                                                                strict=False,
                                                                scatter_plot_path=scatter_plot_path)
            trainer = pl.Trainer(logger=test_logger,
                                log_every_n_steps=50,
                                accelerator="auto")
            
            trainer.test(model, dataloaders=test_dataset)
        else:
            print("🟡 Test mode....")
            print("🟡 No test dataset provided from ProcessData, looking inside config")
            test_dataset, min_depth, max_depth = process_data_instance.get_test_loader(source_tiff_path=args.dataset.isolate_test_source_tiff_path,
                                                                reference_tiff_path=args.dataset.isolate_test_reference_tiff_path)
            print(f"📦 Loading best model from checkpoint {best_ckpt_path}.")
            model = DepthAnythingV2Module.load_from_checkpoint(checkpoint_path=best_ckpt_path,
                                                                map_location="cpu",
                                                                strict=False,
                                                                scatter_plot_path=scatter_plot_path)
            trainer = pl.Trainer(logger=test_logger,
                                log_every_n_steps=50,
                                accelerator="auto")
            trainer.test(model, dataloaders=test_dataset)

if __name__ == "__main__":
    main()
