import os
import torch
import pprint
import random
import logging
import warnings
import argparse
import numpy as np
import transformers
import polars as pl
from typing import Literal
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import nn
from torchinfo import summary
from model import DepthAnythingV2
from lightning import LightningModule
from kornia.geometry.transform import resize
from safetensors.torch import load_file as safe_load
from torchmetrics import MetricCollection, classification, regression


class DepthAnythingV2Module(LightningModule):
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }

    size_map = {
        "vits": "depth-anything/Depth-Anything-V2-Small-hf",
        "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
        "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
        "vitg": None,
    }

    def __init__(
        self,
        encoder: Literal["vits", "vitb", "vitl", "vitg"],
        min_depth: float = 1e-4,
        max_depth: float = 20,
        lr: float = 0.000005,
        use_huggingface: bool = False,
        safetensor_path: str = None,
        freeze_parts: list[str] = None,
        **kwargs,
    ):
        self.kwargs = kwargs
        super().__init__()
        self.save_hyperparameters()

        if not use_huggingface:
            if safetensor_path is None:
                raise ValueError("safetensor_path must be provided if use_huggingface is False")
            
            assert os.path.exists(safetensor_path), f"Path {safetensor_path} does not exist"

            if safetensor_path.endswith(".ckpt"):
                #loading from a .ckpt file
                checkpoint = torch.load(safetensor_path, map_location="cpu", weights_only=False)
                self.model = DepthAnythingV2(**{**self.model_configs[encoder]})
                self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            elif safetensor_path.endswith(".safetensors"):
                checkpoint = safe_load(safetensor_path)
                self.model = DepthAnythingV2(**{**self.model_configs[encoder]})
                self.model.load_state_dict(checkpoint, strict=False)
            
            if freeze_parts is not None:
                for name, param in self.model.named_parameters():
                    if any(name.startswith(part) for part in freeze_parts):
                        param.requires_grad = False
                        print(f"🧊 Froze layer: {name}")
            summary(self.model, 
                input_size=(1, 3, 518, 518),
                col_names=["input_size", "output_size", "num_params"])

        else:
            self.model = transformers.AutoModelForDepthEstimation.from_pretrained(
                self.size_map[encoder], cache_dir="cache"
            ).train()

        self.loss = nn.MSELoss()
        self.h_loss = nn.HuberLoss()
        self.metric = MetricCollection(
            [regression.MeanSquaredError(),
             regression.MeanAbsoluteError(),
             ])#regression.NormalizedRootMeanSquaredError()

        self.classification_metrics = MetricCollection(
            [classification.JaccardIndex(task="binary")]
        )
        self.corr = MetricCollection([regression.PearsonCorrCoef()])
        self.predictions = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.lr,
                                      betas=(0.9, 0.999),
                                      weight_decay=0.01,)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        total_steps=self.trainer.estimated_stepping_batches,
                                                        max_lr=self.hparams.lr,
                                                        pct_start=0.05,
                                                        cycle_momentum=False,
                                                        div_factor=1e9,
                                                        final_div_factor=1e4)

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
    
    def masked_huber_loss(self, pred: torch.Tensor,
                        target: torch.Tensor,
                        mask: torch.Tensor,
                        delta: float = 1.0):
        """
        Compute the masked Huber loss only on masked regions.
        Args:
            pred (torch.Tensor): Predicted depth map, shape (B, 1, H, W).
            target (torch.Tensor): Target depth map, shape (B, 1, H, W).
            mask (torch.Tensor): Mask indicating valid regions, shape (B, 1, H, W), values in {0, 1}.
            delta (float): Threshold for Huber loss.

        Returns:
            torch.Tensor: Masked Huber loss.
        """
        pixel_loss = F.huber_loss(pred, target, reduction="none", delta=delta)
        masked_loss = pixel_loss * mask
        return masked_loss.sum() / mask.sum().clamp(min=1)

    def masked_mse_loss(self,
                        pred: torch.Tensor,
                        target: torch.Tensor, 
                        mask: torch.Tensor):
        """
        Compute the masked mean squared error only on masked regions.
        Args:
            pred (torch.Tensor): Predicted depth map, shape (B, 1, H, W).
            target (torch.Tensor): Target depth map, shape (B, 1, H, W).
            mask (torch.Tensor): Mask indicating valid regions, shape (B, 1, H, W), values in {0, 1}.
        
        Returns:
            torch.Tensor: Masked mean squared error loss.
        """
        pixel_loss = self.loss(pred, target)
        masked_loss = pixel_loss * mask
        return masked_loss.sum() / mask.sum().clamp(min=1)

    def training_step(self, batch, batch_idx):
        """Training step for the model.
        Args:
            batch (dict): A batch of data containing image, depth, and mask.
        
        Returns:
            torch.Tensor: The loss value for the training step.
        """
        if len(batch) == 2:
            img, depth = batch[0], batch[1] #torch.clamp(batch[1], min=self.hparams.min_depth, max=self.hparams.max_depth)
            out = self.model(img)
            pred = out.predicted_depth if hasattr(out, "predicted_depth") else out
        else:
            img, depth, mask = batch[0], batch[1], batch[2] #torch.clamp(batch[1], min=self.hparams.min_depth, max=self.hparams.max_depth)
            out = self.model(img)
            pred = out.predicted_depth if hasattr(out, "predicted_depth") else out
        
        if pred.ndim == 3:  # add batch dimension if missing
            pred = pred.unsqueeze(1)
        
        if len(batch) == 2:
            loss = self.h_loss(pred, depth).mean()
        else:
            loss = self.masked_huber_loss(pred, depth, mask)

        self.log("train_loss", loss)
        self.metric(pred, depth)
        self.log_dict(self.metric)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for the model.
            This method computes the model's predictions, calculates the loss, and logs metrics.
            It plots the results for the first 10 batches if a logger is available.

            Args:
                batch (dict): A batch of data containing image, depth, and mask.
                batch_idx (int): The index of the batch.
          
            Returns:
                torch.Tensor: The loss value for the validation step.
        """
        if len(batch) == 2:
            img, depth = batch[0], batch[1]
            out = self.model(img)
            pred = out.predicted_depth if hasattr(out, "predicted_depth") else out
        else:
            img, depth, mask = batch[0], batch[1], batch[2] #torch.clamp(batch[1], min=self.hparams.min_depth, max=self.hparams.max_depth), batch[2]
            out = self.model(img)
            pred = out.predicted_depth if hasattr(out, "predicted_depth") else out

        if pred.ndim == 3:
            pred = pred.unsqueeze(1)

        if len(batch) == 2:
            loss = self.h_loss(pred, depth).mean()
        else:
            loss = self.masked_huber_loss(pred, depth, mask)
        self.log("val_loss", loss)
        self.metric(pred, depth)
        self.log_dict(self.metric)

        if batch_idx < 10 and self.logger is not None:
            fig, scatter_plot = self.plot(
                img[0].cpu().detach(),
                depth[0].cpu().detach(),
                pred[0].cpu().detach(),
                mask[0].cpu().detach() if len(batch) == 3 else None
            )
            self.logger.experiment.log_figure(figure=fig, figure_name=f"val_{batch_idx}")
            if scatter_plot:
                #self.logger.experiment.log_figure(figure=scatter_plot, figure_name=f"val_scatter_{batch_idx}")
                #save locally
                path_to_save = os.path.join(self.kwargs.get("scatter_plot_path"), f"val_scatter_{batch_idx}.png")
                scatter_plot.savefig(path_to_save)
                if scatter_plot:plt.close(scatter_plot)
            plt.close(fig)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step for the model.
        
        Args:
            batch (dict): A batch of data containing image, depth, and mask.
            batch_idx (int): The index of the batch.
            
        This method computes the model's predictions, calculates the loss, and logs metrics.
        Also, it plots the results for the first 10 batches if a logger is available.
        It handles both cases where the batch contains either 2 or 3 elements (image, depth, and optionally mask).

        Returns:
            None. Logs the loss and metrics, and optionally plots results.
        """
        if len(batch) == 2:
            img, depth = batch[0], batch[1] #torch.clamp(batch[1], min=self.hparams.min_depth, max=self.hparams.max_depth)
            out = self.model(img)
            pred = out.predicted_depth if hasattr(out, "predicted_depth") else out
        else:
            img, depth, mask = batch[0], batch[1], batch[2] #torch.clamp(batch[1], min=self.hparams.min_depth, max=self.hparams.max_depth), batch[2]
            out = self.model(img)
            pred = out.predicted_depth if hasattr(out, "predicted_depth") else out
            
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        
        if len(batch) == 2:
            loss = self.h_loss(pred, depth).mean()
        else:
            loss = self.masked_huber_loss(pred, depth, mask)

        self.log("test_loss", loss)
        self.metric(pred, depth)
        self.log_dict(self.metric)
        self.classification_metrics(pred > 1e-4, depth > 1e-4)
        self.log_dict(self.classification_metrics)
        # self.predictions.append(
        #     {
        #         "prediction": pred[depth > 1e-4].flatten().detach().cpu(),
        #         "depth": depth[depth > 1e-4].flatten().detach().cpu(),
        #     })

        if batch_idx < 10 and self.logger is not None:
            fig, scatter_fig = self.plot(
                img[0].cpu().detach(),
                depth[0].cpu().detach(),
                pred[0].cpu().detach(),
                mask[0].cpu().detach() if len(batch) == 3 else None
            )
            self.logger.experiment.log_figure(figure=fig, figure_name=f"test_{batch_idx}")
            if scatter_fig:
                #self.logger.experiment.log_figure(figure=scatter_fig, figure_name=f"test_scatter_{batch_idx}")
                #save locally
                path_to_save = os.path.join(self.kwargs.get("scatter_plot_path"), f"test_scatter_{batch_idx}.png")
                scatter_fig.savefig(path_to_save)
            plt.close(fig)
            if scatter_fig:plt.close(scatter_fig)
        return loss
                

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if len(batch) == 2:
            img, depth = batch[0], torch.clamp(batch[1], min=self.hparams.min_depth, max=self.hparams.max_depth)
        else:
            img, depth, mask = batch[0], torch.clamp(batch[1], min=self.hparams.min_depth, max=self.hparams.max_depth), batch[2]
        #img = resize(img, (518, 518), interpolation="bilinear")
        if img.ndim == 3: # add batch dimension if missing
            img = img.unsqueeze(0)
        pred = self.model(img).predicted_depth
        pred = resize(pred, depth.shape[-2:], interpolation="bilinear").clamp(self.hparams.min_depth, self.hparams.max_depth)
        return pred

    def _preprocess_batch(self, batch):
        #img, depth = batch["image"], batch["mask"]
        img, depth, mask   = batch[0], batch[1], batch[2]
        img = resize(img, (518, 518), interpolation="bilinear")

        depth = torch.clamp(
            depth, min=self.hparams.min_depth, max=self.hparams.max_depth
        )

        return img, depth, mask

    def plot(self, image, depth, prediction=None, mask=None, show_titles=True):
            """
            Plots RGB image, depth map, NDVI mask, and optionally the prediction and difference.

            Args:
                image (Tensor): RGB image (3, H, W)
                depth (Tensor): Ground truth CHM (1, H, W)
                prediction (Tensor): Optional predicted depth (1, H, W)
                mask (Tensor): Optional NDVI binary mask (1, H, W)
                show_titles (bool): Whether to show plot titles
            Returns:
                fig, scatter_fig: matplotlib figures
            """
            image = image.float()
            image_min, image_max = image.min(), image.max()
            if image_max > image_min:
                image = (image - image_min) / (image_max - image_min)
            else:
                image = torch.zeros_like(image)

            depth = depth.float()
            if mask is not None:
                mask = mask.float()

            ncols = 2  # RGB + Depth
            if mask is not None:
                ncols += 1
            if prediction is not None:
                ncols += 2

            fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))
            axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

            idx = 0
            axs[idx].imshow(image.permute(1, 2, 0).cpu())
            axs[idx].axis("off")
            if show_titles: axs[idx].set_title("RGB Image")
            idx += 1

            im_depth = axs[idx].imshow(depth.squeeze().cpu(), cmap="Spectral_r",
                                    vmin=0,
                                    vmax=1)
            axs[idx].axis("off")
            if show_titles: axs[idx].set_title("Depth Map")
            fig.colorbar(im_depth, ax=axs[idx])
            idx += 1

            if mask is not None:
                im_mask = axs[idx].imshow(mask.squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
                axs[idx].axis("off")
                if show_titles: axs[idx].set_title("NDVI Mask")
                fig.colorbar(im_mask, ax=axs[idx])
                idx += 1

            if prediction is not None:
                prediction = prediction.float()
                if mask is not None:
                    pred_vis = prediction * mask
                else:
                    pred_vis = prediction

                im_pred = axs[idx].imshow(pred_vis.squeeze().cpu(), cmap="Spectral_r",
                                        vmin=0,
                                        vmax=1)
                axs[idx].axis("off")
                if show_titles: axs[idx].set_title("Prediction")
                fig.colorbar(im_pred, ax=axs[idx])
                idx += 1

                diff = torch.abs(prediction - depth).squeeze().cpu()
                im_diff = axs[idx].imshow(diff, cmap="viridis",
                                        vmin=0, vmax=1 - 0)
                axs[idx].axis("off")
                if show_titles: axs[idx].set_title("Difference |Pred - CHM|")
                fig.colorbar(im_diff, ax=axs[idx])

            plt.tight_layout()
            plt.subplots_adjust(wspace=0.3)

            # Scatter plot (Prediction vs Ground Truth)
            scatter_fig = None
            if prediction is not None:
                scatter_fig, ax = plt.subplots(figsize=(6, 6))
                depth_np = depth.flatten().cpu().numpy()
                pred_np = prediction.flatten().cpu().numpy()

                if mask is not None:
                    mask_np = mask.flatten().cpu().numpy()
                    valid = mask_np > 0
                    depth_np = depth_np[valid]
                    pred_np = pred_np[valid]

                hb = ax.hexbin(depth_np, pred_np, gridsize=50, cmap='plasma', bins='log')
                ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)
                ax.set_xlabel("Ground Truth Depth")
                ax.set_ylabel("Predicted Depth")
                ax.set_title("2D Density: Prediction vs Ground Truth")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                fig.colorbar(hb, ax=ax, label='log(count)')
                ax.grid(True)
                plt.tight_layout()

            return fig, scatter_fig

        