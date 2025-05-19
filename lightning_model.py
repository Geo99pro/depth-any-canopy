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
        super().__init__()
        self.save_hyperparameters()

        if not use_huggingface:
            if safetensor_path is None:
                raise ValueError("safetensor_path must be provided if use_huggingface is False")
            
            assert os.path.exists(safetensor_path), f"Path {safetensor_path} does not exist"

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

        self.loss = nn.MSELoss(reduction="none")
        #self.another_loss = F.huber_loss()
        self.metric = MetricCollection(
            [regression.MeanSquaredError(),
             regression.MeanAbsoluteError(),
             regression.NormalizedRootMeanSquaredError()])

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

    def training_step(self, batch):
        """Training step for the model.
        Args:
            batch (dict): A batch of data containing image, depth, and mask.
        
        Returns:
            torch.Tensor: The loss value for the training step.
        """
        
        img, depth, mask = batch[0], torch.clamp(batch[1], min=self.hparams.min_depth, max=self.hparams.max_depth), batch[2]
        out = self.model(img)
        pred = out.predicted_depth if hasattr(out, "predicted_depth") else out
  
        if pred.ndim == 3:
             pred = pred.unsqueeze(1)
        pred = resize(pred, depth.shape[-2:], interpolation="bilinear").clamp(self.hparams.min_depth, self.hparams.max_depth)
        
        pixel_loss = self.loss(pred, depth)
        masked_loss = pixel_loss * mask
        loss = masked_loss.sum() / mask.sum().clamp(min=1)
        #loss = self.loss(pred, depth)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for the model.
        Args:
            batch (dict): A batch of data containing image, depth, and mask.
            batch_idx (int): The index of the batch.
        """
        img, depth, mask = batch[0], torch.clamp(batch[1], min=self.hparams.min_depth, max=self.hparams.max_depth), batch[2]
        out = self.model(img)
        pred = out.predicted_depth if hasattr(out, "predicted_depth") else out
        # pred = self.model(img).predicted_depth

        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        pred = resize(pred, depth.shape[-2:], interpolation="bilinear").clamp(self.hparams.min_depth, self.hparams.max_depth)

        #loss = self.loss(pred, depth)
        pixel_loss = self.loss(pred, depth)
        masked_loss = pixel_loss * mask
        loss = masked_loss.sum() / mask.sum().clamp(min=1)
        self.log("val_loss", loss)
        self.metric(pred, depth)
        self.log_dict(self.metric)

        if batch_idx < 10 and self.logger is not None:
            fig = self.plot(
                img[0].cpu().detach(),
                depth[0].cpu().detach(),
                pred[0].cpu().detach(),
                mask[0].cpu().detach())

            self.logger.experiment.log_figure(
                figure=fig, figure_name=f"val_{batch_idx}"
            )
            plt.close(fig)

        return loss

    def test_step(self, batch, batch_idx):
        img, depth = self._preprocess_batch(batch)

        pred = self.model(img).predicted_depth

        pred = resize(pred, depth.shape[-2:], interpolation="bilinear").clamp(0, 1)

        self.metric(pred, depth)
        self.log_dict(self.metric)

        self.classification_metrics(pred > 1e-4, depth > 1e-4)
        self.log_dict(self.classification_metrics)

        # self.corr(pred[depth > 1e-4].flatten(), depth[depth > 1e-4].flatten())
        # self.log_dict(self.corr)

        self.predictions.append(
            {
                "prediction": pred[depth > 1e-4].flatten().detach().cpu(),
                "depth": depth[depth > 1e-4].flatten().detach().cpu(),
            }
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        img, depth = self._preprocess_batch(batch)

        pred = self.model(img).predicted_depth

        pred = resize(pred, depth.shape[-2:], interpolation="bilinear").clamp(0, 1)

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
        """
        image = image.float()
        image = image - image.min()
        image = image / image.max()

        depth = depth.float()
        if mask is not None:
            mask = mask.float()

        showing_prediction = prediction is not None
        ncols = 3 + int(showing_prediction) + int(showing_prediction)  # RGB + depth + NDVI + pred + diff
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        axs[0].imshow(image.permute(1, 2, 0))
        axs[0].axis("off")
        axs[0].set_title("RGB Image")

        axs[1].imshow(depth.squeeze(), cmap="Spectral_r", 
                    vmin=self.hparams.min_depth, vmax=self.hparams.max_depth)
        axs[1].axis("off")
        axs[1].set_title("Depth Map")

        if mask is not None:
            axs[2].imshow(mask.squeeze(), cmap="gray", vmin=0, vmax=1)
            axs[2].axis("off")
            axs[2].set_title("NDVI Mask")

        col_offset = 3 if mask is not None else 2

        if showing_prediction:
            prediction = prediction.clip(self.hparams.min_depth, self.hparams.max_depth).float()
            axs[col_offset].imshow(prediction.squeeze(), cmap="Spectral_r",
                                vmin=self.hparams.min_depth, vmax=self.hparams.max_depth)
            axs[col_offset].axis("off")
            axs[col_offset].set_title("Prediction")

            diff = torch.abs(prediction - depth).squeeze()
            axs[col_offset + 1].imshow(diff, cmap="Spectral", 
                                    vmin=0, vmax=self.hparams.max_depth - self.hparams.min_depth)
            axs[col_offset + 1].axis("off")
            axs[col_offset + 1].set_title("Difference |Pred - CHM|")

        plt.tight_layout()
        return fig
