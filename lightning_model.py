import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './././'))
sys.path.append(project_root)
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
from dataset.utils.utilities import Utils #save_inference_plot, plot
from torchmetrics import MetricCollection, classification, regression


class DepthAnythingV2Module(LightningModule):
    def __init__(self,
                encoder="vits",
                min_depth=1e-4,
                max_depth=20,
                lr=5e-5,
                use_huggingface=False,
                safetensor_path=None,
                freeze_parts=None,
                **kwargs):
            
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = kwargs
        self.test_references = []
        self.test_predictions = []
        self.test_masks = []
        self.test_source_ids = []

        self.model = self.load_model(encoder, use_huggingface, safetensor_path, freeze_parts)
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss()

        self.metric = MetricCollection([
            regression.MeanSquaredError(),
            regression.MeanAbsoluteError(),
        ])
        self.classification_metrics = MetricCollection([
            classification.JaccardIndex(task="binary")
        ])
        self.corr = MetricCollection([regression.PearsonCorrCoef()])

    def load_model(self, encoder, use_huggingface, safetensor_path, freeze_parts):
        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }

        if not use_huggingface:
            if safetensor_path is None:
                raise ValueError("Safetensor_path must be provided if use_huggingface is False")

            assert os.path.exists(safetensor_path), f"Path {safetensor_path} does not exist"
            model = DepthAnythingV2(**model_configs[encoder])
        
            if safetensor_path.endswith(".ckpt"):
                checkpoint = torch.load(safetensor_path, map_location="cpu")
                model.load_state_dict(checkpoint["state_dict"], strict=False)
            elif safetensor_path.endswith(".safetensors"):
                checkpoint = safe_load(safetensor_path)
                model.load_state_dict(checkpoint, strict=False)

            if freeze_parts:
                for name, param in model.named_parameters():
                    if any(name.startswith(part) for part in freeze_parts):
                        param.requires_grad = False
            summary(model, input_size=(1, 3, 518, 518))
        else:
            # model = transformers.AutoModelForDepthEstimation.from_pretrained(
            #     self.size_map[encoder], cache_dir="cache"
            # ).train()
            raise NotImplementedError("HuggingFace model loading not implemented in this revision.")

        return model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                    lr=self.hparams.lr,
                                    betas=(0.9, 0.999),
                                    weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        total_steps=self.trainer.estimated_stepping_batches,
                                                        max_lr=self.hparams.lr,
                                                        pct_start=0.05,
                                                        cycle_momentum=False,
                                                        div_factor=1e9,
                                                        final_div_factor=1e4)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def masked_loss(self, pred, target, mask, loss_fn):
        pixel_loss = loss_fn(pred, target)
        masked_loss = pixel_loss * mask
        return masked_loss.sum()/mask.sum().clamp(min=1)

    def forward(self, img):
        out = self.model(img)
        return out.predicted_depth if hasattr(out, "predicted_depth") else out

    def step(self, batch):
        mask = None
        if len(batch)==3:
            img, depth, source_ids = batch[0], batch[1], batch[2]
        elif len(batch) == 4:
            img, depth, mask, source_ids = batch[0], batch[1], batch[2],  batch[3]
        
        pred = self.forward(img)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)

        if mask is not None:
            loss = self.masked_loss(pred, depth, mask, self.mse_loss)
        else:
            loss = self.mse_loss(pred, depth)

        return img, depth, pred, mask, source_ids, loss

    def training_step(self, batch, batch_idx):
        img, depth, pred, mask, source_ids, loss = self.step(batch)

        self.log("train_loss", loss)
        if mask is not None:
            self.metric(pred * mask, depth * mask)
        else:
            self.metric(pred, depth)
        self.log_dict(self.metric)

        return loss

    def validation_step(self, batch, batch_idx):
        img, depth, pred, mask, source_ids, loss = self.step(batch)

        self.log("val_loss", loss)
        if mask is not None:
            self.metric(pred * mask, depth * mask)
        else:
            self.metric(pred, depth)
        self.log_dict(self.metric)

        if batch_idx < 10 and self.logger is not None:
            fig, scatter_fig = Utils.plot(
                img[0].cpu().detach(),
                depth[0].cpu().detach(),
                pred[0].cpu().detach(),
                mask[0].cpu().detach() if len(batch) == 4 else None,
                depth_bounds=self.kwargs['depth_bounds'],
                source_id=source_ids[0].item())

            self.logger.experiment.log_figure(figure=fig, figure_name=f"val_{batch_idx}")
            if scatter_fig:
                path_to_save = os.path.join(self.kwargs.get("scatter_plot_path"), f"val_scatter_{batch_idx}.png")
                scatter_fig.savefig(path_to_save)
                plt.close(scatter_fig)
            plt.close(fig)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test time.

        Args:
            batch: torch.D
        """
        img, depth, pred, mask, source_ids, loss = self.step(batch)
        print("Raw prediction stats -> min:", pred.min().item(),
            "max:", pred.max().item(),
            "mean:", pred.mean().item() )

        self.log("test_loss", loss)
        if mask is not None:
            self.metric(pred * mask, depth * mask)
        else:
            self.metric(pred, depth)
        self.log_dict(self.metric)
        
        self.classification_metrics(pred > 1e-8, depth > 1e-8)
        self.log_dict(self.classification_metrics)

        if batch_idx < 10 and self.logger is not None:
            fig, scatter_fig = Utils.plot(
                img[0].cpu().detach(),
                depth[0].cpu().detach(),
                pred[0].cpu().detach(),
                mask[0].cpu().detach() if len(batch) == 4 else None,
                depth_bounds=self.kwargs['depth_bounds'],
                source_id=source_ids[0].item())

            self.logger.experiment.log_figure(figure=fig, figure_name=f"test_{batch_idx}")
            if scatter_fig:
                path_to_save = os.path.join(self.kwargs.get("scatter_plot_path"), f"test_scatter_{batch_idx}.png")
                scatter_fig.savefig(path_to_save)
                plt.close(scatter_fig)
            plt.close(fig)

        output_dir = os.path.join(self.kwargs.get('output_path', 'results'), 'test_outputs')
        os.makedirs(output_dir, exist_ok=True)
        patch_id = batch_idx

        pred_np = pred.squeeze().cpu().numpy()
        depth_np = depth.squeeze().cpu().numpy()
        img_np = img.squeeze().cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy() if mask is not None else None

        source_id=source_ids[0].item()
        real_min, real_max = self.kwargs['depth_bounds'][source_id][2], self.kwargs['depth_bounds'][source_id][3]

        depth_np = depth_np * (real_max - real_min) + real_min
        pred_np = pred_np * (real_max - real_min) + real_min

        np.save(os.path.join(output_dir, f"depth_pred_patch_{patch_id}.npy"), pred_np)
        np.save(os.path.join(output_dir, f"depth_gt_patch_{patch_id}.npy"), depth_np)
        np.save(os.path.join(output_dir, f"image_patch_{patch_id}.npy"), img_np)
        np.save(os.path.join(output_dir, f"mask_patch_{patch_id}.npy"), mask_np) if mask is not None else None

        for i in range(depth_np.shape[0]):
            if mask is not None:
                valid = mask_np[i] > 0
                self.test_references.append(depth_np[i][valid])
                self.test_predictions.append(pred_np[i][valid])

            else:
                self.test_references.append(depth_np[i].flatten())
                self.test_predictions.append(pred_np[i].flatten())

        self.test_source_ids.append(source_ids.cpu().numpy())            

        # for i in range(img_np.shape[0]):
        #     Utils.save_inference_plot(rgb=img_np[i],
        #                 depth_gt=depth_np[i],
        #                 depth_pred=pred_np[i],
        #                 mask=mask_np[i] if mask is not None else None,
        #                 save_path=os.path.join(output_dir, f"inference_plot_patch_{patch_id}_img_{i}.png"),
        #                 title=f"Patch {patch_id} - Image {i} - Source {source_id}")

        # return loss

    def on_test_epoch_end(self):
        output_dir = os.path.join(self.kwargs.get('output_path', 'results'), 'test_outputs')
        os.makedirs(output_dir, exist_ok=True)

        refs = np.concatenate(self.test_references)
        preds = np.concatenate(self.test_predictions)

        print(f"\n=== Accumulated {len(refs)} points ===")
        print(f"References min/max: {refs.min()} / {refs.max()}")
        print(f"Predictions min/max: {preds.min()} / {preds.max()}")

        # ✅ Plot des bandes ±1 std et ±2 std
        Utils.get_reference_mean_std_dual_band(
            reference=refs,
            predicted=preds,
            max_bin_value=35,
            output_path=os.path.join(output_dir, "reference_std_bands.png")
        )

        # ✅ Plot matrice de corrélation hexbin
        fig, ax = plt.subplots(figsize=(6, 6))
        hb = ax.hexbin(refs, preds, gridsize=50, cmap='plasma', bins='log')
        ax.plot([0, max(refs.max(), preds.max())], [0, max(refs.max(), preds.max())], 'r--', linewidth=2)

        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('log(count)')

        ax.set_xlabel('Ground Truth Depth')
        ax.set_ylabel('Predicted Depth')
        ax.set_title('Correlation Matrix')

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "correlation_matrix.png"))
        plt.close(fig)

        print(f"=== Saved reference_std_bands.png and correlation_matrix.png to {output_dir} ===")


    # model_configs = {
    #     "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    #     "vitb": {
    #         "encoder": "vitb",
    #         "features": 128,
    #         "out_channels": [96, 192, 384, 768],
    #     },
    #     "vitl": {
    #         "encoder": "vitl",
    #         "features": 256,
    #         "out_channels": [256, 512, 1024, 1024],
    #     },
    #     "vitg": {
    #         "encoder": "vitg",
    #         "features": 384,
    #         "out_channels": [1536, 1536, 1536, 1536],
    #     },
    # }

    # size_map = {
    #     "vits": "depth-anything/Depth-Anything-V2-Small-hf",
    #     "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
    #     "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
    #     "vitg": None,
    # }

    # def __init__(
    #     self,
    #     encoder: Literal["vits", "vitb", "vitl", "vitg"],
    #     min_depth: float = 1e-4,
    #     max_depth: float = 20,
    #     lr: float = 0.000005,
    #     use_huggingface: bool = False,
    #     safetensor_path: str = None,
    #     freeze_parts: list[str] = None,
    #     **kwargs):

    #     self.kwargs = kwargs
    #     super().__init__()
    #     self.save_hyperparameters()

    #     if not use_huggingface:
    #         if safetensor_path is None:
    #             raise ValueError("safetensor_path must be provided if use_huggingface is False")
            
    #         assert os.path.exists(safetensor_path), f"Path {safetensor_path} does not exist"

    #         if safetensor_path.endswith(".ckpt"):
    #             #loading from a .ckpt file
    #             checkpoint = torch.load(safetensor_path, map_location="cpu", weights_only=False)
    #             self.model = DepthAnythingV2(**{**self.model_configs[encoder]})
    #             self.model.load_state_dict(checkpoint["state_dict"], strict=False)
    #         elif safetensor_path.endswith(".safetensors"):
    #             checkpoint = safe_load(safetensor_path)
    #             self.model = DepthAnythingV2(**{**self.model_configs[encoder]})
    #             self.model.load_state_dict(checkpoint, strict=False)

    #         if freeze_parts is not None:
    #             for name, param in self.model.named_parameters():
    #                 if any(name.startswith(part) for part in freeze_parts):
    #                     param.requires_grad = False
    #                     print(f"🧊 Froze layer: {name}")
    #         summary(self.model, 
    #             input_size=(1, 3, 518, 518),
    #             col_names=["input_size", "output_size", "num_params"])

    #     else:
    #         self.model = transformers.AutoModelForDepthEstimation.from_pretrained(
    #             self.size_map[encoder], cache_dir="cache"
    #         ).train()

    #     self.loss = nn.MSELoss()
    #     self.mae_loss = nn.L1Loss()
    #     self.h_loss = nn.HuberLoss()

    #     self.metric = MetricCollection(
    #         [regression.MeanSquaredError(),
    #         regression.MeanAbsoluteError(),
    #         ])

    #     self.classification_metrics = MetricCollection(
    #         [classification.JaccardIndex(task="binary")]
    #     )
    #     self.corr = MetricCollection([regression.PearsonCorrCoef()])
    #     self.predictions = []

    

    # def masked_huber_loss(self, pred: torch.Tensor,
    #                     target: torch.Tensor,
    #                     mask: torch.Tensor,
    #                     delta: float = 1.0):
    #     """
    #     Compute the masked Huber loss only on masked regions.
    #     Args:
    #         pred (torch.Tensor): Predicted depth map, shape (B, 1, H, W).
    #         target (torch.Tensor): Target depth map, shape (B, 1, H, W).
    #         mask (torch.Tensor): Mask indicating valid regions, shape (B, 1, H, W), values in {0, 1}.
    #         delta (float): Threshold for Huber loss.

    #     Returns:
    #         torch.Tensor: Masked Huber loss.
    #     """
    #     pixel_loss = F.huber_loss(pred, target, reduction="none", delta=delta)
    #     masked_loss = pixel_loss * mask
    #     return masked_loss.sum() / mask.sum().clamp(min=1)

    # def masked_mse_loss(self,
    #                     pred: torch.Tensor,
    #                     target: torch.Tensor, 
    #                     mask: torch.Tensor):
    #     """
    #     Compute the masked mean squared error only on masked regions.
    #     Args:
    #         pred (torch.Tensor): Predicted depth map, shape (B, 1, H, W).
    #         target (torch.Tensor): Target depth map, shape (B, 1, H, W).
    #         mask (torch.Tensor): Mask indicating valid regions, shape (B, 1, H, W), values in {0, 1}.
        
    #     Returns:
    #         torch.Tensor: Masked mean squared error loss.
    #     """
    #     pixel_loss = self.loss(pred, target)
    #     masked_loss = pixel_loss * mask
    #     return masked_loss.sum() / mask.sum().clamp(min=1)

    # def masked_mae_loss(self,
    #                     pred: torch.Tensor,
    #                     target: torch.Tensor,
    #                     mask: torch.Tensor):
    #         pixel_loss = self.mae_loss(pred, target)
    #         masked_loss = pixel_loss * mask
    #         return masked_loss.sum() / mask.sum().clamp(min=1)

    # def training_step(self, batch, batch_idx):
    #     """Training step for the model.
    #     Args:
    #         batch (dict): A batch of data containing image, depth, and mask.
        
    #     Returns:
    #         torch.Tensor: The loss value for the training step.
    #     """
    #     mask = None
    #     if len(batch) == 2:
    #         img, depth = batch[0], batch[1]
    #         out = self.model(img)
    #         pred = out.predicted_depth if hasattr(out, "predicted_depth") else out

    #     elif len(batch)==4:
    #         img, depth, mask, source_ids = batch[0], batch[1], batch[2], batch[3]
    #         out = self.model(img)
    #         pred = out.predicted_depth if hasattr(out, "predicted_depth") else out

    #     elif len(batch) == 3:
    #         img, depth, source_ids = batch[0], batch[1], batch[2]
    #         out = self.model(img)
    #         pred = out.predicted_depth if hasattr(out, "predicted_depth") else out
            
    #     if pred.ndim == 3:  # add batch dimension if missing
    #         pred = pred.unsqueeze(1)
        
    #     if len(batch) == 2:
    #         loss = self.h_loss(pred, depth).mean()
    #     elif len(batch) == 3:
    #         loss = self.mae_loss(pred, depth).mean()
    #     elif len(batch) == 4:
    #         loss = self.masked_huber_loss(pred, depth, mask)

    #     self.log("train_loss", loss)
    #     if mask is not None:
    #         self.metric(pred * mask, depth * mask)
    #     else:
    #         self.metric(pred, depth)
    #     self.log_dict(self.metric)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     """Validation step for the model.
    #         This method computes the model's predictions, calculates the loss, and logs metrics.
    #         It plots the results for the first 10 batches if a logger is available.

    #         Args:
    #             batch (dict): A batch of data containing image, depth, and mask.
    #             batch_idx (int): The index of the batch.

    #         Returns:
    #             torch.Tensor: The loss value for the validation step.
    #     """
    #     mask = None
    #     if len(batch) == 2:
    #         img, depth = batch[0], batch[1]
    #         out = self.model(img)
    #         pred = out.predicted_depth if hasattr(out, "predicted_depth") else out

    #     elif len(batch)==4:
    #         img, depth, mask, source_ids = batch[0], batch[1], batch[2], batch[3]
    #         out = self.model(img)
    #         pred = out.predicted_depth if hasattr(out, "predicted_depth") else out

    #     elif len(batch) == 3:
    #         img, depth, source_ids = batch[0], batch[1], batch[2]
    #         out = self.model(img)
    #         pred = out.predicted_depth if hasattr(out, "predicted_depth") else out

    #     if pred.ndim == 3:
    #         pred = pred.unsqueeze(1)

    #     if len(batch) == 2:
    #         loss = self.h_loss(pred, depth).mean()

    #     elif len(batch) == 4:
    #         loss = self.masked_huber_loss(pred, depth, mask)

    #     elif len(batch) == 3:
    #         loss = self.mae_loss(pred, depth).mean()

    #     self.log("val_loss", loss)
    #     if mask is not None:
    #         self.metric(pred * mask, depth * mask)
    #     else:
    #         self.metric(pred, depth)
    #     self.log_dict(self.metric)

    #     if batch_idx < 10 and self.logger is not None:
    #         fig, scatter_fig = self.plot(
    #             img[0].cpu().detach(),
    #             depth[0].cpu().detach(),
    #             pred[0].cpu().detach(),
    #             mask[0].cpu().detach() if len(batch) == 4 else None,
    #             source_id=source_ids[0].item()
    #         )
    #         self.logger.experiment.log_figure(figure=fig, figure_name=f"val_{batch_idx}")
    #         if scatter_fig:
    #             path_to_save = os.path.join(self.kwargs.get("scatter_plot_path"), f"val_scatter_{batch_idx}.png")
    #             scatter_fig.savefig(path_to_save)
    #             plt.close(scatter_fig)
    #         plt.close(fig)
    #     return loss

    # def test_step(self, batch, batch_idx):
    #     """Test step for the model.
        
    #     Args:
    #         batch (dict): A batch of data containing image, depth, and mask.
    #         batch_idx (int): The index of the batch.
            
    #     This method computes the model's predictions, calculates the loss, and logs metrics.
    #     Also, it plots the results for the first 10 batches if a logger is available.
    #     It handles both cases where the batch contains either 2 or 3 elements (image, depth, and optionally mask).

    #     Returns:
    #         None. Logs the loss and metrics, and optionally plots results.
    #     """
    #     mask = None
    #     if len(batch) == 2:
    #         img, depth = batch[0], batch[1] #torch.clamp(batch[1], min=self.hparams.min_depth, max=self.hparams.max_depth)
    #         out = self.model(img)
    #         pred = out.predicted_depth if hasattr(out, "predicted_depth") else out

    #     if len(batch) == 3:
    #         img, depth, source_ids = batch[0], batch[1], batch[2]
    #         out = self.model(img)
    #         pred = out.predicted_depth if hasattr(out, "predicted_depth") else out

    #     elif len(batch) == 4:
    #         img, depth, mask, source_ids = batch[0], batch[1], batch[2], batch[3]
    #         out = self.model(img)
    #         pred = out.predicted_depth if hasattr(out, "predicted_depth") else out

    #     if pred.ndim == 3:
    #         pred = pred.unsqueeze(1)

    #     if len(batch) == 2:
    #         loss = self.h_loss(pred, depth).mean()
    #     elif len(batch) == 3 :
    #         loss = self.loss(pred, depth)

    #     self.log("test_loss", loss)
    #     if mask is not None:
    #         self.metric(pred * mask, depth * mask)
    #     else:
    #         self.metric(pred, depth)
    #     self.log_dict(self.metric)
    #     self.classification_metrics(pred > 1e-8, depth > 1e-8)
    #     self.log_dict(self.classification_metrics)

    #     if batch_idx < 10 and self.logger is not None:
    #         fig, scatter_fig = self.plot(
    #             img[0].cpu().detach(),
    #             depth[0].cpu().detach(),
    #             pred[0].cpu().detach(),
    #             mask[0].cpu().detach() if len(batch) == 4 else None,
    #             source_id=source_ids[0].item()
    #         )
    #         self.logger.experiment.log_figure(figure=fig, figure_name=f"test_{batch_idx}")
    #         if scatter_fig:
    #             path_to_save = os.path.join(self.kwargs.get("scatter_plot_path"), f"test_scatter_{batch_idx}.png")
    #             scatter_fig.savefig(path_to_save)
    #             plt.close(scatter_fig)
    #         plt.close(fig)

    #     output_dir = os.path.join(self.kwargs.get('output_path', 'results'), 'test_outputs')
    #     os.makedirs(output_dir, exist_ok=True)
    #     patch_id = batch_idx
        
    #     pred_np = pred.squeeze().cpu().numpy()
    #     depth_np = depth.squeeze().cpu().numpy()
    #     img_np = img.squeeze().cpu().numpy()
    #     source_id=source_ids[0].item()
    #     real_min, real_max = self.kwargs['depth_bounds'][source_id][2], self.kwargs['depth_bounds'][source_id][3]
    #     depth_np = depth_np * (real_max - real_min) + real_min
    #     pred_np = pred_np * (real_max - real_min) + real_min
    #     np.save(os.path.join(output_dir, f"depth_pred_patch_{patch_id}.npy"), pred_np)
    #     np.save(os.path.join(output_dir, f"depth_gt_patch_{patch_id}.npy"), depth_np)
    #     np.save(os.path.join(output_dir, f"image_patch_{patch_id}.npy"), img_np)
        
    #     if mask is not None:
    #         mask_np = mask.squeeze().cpu().numpy()
    #         np.save(os.path.join(output_dir, f"mask_patch_{patch_id}.npy"), mask_np)
            
    #     # ➕ Sauvegarde du plot
    #     self.save_inference_plot(
    #         rgb=img_np,
    #         depth_gt=depth_np,
    #         depth_pred=pred_np,
    #         mask=mask_np if mask is not None else None,
    #         save_path=os.path.join(output_dir, f"inference_plot_patch_{patch_id}.png"),
    #         title=f"Patch {patch_id} - Source {source_id}"
    #     )


    #     return loss

    # # output_dir = "results/test_outputs"
    # # os.makedirs(output_dir, exist_ok=True)

    # # patch_id = source_ids[0] if source_ids is not None else batch_idx

    # # pred_np = pred.squeeze().cpu().numpy()
    # # depth_np = depth.squeeze().cpu().numpy()
    # # img_np = img.squeeze().cpu().numpy()

    # # np.save(os.path.join(output_dir, f"depth_pred_patch_{patch_id}.npy"), pred_np)
    # # np.save(os.path.join(output_dir, f"depth_gt_patch_{patch_id}.npy"), depth_np)
    # # np.save(os.path.join(output_dir, f"image_patch_{patch_id}.npy"), img_np)

    # # if mask is not None:
    # #     mask_np = mask.squeeze().cpu().numpy()
    # #     np.save(os.path.join(output_dir, f"mask_patch_{patch_id}.npy"), mask_np)


    # def plot(self, image, depth, prediction=None, mask=None, source_id=None, show_titles=True):
    #     """
    #     Plots RGB image, depth map, NDVI mask, and optionally the prediction and difference.
    #     Also rescales using the real depth values based on source_id.

    #     Args:
    #         image (Tensor): RGB image (3, H, W)
    #         depth (Tensor): Ground truth CHM normalized (1, H, W)
    #         prediction (Tensor): Optional predicted depth normalized (1, H, W)
    #         mask (Tensor): Optional NDVI binary mask (1, H, W)
    #         source_id (int): Index of the original image to retrieve real min/max
    #         show_titles (bool): Whether to show plot titles
    #     Returns:
    #         fig, scatter_fig: matplotlib figures
    #     """

    #     # Normalize image for display
    #     image = image.float()
    #     image_min, image_max = image.min(), image.max()
    #     image = (image - image_min) / (image_max - image_min) if image_max > image_min else torch.zeros_like(image)

    #     depth = depth.float()
    #     if mask is not None:
    #         mask = mask.float()

    #     # === Get real depth range from source_id ===
    #     if source_id is not None:
    #         real_min, real_max = self.kwargs['depth_bounds'][source_id][2], self.kwargs['depth_bounds'][source_id][3]
    #         # Rescale depth and prediction
    #         depth = depth * (real_max - real_min) + real_min
    #         if prediction is not None:
    #             prediction = prediction * (real_max - real_min) + real_min
    #         vmax = real_max
    #     else:
    #         vmax = 1  # fallback if no source_id

    #     # === Number of columns ===
    #     ncols = 2  # RGB + Depth
    #     if mask is not None:
    #         ncols += 2
    #     if prediction is not None:
    #         ncols += 2

    #     fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))
    #     axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    #     idx = 0
    #     axs[idx].imshow(image.permute(1, 2, 0).cpu())
    #     axs[idx].axis("off")
    #     if show_titles:
    #         axs[idx].set_title("RGB Image")
    #     idx += 1

    #     im_depth = axs[idx].imshow(depth.squeeze().cpu(), cmap="Spectral_r", vmin=0, vmax=vmax)
    #     axs[idx].axis("off")
    #     if show_titles:
    #         axs[idx].set_title("Depth Map")
    #     fig.colorbar(im_depth, ax=axs[idx])
    #     idx += 1

    #     if mask is not None:
    #         im_mask = axs[idx].imshow(mask.squeeze().cpu(), cmap="gray", vmin=0, vmax=1)

    #         axs[idx].axis("off")
    #         if show_titles:
    #             axs[idx].set_title("Binary Mask")
    #         fig.colorbar(im_mask, ax=axs[idx])
    #         idx += 1

    #         depth_masked = depth * mask
    #         im_depth_masked = axs[idx].imshow(depth_masked.squeeze().cpu(), cmap="Spectral_r", vmin=0, vmax=vmax)
    #         axs[idx].axis("off")
    #         if show_titles:
    #             axs[idx].set_title("Depth * Mask")
    #         fig.colorbar(im_depth_masked, ax=axs[idx])
    #         idx += 1

    #     if prediction is not None:
    #         pred_vis = prediction * mask if mask is not None else prediction

    #         im_pred = axs[idx].imshow(pred_vis.squeeze().cpu(), cmap="Spectral_r", vmin=0, vmax=vmax)
    #         axs[idx].axis("off")
    #         if show_titles:
    #             axs[idx].set_title("Prediction")
    #         fig.colorbar(im_pred, ax=axs[idx])
    #         idx += 1

    #         diff = torch.abs(prediction - depth).squeeze().cpu()
    #         im_diff = axs[idx].imshow(diff, cmap="viridis", vmin=0, vmax=(vmax - 0))
    #         axs[idx].axis("off")
    #         if show_titles:
    #             axs[idx].set_title("Difference |Pred - CHM|")
    #         fig.colorbar(im_diff, ax=axs[idx])

    #     plt.tight_layout()
    #     plt.subplots_adjust(wspace=0.3)

    #     scatter_fig = None
    #     if prediction is not None:
    #         scatter_fig, ax = plt.subplots(figsize=(6, 6))
    #         depth_np = depth.flatten().cpu().numpy()
    #         pred_np = prediction.flatten().cpu().numpy()

    #         if mask is not None:
    #             mask_np = mask.flatten().cpu().numpy()
    #             valid = mask_np > 0
    #             depth_np = depth_np[valid]
    #             pred_np = pred_np[valid]

    #         hb = ax.hexbin(depth_np, pred_np, gridsize=50, cmap='plasma', bins='log')
    #         ax.plot([0, vmax], [0, vmax], color='red', linestyle='--', linewidth=2)
    #         ax.set_xlabel("Ground Truth Depth")
    #         ax.set_ylabel("Predicted Depth")
    #         ax.set_title("2D Density: Prediction vs Ground Truth")
    #         ax.set_xlim(0, vmax)
    #         ax.set_ylim(0, vmax)
    #         fig.colorbar(hb, ax=ax, label='log(count)')
    #         ax.grid(True)
    #         plt.tight_layout()

    #     return fig, scatter_fig

    # def save_inference_plot(self, rgb, depth_gt, depth_pred, save_path, mask=None, title=None):
    #     image = np.transpose(rgb, (1, 2, 0))
    #     image_min, image_max = image.min(), image.max()
    #     image = (image - image_min) / (image_max - image_min) if image_max > image_min else torch.zeros_like(image)

    #     ncols = 4 if mask is not None else 3
    #     fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    #     axes[0].imshow(image)
    #     axes[0].set_title("RGB Image")
    #     axes[0].axis('off')
    #     vmin=0
    #     vmax = max(depth_gt.max(), depth_pred.max())
    #     im1 = axes[1].imshow(depth_gt, cmap="Spectral_r", vmin=vmin, vmax=vmax)
    #     axes[1].set_title("Depth Ground Truth")
    #     axes[1].axis('off')
    #     plt.colorbar(im1, ax=axes[1])

    #     im2 = axes[2].imshow(depth_pred, cmap="Spectral_r",vmin=vmin, vmax=vmax)
    #     axes[2].set_title("Depth Prediction")
    #     axes[2].axis('off')
    #     plt.colorbar(im2, ax=axes[2])

    #     if mask is not None:
    #         axes[3].imshow(mask, cmap="gray")
    #         axes[3].set_title("Mask")
    #         axes[3].axis('off')

    #     if title:
    #         fig.suptitle(title, fontsize=16)

    #     plt.tight_layout()
    #     plt.savefig(save_path, bbox_inches='tight')
    #     plt.close(fig)

