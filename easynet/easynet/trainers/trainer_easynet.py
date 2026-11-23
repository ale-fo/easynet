from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Any
import numpy as np
import torch

from moviad.datasets.vad_dataset import VADDataset
from moviad.models.training_args import TrainingArgs
from moviad.trainers.trainer import Trainer
from moviad.utilities.evaluation.metrics import AuPro, Metric, MetricLvl

from easynet.models.easynet.easynet import TOP_EPOCHS, EasyNet, EasyNetFusionMode, top_epochs_order


class TrainerEasynet(Trainer):

    LR_DECAY_EPOCHS = (0.8, 0.9)
    LR_GAMMA = 0.2
    SELECTION_METRICS = ("img_roc_auc", "img_avg_prec", "pxl_roc_auc", "pxl_aupro")

    def __init__(
        self,
        train_args: TrainingArgs,
        model: EasyNet,
        train_dataset: VADDataset,
        eval_dataset: VADDataset | None,
        metrics: list[Metric],
        device: torch.device,
        logger: Any | None = None,
        logging_prefix: str = "",
        save_path: str | None = None,
    ) -> None:
        super().__init__(
            train_args,
            model,
            train_dataset,
            eval_dataset,
            metrics,
            device,
            logger = logger,
            logging_prefix = logging_prefix,
            save_path = save_path,
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = train_args.batch_size,
            shuffle = train_args.shuffle,
            num_workers = 1,
        )
        if eval_dataset is not None:
            self.eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size = train_args.batch_size,
                shuffle = False,
                num_workers = 1,
                generator = torch.Generator(),
            )
            metric_names = []
            for metric in metrics:
                metric_names.append(metric.name)
            for name in self.SELECTION_METRICS:
                if name not in metric_names:
                    raise ValueError(f"{name} not found in selected metrics.")
        if save_path is not None:
            Path(save_path).parent.mkdir(parents = True, exist_ok = True)
        self.top_epochs = []

    @staticmethod
    def predict(model: EasyNet, image: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if model.fusion_mode == EasyNetFusionMode.SINGLE:
            return model(image)
        if model.fusion_mode == EasyNetFusionMode.DEPTH:
            return model(depth)
        return model(image, depth)

    @staticmethod
    def evaluate(model: EasyNet, dataloader: torch.utils.data.DataLoader, metrics: list[Metric], device: torch.device) -> dict:
        model.eval()
        scores, labels, pixel_scores, pixel_labels, anomaly_maps = [], [], [], [], []
        with torch.no_grad():
            for batch in dataloader:
                image, depth, label, mask = batch[0].to(device), batch[1].to(device), batch[2], batch[3]
                anomaly_map, pred_score = TrainerEasynet.predict(model, image, depth)
                anomaly_map = anomaly_map.cpu().numpy()
                for i in range(anomaly_map.shape[0]):
                    anomaly_maps.append(anomaly_map[i, 0])
                scores.extend(pred_score.cpu().numpy().tolist())
                labels.extend(label.numpy().tolist())
                pixel_scores.append(anomaly_map.reshape(anomaly_map.shape[0], -1))
                pixel_labels.append(mask.numpy().reshape(mask.shape[0], -1))

        pro_masks = []
        for i in range(len(dataloader.dataset)):
            pro_masks.append(dataloader.dataset.load_pro_mask(i))

        labels, scores = np.array(labels), np.array(scores)
        pixel_labels = np.concatenate(pixel_labels).reshape(-1)
        pixel_scores = np.concatenate(pixel_scores).reshape(-1)
        report = {}
        for metric in metrics:
            if isinstance(metric, AuPro):
                value = metric.compute(np.stack(pro_masks), np.stack(anomaly_maps))
            elif metric.level == MetricLvl.IMAGE:
                value = metric.compute(labels, scores)
            else:
                value = metric.compute(pixel_labels, pixel_scores)
            report[metric.name] = float(value)
        return report

    def update_top_epochs(self, epoch: int, metrics: dict) -> None:
        total = 0.0
        for name in self.SELECTION_METRICS:
            total += metrics[name]
        score = total / len(self.SELECTION_METRICS)
        if not math.isfinite(score):
            return
        candidates = list(self.top_epochs)
        candidates.append({"epoch": epoch, "score": score, "metrics": metrics})
        candidates.sort(key = top_epochs_order)
        self.top_epochs = candidates[:TOP_EPOCHS]

    @staticmethod
    def atomic_save(state: dict, path: Path) -> None:
        temporary_path = path.with_name(path.name + ".tmp")
        torch.save(state, temporary_path)
        os.replace(temporary_path, path)

    def epoch_weights_path(self, epoch: int) -> Path:
        best_path = Path(self.save_path)
        return best_path.with_name(f"{best_path.stem.removesuffix('_best')}_epoch{epoch}{best_path.suffix}")

    def save_top_weights(self, previous_top_epochs: list, epoch: int) -> None:
        if self.save_path is None:
            return
        top_epoch_numbers = []
        for entry in self.top_epochs:
            top_epoch_numbers.append(entry["epoch"])
        if epoch in top_epoch_numbers:
            self.atomic_save(self.model.state_dict(), self.epoch_weights_path(epoch))
            if top_epoch_numbers[0] == epoch:
                self.atomic_save(self.model.state_dict(), Path(self.save_path))
        for entry in previous_top_epochs:
            if entry["epoch"] not in top_epoch_numbers:
                self.epoch_weights_path(entry["epoch"]).unlink(missing_ok = True)

    def train(self, epochs: int, evaluation_epoch_interval: int = 10) -> dict[str, float]:
        print(f"Train EasyNet. Mode: {self.model.fusion_mode.value}. Scales: {self.model.num_scales}")
        self.train_args.init_train(self.model)
        self.train_args.init_task(self.model, len(self.train_dataloader))

        self.model.train()

        if self.logger is not None:
            self.logger.watch(self.model)

        if self.train_args.optimizer is None:
            raise ValueError("train_args.optimizer is required")
        milestones = []
        for fraction in self.LR_DECAY_EPOCHS:
            milestones.append(int(fraction * epochs))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.train_args.optimizer, milestones = milestones, gamma = self.LR_GAMMA)

        self.top_epochs = []
        metrics = {}
        for epoch in range(epochs):

            print(f"EPOCH: {epoch}")

            train_loss = self.model.train_epoch(epoch, self.train_dataloader, self.train_args)
            scheduler.step()

            if self.logger is not None:
                self.logger.log({
                    f"{self.logging_prefix}train/epoch": epoch,
                    f"{self.logging_prefix}train/train_loss": train_loss,
                })

            if self.eval_dataloader is not None and (epoch + 1) % evaluation_epoch_interval == 0:
                print("Evaluating model...")
                metrics = self.evaluate(self.model, self.eval_dataloader, self.metrics, self.device)
                previous_top_epochs = self.top_epochs
                self.update_top_epochs(epoch, metrics)
                self.save_top_weights(previous_top_epochs, epoch)
                if self.logger is not None:
                    self.logger.log(metrics)

        if self.logger is not None:
            self.logger.log(metrics)

        print("End training performances:")
        self.print_metrics(metrics)

        if self.top_epochs:
            top_epoch_texts = []
            for entry in self.top_epochs:
                top_epoch_texts.append(str(entry["epoch"]))
            print(f"Best training performances (epoch {self.top_epochs[0]['epoch']}, top epochs {' '.join(top_epoch_texts)}):")
            self.print_metrics(self.top_epochs[0]["metrics"])

        return metrics