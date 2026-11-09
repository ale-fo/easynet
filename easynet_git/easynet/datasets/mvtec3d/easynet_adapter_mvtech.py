from __future__ import annotations
from typing import Optional
import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset

from moviad.utilities.configurations import LabelName, Split

from .perlin_anomaly_generator import PerlinAnomalyGenerator
from .mvtec3d_dataset import MVTec3DDataset


def organized_pc_to_depth_image(organized_pc: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    depth_image = np.repeat(organized_pc[:, :, 2:3], 3, axis = 2)
    depth_image = ((depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image)) * 255.0).astype(np.uint8)
    depth_image = cv2.resize(depth_image, dsize = (target_width, target_height))
    return depth_image.astype(np.float32) / 255.0


class EasyNetMVTec3DDataset(Dataset):
    def __init__(self, dataset: MVTec3DDataset, anomaly_source_paths: Optional[list[str]] = None) -> None:
        self.dataset = dataset
        self.resize_shape = tuple(dataset.dataset_arguments.img_size)

        self.anomaly_generator = None
        if dataset.split == Split.TRAIN:
            if anomaly_source_paths is None:
                raise ValueError("anomaly_source_paths is required")
            self.anomaly_generator = PerlinAnomalyGenerator(self.resize_shape, anomaly_source_paths, foreground_threshold = 0.001)

    @property
    def samples(self) -> pd.DataFrame:
        return self.dataset.samples

    def load_rgb(self, index: int) -> np.ndarray:
        rgb = cv2.imread(self.samples.iloc[index].image_path, cv2.IMREAD_COLOR)
        rgb = cv2.resize(rgb, dsize = (self.resize_shape[1], self.resize_shape[0]))
        return rgb.astype(np.float32) / 255.0

    def load_depth(self, index: int) -> np.ndarray:
        organized_pc = tifffile.imread(self.samples.iloc[index].pc_path)
        return organized_pc_to_depth_image(organized_pc, self.resize_shape[0], self.resize_shape[1])

    def load_mask(self, index: int) -> np.ndarray:
        mask_path = self.samples.iloc[index].mask_path
        if pd.isna(mask_path):
            return np.zeros(self.resize_shape, dtype = np.uint8)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(mask, dsize = (self.resize_shape[1], self.resize_shape[0]))

    def load_pro_mask(self, index: int) -> np.ndarray:
        return self.load_mask(index).astype(np.float32) / 255.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rgb = self.load_rgb(index)
        depth = self.load_depth(index)
        sample_row = self.samples.iloc[index]
        is_normal = sample_row.label_index == LabelName.NORMAL.value

        if self.anomaly_generator is not None and is_normal:
            augmented_image, augmented_depth, mask = self.anomaly_generator.augment_image_and_depth(rgb, depth)
            image_t = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float()
            depth_t = torch.from_numpy(np.transpose(depth, (2, 0, 1))).float()
            aug_image_t = torch.from_numpy(np.transpose(augmented_image, (2, 0, 1))).float()
            aug_depth_t = torch.from_numpy(np.transpose(augmented_depth, (2, 0, 1))).float()
            mask_t = torch.from_numpy(np.transpose(mask, (2, 0, 1))).float()
            return image_t, depth_t, aug_image_t, aug_depth_t, mask_t

        image_t = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float()
        depth_t = torch.from_numpy(np.transpose(depth, (2, 0, 1))).float()
        mask = self.load_mask(index)
        mask_t = torch.from_numpy((mask == 255).astype(np.float32)).unsqueeze(0)

        if self.anomaly_generator is not None:
            return image_t, depth_t, image_t.clone(), depth_t.clone(), mask_t

        return image_t, depth_t, int(sample_row.label_index), mask_t.int(), sample_row.image_path