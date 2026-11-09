from __future__ import annotations
import math
import cv2
import numpy as np
import imgaug.augmenters as iaa
import torch

if "sctypes" not in vars(np):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, bytes, str, np.void],
    }

def apply_autocontrast(image, cutoff):
    return iaa.pillike.autocontrast(image, cutoff.item())

def make_autocontrast():
    autocontrast = iaa.pillike.Autocontrast()
    autocontrast.func = apply_autocontrast
    return autocontrast

def linear_interpolation(x, y, w):
    final_output = (y - x) * w + x
    return final_output

def random_perlin_noise_2d(shape, res):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis = -1)

    tile_gradients = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis = 0), d[1], axis = 1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis = -1) * grad[:shape[0], :shape[1]]).sum(axis = -1)

    noise_00 = dot(tile_gradients([0, -1], [0, -1]), [0, 0])
    noise_10 = dot(tile_gradients([1, None], [0, -1]), [-1, 0])
    noise_01 = dot(tile_gradients([0, -1], [1, None]), [0, -1])
    noise_11 = dot(tile_gradients([1, None], [1, None]), [-1, -1])
    t = grid[:shape[0], :shape[1]]
    t = 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
    return math.sqrt(2) * linear_interpolation(linear_interpolation(noise_00, noise_10, t[..., 0]), linear_interpolation(noise_01, noise_11, t[..., 0]), t[..., 1])


class PerlinAnomalyGenerator:
    def __init__(
        self,
        resize_shape: tuple[int, int],
        anomaly_source_paths: list[str],
        foreground_threshold: float,
    ) -> None:
        self.resize_shape = resize_shape
        self.foreground_threshold = foreground_threshold

        self.anomaly_source_paths = list(anomaly_source_paths)
        
        self.augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel = True),
            iaa.MultiplyAndAddToBrightness(mul = (0.8, 1.2), add = (-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel = True),
            iaa.Solarize(0.5, threshold = (32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            make_autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate = (-45, 45)),
        ]
        self.rot = iaa.Sequential([iaa.Affine(rotate = (-90, 90))])

    def rand_augmenter(self) -> iaa.Sequential:
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace = False)
        chosen_augmenters = []
        for i in aug_ind:
            chosen_augmenters.append(self.augmenters[i])
        return iaa.Sequential(chosen_augmenters)

    def augment_image_and_depth(self, image: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        anomaly_source_path = self.anomaly_source_paths[torch.randint(0, len(self.anomaly_source_paths), (1,)).item()]
        aug = self.rand_augmenter()
        texture_img = cv2.imread(anomaly_source_path)
        texture_img = cv2.resize(texture_img, dsize = (self.resize_shape[1], self.resize_shape[0]))
        texture_img_augmented = aug(image = texture_img)

        perlin_scalex = 2 ** torch.randint(0, 6, (1,)).item()
        perlin_scaley = 2 ** torch.randint(0, 6, (1,)).item()
        perlin_noise = random_perlin_noise_2d((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image = perlin_noise)

        perlin_mask = np.where(perlin_noise > 0.5, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_mask = np.expand_dims(perlin_mask, axis = 2)
        perlin_noise = np.expand_dims(perlin_noise, axis = 2)

        depth_mask = np.where(depth > self.foreground_threshold, np.ones_like(depth), np.zeros_like(depth))
        perlin_mask = depth_mask.astype(np.float32) * perlin_mask

        masked_texture = texture_img_augmented.astype(np.float32) * perlin_mask / 255.0
        beta = torch.rand(1).item() * 0.8

        augmented_image = image * (1 - perlin_mask) + (1 - beta) * masked_texture + beta * image * perlin_mask
        augmented_depth = depth * (1 - perlin_mask) + perlin_mask * perlin_noise

        if torch.rand(1).item() > 0.5:
            return (image.astype(np.float32), depth.astype(np.float32), np.zeros_like(perlin_mask[:, :, -1:], dtype = np.float32))

        mask = perlin_mask[:, :, -1:]
        return augmented_image.astype(np.float32), augmented_depth.astype(np.float32), mask