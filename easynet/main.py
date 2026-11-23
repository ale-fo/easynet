from __future__ import annotations
import glob
import random
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.models.training_args import TrainingArgs
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, AuPro

from easynet.datasets.mvtec3d.easynet_adapter_mvtech import EasyNetMVTec3DDataset
from easynet.datasets.mvtec3d.mvtec3d_dataset import MVTec3DDataset
from easynet.models.easynet.easynet import ENTROPY_MARGINS, EasyNet, EasyNetFusionMode
from easynet.trainers.trainer_easynet import TrainerEasynet

import imgaug


BATCH_SIZE = 8


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)
    imgaug.seed(seed)


def train_easynet():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    dataset_args = {
        "dataset_path": "/dataset/mvtec3d_preprocessed",
        "img_size": (256, 256),
        "gt_mask_size": (256, 256),
        "image_transform_list": [
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias = True),
        ],
    }

    args = {
        "category": "bagel",
        "fusion_mode": "single",
        "num_scales": 2,
        "seed": 0,
        "base_width": 128,
        "anomaly_source_path": "/dataset/dtd",
        "rgb_model_checkpoint": "/checkpoints/bagel_single_2_seed0_best.pt",
        "checkpoint_dir": "/checkpoints",
        "lr": 0.002,
        "epochs": 800,
        "evaluation_epoch_interval": 10,
    }

    set_seed(args["seed"])

    rgb_model = None
    if args["fusion_mode"] == EasyNetFusionMode.FUSION2.value:
        rgb_model = EasyNet(EasyNetFusionMode.SINGLE, num_scales = args["num_scales"], base_width = args["base_width"], device = device)
        rgb_model.load_state_dict(torch.load(args["rgb_model_checkpoint"], map_location = "cpu"))

    anomaly_source_paths = sorted(glob.glob(str(Path(args["anomaly_source_path"]) / "**" / "*.jpg"), recursive = True))
    if not anomaly_source_paths:
        raise FileNotFoundError(f"no .jpg texture below {args['anomaly_source_path']}")

    train_dataset = EasyNetMVTec3DDataset(
        MVTec3DDataset(DatasetArguments(**dataset_args), category = args["category"], split = Split.TRAIN), anomaly_source_paths)
    test_dataset = EasyNetMVTec3DDataset(MVTec3DDataset(DatasetArguments(**dataset_args), category = args["category"], split = Split.TEST))

    model = EasyNet(
        fusion_mode = EasyNetFusionMode(args["fusion_mode"]),
        num_scales = args["num_scales"],
        base_width = args["base_width"],
        entropy_margin = ENTROPY_MARGINS.get(args["category"], 0.0),
        rgb_model = rgb_model,
        device = device,
    ).to(device)

    training_args = TrainingArgs(
        batch_size = BATCH_SIZE,
        epochs = args["epochs"],
        evaluation_epoch_interval = args["evaluation_epoch_interval"],
        optimizer = torch.optim.Adam(model.parameters(), lr = args["lr"]),
        shuffle = False,
        num_workers = 1,
    )

    run_name = f"{args['category']}_{args['fusion_mode']}_{args['num_scales']}_seed{args['seed']}"
    trainer = TrainerEasynet(
        training_args,
        model,
        train_dataset,
        test_dataset,
        metrics = [
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            AuPro(MetricLvl.PIXEL),
        ],
        device = device,
        logger = None,
        save_path = f"{args['checkpoint_dir']}/{run_name}_best.pt",
    )

    trainer.train(training_args.epochs, training_args.evaluation_epoch_interval)


if __name__ == "__main__":
    train_easynet()
