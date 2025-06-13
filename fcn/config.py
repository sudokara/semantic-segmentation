import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import vgg16, vgg19
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from torchmetrics.segmentation import MeanIoU
import wandb
from pathlib import Path
import gc
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import random
import torchvision.transforms as T

CONFIG = {
    "dataset_path": "dataset_224",
    "num_classes": 13,
    "backbone": "vgg16",
    "pretrained": True,
    "image_size": (224, 224),
    "batch_size": 8,
    "num_workers": 4,
    "learning_rate": 1e-4,
    "num_epochs": 15,
    "val_split": 0.2,
    "device": torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    ),
    "models_dir": Path("models"),
    "wandb_project": "Segmentation_FCN",
    "norm_mean": [0.485, 0.456, 0.406],
    "norm_std": [0.229, 0.224, 0.225],
}
# Path("/scratch/shrikara").mkdir(exist_ok=True)
CONFIG["models_dir"].mkdir(exist_ok=True)
