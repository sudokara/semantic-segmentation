import torch
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torchvision
import torchvision.transforms.v2 as transforms
import einops
import cv2
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

load_dotenv()

DATASET_ROOT_DIR = os.environ["DATASET_ROOT_DIR"]

TRAIN_DATASET = DATASET_ROOT_DIR + "train/"
TEST_DATASET = DATASET_ROOT_DIR + "test/"

TRAIN_IMAGES = TRAIN_DATASET + "images/"
TRAIN_LABELS = TRAIN_DATASET + "labels/"

TEST_IMAGES = TEST_DATASET + "images/"
TEST_LABELS = TEST_DATASET + "labels/"

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, mode: str = "train", train_split: float = 0.8, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(self.root_dir, "images")
        self.labels_dir = os.path.join(self.root_dir, "labels")
        self.classes = [
            "unlabeled", "building", "fence", "other", "pedestrian", "pole", "roadline",
            "road", "sidewalk", "vegetation", "car", "wall", "traffic sign"
        ]
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
        self.class_to_idx = {c: i for i, c in self.idx_to_class.items()}

        self.images = sorted(glob.glob(os.path.join(self.images_dir, "*.png")))
        self.labels = sorted(glob.glob(os.path.join(self.labels_dir, "*.png")))
        assert len(self.images) == len(self.labels)

        last_train_index = int(train_split * len(self.images))
        if mode == "train":
            self.images = self.images[:last_train_index]
            self.labels = self.labels[:last_train_index]
        else:
            self.images = self.images[last_train_index:]
            self.labels = self.labels[last_train_index:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.labels[idx]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        label = cv2.imread(label_path, cv2.IMREAD_COLOR_RGB)[:,:,0]

        im_tensor = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])(image)
        label_tensor = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])(label)

        if self.transform:
            im_tensor, label_tensor = self.transform(im_tensor, label_tensor)

        return im_tensor, label_tensor, image_path


def get_dataloaders(train_split: float = 0.8, batch_sizes: list = [32, 32, 32], num_workers: int = 4, return_datasets=False):
    train_dataset = SegmentationDataset(TRAIN_DATASET, mode="train", train_split=train_split)
    val_dataset = SegmentationDataset(TRAIN_DATASET, mode="val", train_split=train_split)
    test_dataset = SegmentationDataset(TEST_DATASET, mode="test", train_split=0)

    train_loader = DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_sizes[1], shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes[2], shuffle=False, num_workers=num_workers)

    if return_datasets:
        return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    else:
        return train_loader, val_loader, test_loader
