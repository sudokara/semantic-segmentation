from config import *


class SegmentationDataset(Dataset):
    """Loads images and segmentation masks."""

    def __init__(self, images_dir: Path, labels_dir: Path, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = sorted(
            [f for f in os.listdir(images_dir) if f.endswith(
                (".png", ".jpg", ".jpeg"))]
        )
        self.labels = self.images

        if len(self.images) == 0:
            raise FileNotFoundError(f"No images found in {images_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label_name = self.labels[idx]

        img_path = self.images_dir / img_name
        label_path = self.labels_dir / label_name

        try:
            image = Image.open(img_path).convert("RGB")
            label = Image.open(label_path)

            label_np = np.array(label)
            if label_np.ndim == 3:
                label_np = label_np[
                    :, :, 0
                ]

            label_tensor = torch.from_numpy(label_np).long()

            if self.transform:
                image = self.transform(image)

            return image, label_tensor

        except FileNotFoundError:
            print(
                f"Warning: File not found. Image: {img_path}, Label: {label_path}")
            raise FileNotFoundError(
                f"Missing file: {img_path} or {label_path}")
        except Exception as e:
            print(f"Error loading item {idx} (Image: {img_name}): {e}")
            raise e


def get_dataloaders(config):
    """Creates train, validation, and test dataloaders."""
    base_path = Path(config["dataset_path"])
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["norm_mean"], std=config["norm_std"]),
        ]
    )

    train_val_imgs_path = base_path / "train" / "images"
    train_val_labels_path = base_path / "train" / "labels"
    if not train_val_imgs_path.exists() or not train_val_labels_path.exists():
        raise FileNotFoundError(
            f"Training data not found in {base_path}/train")
    full_train_dataset = SegmentationDataset(
        train_val_imgs_path, train_val_labels_path, transform
    )

    total_size = len(full_train_dataset)
    if total_size == 0:
        raise ValueError(
            "Full training dataset is empty. Check data paths and content."
        )
    val_size = int(config["val_split"] * total_size)
    train_size = total_size - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError(
            f"Train ({train_size}) or Val ({val_size}) split resulted in zero samples."
        )
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    test_imgs_path = base_path / "test" / "images"
    test_labels_path = base_path / "test" / "labels"
    if not test_imgs_path.exists() or not test_labels_path.exists():
        print(
            f"Warning: Test data not found in {base_path}/test. Test loader will be empty."
        )
        test_dataset = None
    else:
        test_dataset = SegmentationDataset(
            test_imgs_path, test_labels_path, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    test_loader = None
    if test_dataset and len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
        )
        print(f"Test dataset size: {len(test_dataset)}")
    elif test_dataset is None:
        print("Test dataset not found or path invalid.")
    else:
        print("Test dataset found but is empty.")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    return train_loader, val_loader, test_loader
