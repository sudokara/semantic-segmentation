from models import *


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    metric,
    device,
    epoch_num,
    total_epochs,
    global_step=0,
):
    """Runs a single training epoch with batch-wise logging."""
    model.train()
    total_loss = 0.0
    metric.reset()
    progress_bar = tqdm(
        loader, desc=f"Epoch {epoch_num}/{total_epochs} [Train]", leave=False
    )
    batch_count = 0

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_count += 1
        current_global_step = global_step + batch_count
        total_loss += loss.item()
        current_batch_loss = loss.item()

        with torch.no_grad():
            _, preds = torch.max(outputs, 1)
            metric.update(preds, labels)
            current_batch_miou = metric.compute().item()

        wandb.log(
            {
                "batch/train_loss": current_batch_loss,
                "batch/train_miou": current_batch_miou,
                "batch/global_step": current_global_step,
            }
        )

        progress_bar.set_postfix({"loss": f"{current_batch_loss:.4f}"})

    avg_loss = total_loss / len(loader)
    epoch_miou = metric.compute().item()

    return avg_loss, epoch_miou, global_step + batch_count


def validate_epoch(model, loader, criterion, metric, device, epoch_num, total_epochs):
    """Runs a single validation epoch."""
    model.eval()
    total_loss = 0.0
    metric.reset()
    progress_bar = tqdm(
        loader, desc=f"Epoch {epoch_num}/{total_epochs} [Val]", leave=False
    )

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            metric.update(preds, labels)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    epoch_miou = metric.compute().item()
    return avg_loss, epoch_miou


def run_training_loop(model, train_loader, val_loader, config, model_name):
    """Manages the overall training process for a given model."""
    print(f"\n--- Training {model_name} ---")
    device = config["device"]
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    miou_metric = MeanIoU(num_classes=config["num_classes"]).to(device)

    wandb.init(
        project=config["wandb_project"], name=model_name, config=config, reinit=True
    )

    best_val_miou = 0.0
    best_model_path = None
    global_step = 0

    for epoch in range(config["num_epochs"]):
        current_epoch = epoch + 1

        train_loss, train_miou, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            miou_metric,
            device,
            current_epoch,
            config["num_epochs"],
            global_step,
        )
        val_loss, val_miou = validate_epoch(
            model,
            val_loader,
            criterion,
            miou_metric,
            device,
            current_epoch,
            config["num_epochs"],
        )

        print(
            f"Epoch {current_epoch}/{config['num_epochs']} | "
            f"Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}"
        )

        wandb.log(
            {
                "epoch": current_epoch,
                "epoch/train_loss": train_loss,
                "epoch/train_miou": train_miou,
                "epoch/val_loss": val_loss,
                "epoch/val_miou": val_miou,
                "epoch/global_step": global_step,
            }
        )

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            model_path = config["models_dir"] / f"{model_name}_best.pth"
            torch.save(model.state_dict(), model_path)
            best_model_path = model_path
            print(
                f"   -> New best model saved with Val mIoU: {val_miou:.4f} at {model_path}"
            )

    wandb.finish()
    print(
        f"--- Training Finished for {model_name}. Best Val mIoU: {best_val_miou:.4f} ---"
    )
    return best_model_path


def test_model(model, test_loader, config, model_name):
    """Tests the model on the test set and prints metrics."""
    if test_loader is None:
        print(f"\n--- Skipping Testing for {model_name} (No Test Loader) ---")
        return None, None

    print(f"\n--- Testing {model_name} ---")
    device = config["device"]
    model.to(device)
    model.eval()

    wandb.init(
        project=config["wandb_project"],
        name=f"{model_name}_test",
        config=config,
        reinit=True,
    )

    criterion = nn.CrossEntropyLoss()
    miou_metric = MeanIoU(num_classes=config["num_classes"]).to(device)
    test_loss = 0.0
    intersection = torch.zeros(config["num_classes"], device=device)
    union = torch.zeros(config["num_classes"], device=device)
    target_counts = torch.zeros(config["num_classes"], device=device)

    progress_bar = tqdm(test_loader, desc=f"Testing {model_name}", leave=False)
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            miou_metric.update(preds, labels)

            for i in range(config["num_classes"]):
                pred_i = preds == i
                label_i = labels == i
                intersection[i] += (pred_i & label_i).sum()
                union[i] += (pred_i | label_i).sum()
                target_counts[i] += label_i.sum()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_test_loss = test_loss / len(test_loader)
    test_miou = miou_metric.compute().item()

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test mIoU (TorchMetrics): {test_miou:.4f}")

    iou_per_class = intersection / (union + 1e-8)

    class_iou_dict = {}

    print("Per-class IoU:")
    for i in range(config["num_classes"]):
        count = target_counts[i].item()
        class_iou = iou_per_class[i].item()
        class_iou_dict[f"test/class_{i}_iou"] = class_iou

        if count > 0:
            print(
                f"  Class {i}: {class_iou:.4f} (Present in {count:.0f} pixels)")
            class_iou_dict[f"test/class_{i}_presence"] = count

    wandb.log(
        {
            "test/loss": avg_test_loss,
            "test/miou": test_miou,
            **class_iou_dict,
        }
    )

    wandb.summary["test_loss"] = avg_test_loss
    wandb.summary["test_miou"] = test_miou

    data = [
        [i, iou_per_class[i].item()]
        for i in range(config["num_classes"])
        if target_counts[i].item() > 0
    ]

    if data:
        table = wandb.Table(data=data, columns=["class", "iou"])
        wandb.log(
            {
                "test/class_iou_chart": wandb.plot.bar(
                    table, "class", "iou", title="IoU by Class"
                )
            }
        )

    wandb.finish()

    return avg_test_loss, test_miou


# LLM generated function to visualize predictions
def visualize_predictions(model, val_loader, config, model_name, num_samples=5):
    """
    Visualizes model predictions on random validation samples and saves the results.

    Args:
        model: The trained model
        val_loader: Validation data loader
        config: Configuration dictionary
        model_name: Name of the model
        num_samples: Number of samples to visualize
    """
    print(f"\n--- Visualizing {model_name} predictions ---")
    device = config["device"]
    model.to(device)
    model.eval()

    # Create output directory
    vis_dir = Path(f"visualizations/{model_name}")
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Get random batch
    dataiter = iter(val_loader)
    batch_idx = random.randint(0, len(val_loader) - 1)

    # Get to the randomly selected batch
    for _ in range(batch_idx):
        images, labels = next(dataiter)

    images, labels = next(dataiter)

    # Limit to number of samples we want to visualize
    images = images[:num_samples]
    labels = labels[:num_samples]

    images = images.to(device)
    labels = labels.to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    # Create a colormap for better visualization
    # Simple colormap function - assign distinct colors to each class
    def get_colored_mask(mask, num_classes=config["num_classes"]):
        # Generate a color map with distinct colors for each class
        cmap = plt.cm.get_cmap("tab20", num_classes)
        colored_mask = torch.zeros(
            (mask.shape[0], 3, mask.shape[1], mask.shape[2]), device=mask.device
        )

        for cls in range(num_classes):
            # Get boolean mask for this class
            class_mask = mask == cls

            # Get color for this class (RGB tuple)
            color = (
                torch.tensor(cmap(cls)[:3], device=mask.device).float().view(
                    1, 3, 1, 1)
            )

            # Apply color where this class exists
            colored_mask = colored_mask + \
                class_mask.unsqueeze(1).float() * color

        return colored_mask

    # Convert tensors for visualization
    # Denormalize images
    mean = torch.tensor(config["norm_mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["norm_std"], device=device).view(1, 3, 1, 1)
    images_vis = images * std + mean
    images_vis = torch.clamp(images_vis, 0, 1)

    # Apply colormaps to masks
    labels_vis = get_colored_mask(labels)
    preds_vis = get_colored_mask(preds)

    # Save individual samples
    for i in range(num_samples):
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(images_vis[i].permute(1, 2, 0).cpu().numpy())
        plt.title("Original Image")
        plt.axis("off")

        # Ground truth
        plt.subplot(1, 3, 2)
        plt.imshow(labels_vis[i].permute(1, 2, 0).cpu().numpy())
        plt.title("Ground Truth")
        plt.axis("off")

        # Prediction
        plt.subplot(1, 3, 3)
        plt.imshow(preds_vis[i].permute(1, 2, 0).cpu().numpy())
        plt.title("Model Prediction")
        plt.axis("off")

        # Save figure
        plt.tight_layout()
        plt.savefig(vis_dir / f"sample_{i}.png", dpi=150)
        plt.close()

    # Create a grid visualization of all samples
    grid_img = torch.cat(
        [
            make_grid(images_vis, nrow=num_samples, padding=2),
            make_grid(labels_vis, nrow=num_samples, padding=2),
            make_grid(preds_vis, nrow=num_samples, padding=2),
        ],
        dim=1,
    )

    plt.figure(figsize=(15, 15))
    plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
    plt.title(f"{model_name} - Predictions vs Ground Truth")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(vis_dir / "grid_comparison.png", dpi=200)
    plt.close()

    print(f"Saved visualizations to {vis_dir}")
