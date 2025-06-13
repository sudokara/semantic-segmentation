from training import *


def main():
    print(f"Using device: {CONFIG['device']}")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(CONFIG)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error initializing dataloaders: {e}")
        return

    fcn_variants = {"FCN32s": FCN32s, "FCN16s": FCN16s, "FCN8s": FCN8s}
    training_modes = ["freeze", "finetune"]

    results = {}

    for model_arch, model_class in fcn_variants.items():
        for mode in training_modes:
            freeze_backbone = mode == "freeze"
            model_name = f"{model_arch}_{CONFIG['backbone']}_{mode}"

            print(f"\n{'='*20} Starting Experiment: {model_name} {'='*20}")

            model = model_class(
                num_classes=CONFIG["num_classes"],
                backbone_name=CONFIG["backbone"],
                pretrained=CONFIG["pretrained"],
                freeze_backbone=freeze_backbone,
            )

            best_model_path = run_training_loop(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=CONFIG,
                model_name=model_name,
            )

            if best_model_path and best_model_path.exists():
                print(
                    f"Loading best model for testing from: {best_model_path}")
                test_model_instance = model_class(
                    num_classes=CONFIG["num_classes"],
                    backbone_name=CONFIG["backbone"],
                    pretrained=False,
                    freeze_backbone=freeze_backbone,
                ).to(CONFIG["device"])
                try:
                    test_model_instance.load_state_dict(
                        torch.load(best_model_path,
                                   map_location=CONFIG["device"])
                    )

                    test_loss, test_miou = test_model(
                        test_model_instance, test_loader, CONFIG, model_name
                    )
                    if test_loss is not None and test_miou is not None:
                        results[model_name] = {
                            "test_loss": test_loss,
                            "test_miou": test_miou,
                        }

                    visualize_predictions(
                        test_model_instance, val_loader, CONFIG, model_name
                    )

                except Exception as e:
                    print(
                        f"Error during testing or visualization: {e}. Skipping remaining steps."
                    )

                del test_model_instance

            del model
            if CONFIG["device"] == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            print(f"{'='*20} Finished Experiment: {model_name} {'='*20}\n")

    print("\n--- Overall Test Results ---")
    if results:
        for name, metrics in results.items():
            print(
                f"{name}: Test Loss = {metrics['test_loss']:.4f}, Test mIoU = {metrics['test_miou']:.4f}"
            )
    else:
        print("No test results recorded.")


if __name__ == "__main__":
    main()
