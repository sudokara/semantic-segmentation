from training import *
import torch
import gc
import traceback


def main():
    print(f"Using device: {CONFIG['device']}")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(CONFIG)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error initializing dataloaders: {e}")
        print("Aborting all experiments.")
        return

    experiments = [
        {
            "name": "UNet_Standard_256",
            "model_class": UNet,
            "model_params": {
                "n_channels": 3,
                "n_classes": CONFIG["num_classes"]
            }
        },
        {
            "name": "UNet_NoSkip_256",
            "model_class": UNet,
            "model_params": {
                "n_channels": 3,
                "n_classes": CONFIG["num_classes"],
                "use_skip_connections": False
            }
        },
        {
            "name": "ResidualUNet_256",
            "model_class": UNet,
            "model_params": {
                "n_channels": 3,
                "n_classes": CONFIG["num_classes"],
                "use_skip_connections": True,
                "use_residual_block": True 
            },
        },
        {
            "name": "AttentionResidualUNet_256",
            "model_class": UNet,
            "model_params": {
                "n_channels": 3,
                "n_classes": CONFIG["num_classes"],
                "use_skip_connections": True,
                "use_residual_block": True,
                "use_gated_attention": True
            },
        },
        ]

    for experiment in experiments:
        model_name = experiment["name"]
        ModelClass = experiment["model_class"]
        model_params = experiment["model_params"]
        # config_overrides = experiment.get("config_overrides", {}) # Optional: Get overrides

        print(f"\n{'='*30}")
        print(f" Running Experiment: {model_name} ")
        print(f" Model Class: {ModelClass.__name__} ")
        print(f"{'='*30}\n")

        try:
            model = ModelClass(**model_params)
            print(f"Successfully instantiated model {model_name}")
        except Exception as e:
            print(f"Error instantiating model {model_name}: {e}")
            print("Skipping this experiment.")
            traceback.print_exc()
            continue

        best_model_path = None
        try:
            best_model_path = run_training_loop(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=CONFIG,
                model_name=model_name,
            )
        except Exception as e:
            print(f"Error during training for {model_name}: {e}")
            print("Proceeding to next experiment if any.")
            traceback.print_exc()
            del model
            if CONFIG["device"] == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            continue

        if best_model_path and best_model_path.exists():
            print(
                f"\nLoading best model for {model_name} for testing/visualization from: {best_model_path}"
            )

            test_model_instance = ModelClass(
                **model_params).to(CONFIG["device"])

            try:
                test_model_instance.load_state_dict(
                    torch.load(best_model_path, map_location=CONFIG["device"])
                )

                test_loss, test_miou = test_model(
                    test_model_instance, test_loader, CONFIG, model_name
                )
                if test_loss is not None and test_miou is not None:
                    print(f"\n--- Test Results for {model_name} ---")
                    print(f"  Test Loss: {test_loss:.4f}")
                    print(f"  Test mIoU: {test_miou:.4f}")

                visualize_predictions(
                    test_model_instance,
                    val_loader,
                    CONFIG,
                    model_name
                )

            except Exception as e:
                print(
                    f"Error during testing or visualization for {model_name}: {e}."
                )
                print("Detailed error traceback:")
                traceback.print_exc()

            del test_model_instance

        else:
            print(f"\nSkipping testing and visualization for {model_name}: "
                  "No best model path found or the file doesn't exist.")

        del model  # Delete the model used for training
        if CONFIG["device"] == "cuda":
            torch.cuda.empty_cache()  # Clear GPU memory
        gc.collect()  # Trigger garbage collection

        print(f"\n--- Finished Experiment: {model_name} ---")

    print(f"\n{'='*30}")
    print(" All experiments completed. ")
    print(f"{'='*30}\n")


if __name__ == "__main__":
    main()
