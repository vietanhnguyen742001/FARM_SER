# import logging
# import os
# import sys
# import time # Import module time

# lib_path = os.path.abspath("").replace("scripts", "src")
# sys.path.append(lib_path)

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )

# import csv
# import glob
# import argparse
# import torch
# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from sklearn.metrics import (
#     balanced_accuracy_score,
#     accuracy_score,
#     confusion_matrix,
#     f1_score,
# )
# from data.dataloader import build_train_test_dataset
# from tqdm.auto import tqdm
# from models import networks
# from configs.base import Config
# from collections import Counter
# from typing import Tuple, Dict, Any

# # --- Helper Functions ---
# def calculate_accuracy(y_true, y_pred) -> Tuple[float, float]:
#     """Calculates balanced accuracy and standard accuracy."""
#     class_weights = {cls: 1.0 / count for cls, count in Counter(y_true).items()}
#     bacc = float(
#         balanced_accuracy_score(
#             y_true, y_pred, sample_weight=[class_weights[cls] for cls in y_true]
#         )
#     )
#     acc = float(accuracy_score(y_true, y_pred))
#     return bacc, acc


# def calculate_f1_score(y_true, y_pred) -> Tuple[float, float]:
#     """Calculates macro and weighted F1-scores."""
#     macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
#     weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
#     return macro_f1, weighted_f1


# def plot_confusion_matrix(cm, labels, save_path, title="Confusion Matrix"):
#     """Plots and saves a confusion matrix."""
#     cmn = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]) * 100

#     ax = plt.subplots(figsize=(8, 5.5))[1]
#     sns.heatmap(
#         cmn,
#         cmap="YlOrBr",
#         annot=True,
#         square=True,
#         linecolor="black",
#         linewidths=0.75,
#         ax=ax,
#         fmt=".2f",
#         annot_kws={"size": 16},
#     )
#     ax.set_xlabel("Predicted", fontsize=18, fontweight="bold")
#     ax.xaxis.set_label_position("bottom")
#     ax.xaxis.set_ticklabels(labels, fontsize=16)
#     ax.set_ylabel("Ground Truth", fontsize=18, fontweight="bold")
#     ax.yaxis.set_ticklabels(labels, fontsize=16)
#     plt.title(title, fontsize=20, fontweight="bold")
#     plt.tight_layout()
#     plt.savefig(save_path, format="png", dpi=1200)
#     plt.close() # Close plot to free memory


# def find_checkpoint_folder(path):
#     """Recursively finds folders containing checkpoint structure."""
#     list_candidates = []
#     if os.path.isdir(path): # Ensure it's a directory
#         # Check for direct checkpoint structure
#         if all(os.path.exists(os.path.join(path, sub_dir)) for sub_dir in ["logs", "weights", "cfg.log"]):
#             list_candidates.append(path)
#         else:
#             # Recursively search in subdirectories
#             for c in os.listdir(path):
#                 list_candidates.extend(find_checkpoint_folder(os.path.join(path, c)))
#     return list_candidates


# # --- Main Evaluation Function ---
# def evaluate_model(cfg: Config, checkpoint_path: str, all_state_dict: bool = True, cm_output: bool = False) -> Dict[str, Any]:
#     """
#     Evaluates a model and returns various metrics including inference time.

#     Args:
#         cfg: Configuration object.
#         checkpoint_path: Path to the model checkpoint.
#         all_state_dict: True if checkpoint is full state_dict, False if just model weights.
#         cm_output: Whether to output confusion matrix.

#     Returns:
#         A dictionary containing evaluation metrics and inference time.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logging.info(f"Loading model '{cfg.model_type}' from checkpoint: {checkpoint_path}")

#     network = getattr(networks, cfg.model_type)(cfg)
#     network.to(device)

#     # Load weights
#     weight = torch.load(checkpoint_path, map_location=device)
#     if all_state_dict:
#         # Assuming your saved state_dict is under 'state_dict_network' key
#         if 'state_dict_network' in weight:
#             network.load_state_dict(weight["state_dict_network"])
#         else:
#             # Fallback if the key doesn't exist, try direct load
#             logging.warning("Key 'state_dict_network' not found in checkpoint. Attempting direct load.")
#             network.load_state_dict(weight)
#     else:
#         network.load_state_dict(weight)
    
#     network.eval() # Set model to evaluation mode
    
#     # Build dataset
#     logging.info(f"Building test dataset from {cfg.data_root} for '{cfg.data_valid}'...")
#     _, test_ds = build_train_test_dataset(cfg)
    
#     y_true = []
#     y_pred = []
    
#     start_time = time.perf_counter() # Start timing inference
#     num_samples = 0

#     with torch.no_grad(): # Disable gradient calculation for inference
#         for every_test_list in tqdm(test_ds, desc="Inferencing"):
#             input_ids, audio, label = every_test_list
#             input_ids = input_ids.to(device)
#             audio = audio.to(device)
#             label = label.to(device)

#             output = network(input_ids, audio)[0] # Get logits
#             _, preds = torch.max(output, 1) # Get predicted class
            
#             y_true.extend(label.detach().cpu().numpy())
#             y_pred.extend(preds.detach().cpu().numpy())
#             num_samples += label.size(0) # Count actual samples, not batches

#     end_time = time.perf_counter() # End timing inference

#     total_inference_time = end_time - start_time
#     avg_inference_time_per_sample = (total_inference_time / num_samples) if num_samples > 0 else 0

#     logging.info(f"Total inference time: {total_inference_time:.4f} seconds")
#     logging.info(f"Average inference time per sample: {avg_inference_time_per_sample:.6f} seconds")

#     # Calculate metrics
#     bacc, acc = calculate_accuracy(y_true, y_pred)
#     macro_f1, weighted_f1 = calculate_f1_score(y_true, y_pred)
    
#     results = {
#         "bacc": bacc,
#         "acc": acc,
#         "macro_f1": macro_f1,
#         "weighted_f1": weighted_f1,
#         "total_inference_time_seconds": total_inference_time,
#         "avg_inference_time_per_sample_seconds": avg_inference_time_per_sample,
#         "num_samples_tested": num_samples,
#     }

#     # Plot Confusion Matrix if requested
#     if cm_output:
#         cm_matrix = confusion_matrix(y_true, y_pred)
#         logging.info(f"Confusion Matrix: \n{cm_matrix}")
        
#         label_names = ["Anger", "Happiness", "Sadness", "Neutral"]
#         if cfg.num_classes != 4:
#             try:
#                 with open(os.path.join(cfg.data_root, "classes.json"), "r") as f:
#                     label_data = json.load(f)
#                     label_names = list(label_data.keys())
#             except FileNotFoundError:
#                 logging.warning(f"classes.json not found at {os.path.join(cfg.data_root, 'classes.json')}. Using default labels.")

#         cm_save_path = f"confusion_matrix_{cfg.name}_{os.path.basename(cfg.data_valid)}.png"
#         plot_confusion_matrix(cm_matrix, label_names, cm_save_path, 
#                               title=f"Confusion Matrix for {cfg.name} on {os.path.basename(cfg.data_valid)}")
#         logging.info(f"Confusion matrix saved to {cm_save_path}")

#     return results


# # --- Main Execution Logic ---
# def main(args):
#     logging.info("Starting evaluation process...")
    
#     if not args.recursive:
#         # Single checkpoint evaluation
#         cfg_path = os.path.join(args.checkpoint_path, "cfg.log")
#         if not os.path.exists(cfg_path):
#             logging.error(f"Config file not found at {cfg_path}. Exiting.")
#             sys.exit(1)

#         # Determine checkpoint file and load type
#         ckpt_pt_path = os.path.join(args.checkpoint_path, "weights", "best_acc", "checkpoint_0_0.pt")
#         ckpt_pth_path = os.path.join(args.checkpoint_path, "weights", "best_acc", "checkpoint_0.pth")
        
#         all_state_dict = True
#         ckpt_to_load = ""

#         if args.latest:
#             # Look for any .pt or .pth in weights folder if latest is requested
#             latest_pt = glob.glob(os.path.join(args.checkpoint_path, "weights", "*.pt"))
#             latest_pth = glob.glob(os.path.join(args.checkpoint_path, "weights", "*.pth"))
            
#             if latest_pt:
#                 ckpt_to_load = latest_pt[0] # Assuming only one .pt or pick the latest by mtime
#                 all_state_dict = True
#             elif latest_pth:
#                 ckpt_to_load = latest_pth[0] # Assuming only one .pth or pick the latest by mtime
#                 all_state_dict = False
#             else:
#                 logging.error(f"No checkpoint found in {os.path.join(args.checkpoint_path, 'weights')}. Exiting.")
#                 sys.exit(1)
#         else:
#             # Default to best_acc/checkpoint_0_0.pt or .pth
#             if os.path.exists(ckpt_pt_path):
#                 ckpt_to_load = ckpt_pt_path
#                 all_state_dict = True
#             elif os.path.exists(ckpt_pth_path):
#                 ckpt_to_load = ckpt_pth_path
#                 all_state_dict = False
#             else:
#                 logging.error(f"Best_acc checkpoint not found in {os.path.join(args.checkpoint_path, 'weights/best_acc')}. Exiting.")
#                 sys.exit(1)

#         # Load config
#         cfg = Config()
#         cfg.load(cfg_path)
        
#         # Override dataset settings if specified
#         test_set = args.test_set if args.test_set is not None else "test.pkl"
#         cfg.data_valid = test_set
#         if args.data_root is not None:
#             if args.data_name is None:
#                 logging.error("Change validation dataset requires --data_name argument.")
#                 sys.exit(1)
#             cfg.data_root = args.data_root
#             cfg.data_name = args.data_name
        
#         # Perform evaluation
#         results = evaluate_model(cfg, ckpt_to_load, all_state_dict=all_state_dict, cm_output=args.confusion_matrix)

#         # Log and save results for single evaluation
#         logging.info(
#             f"\n--- Evaluation Results for {cfg.name} on {os.path.basename(cfg.data_valid)} ---"
#             f"\nBACC: {results['bacc'] * 100:.2f}%"
#             f"\nACC: {results['acc'] * 100:.2f}%"
#             f"\nMACRO_F1: {results['macro_f1'] * 100:.2f}%"
#             f"\nWEIGHTED_F1: {results['weighted_f1'] * 100:.2f}%"
#             f"\nTotal Inference Time: {results['total_inference_time_seconds']:.4f} seconds"
#             f"\nAvg. Inference Time per Sample: {results['avg_inference_time_per_sample_seconds']:.6f} seconds"
#         )
        
#         output_json_path = os.path.join(args.checkpoint_path, f"evaluation_results_{os.path.basename(test_set)}.json")
#         with open(output_json_path, 'w') as f:
#             json.dump(results, f, indent=4)
#         logging.info(f"Evaluation results saved to {output_json_path}")

#     else: # Recursive evaluation mode
#         logging.info(f"Searching for checkpoints recursively in: {args.checkpoint_path}")
#         list_checkpoints = find_checkpoint_folder(args.checkpoint_path)
        
#         if not list_checkpoints:
#             logging.warning(f"No checkpoint folders found in {args.checkpoint_path} or its subdirectories.")
#             return

#         test_set = args.test_set if args.test_set is not None else "test.pkl"
#         csv_path = os.path.join(args.checkpoint_path, f"recursive_evaluation_{os.path.basename(test_set)}.csv")
#         json_summary_path = os.path.join(args.checkpoint_path, f"recursive_evaluation_summary_{os.path.basename(test_set)}.json")

#         fields = ["BACC", "ACC", "MACRO_F1", "WEIGHTED_F1", 
#                   "TOTAL_INFERENCE_TIME_SECONDS", "AVG_INFERENCE_TIME_PER_SAMPLE_SECONDS", "NUM_SAMPLES_TESTED", 
#                   "MODEL_NAME", "SETTINGS", "TIMESTAMP", "CHECKPOINT_PATH_RELATIVE"]
        
#         all_results_for_json = {}

#         with open(csv_path, "w", newline='') as csvfile: # Use 'w' to create new file, 'newline=' for proper CSV
#             writer = csv.DictWriter(csvfile, fieldnames=fields)
#             writer.writeheader()

#             for ckpt_folder in tqdm(list_checkpoints, desc="Processing checkpoints"):
#                 try:
#                     cfg_path = os.path.join(ckpt_folder, "cfg.log")
#                     if not os.path.exists(cfg_path):
#                         logging.warning(f"Skipping {ckpt_folder}: cfg.log not found.")
#                         continue

#                     meta_info_parts = ckpt_folder.split(os.sep)
#                     # Assuming path structure is .../model_name/settings/timestamp
#                     timestamp = meta_info_parts[-1]
#                     settings = meta_info_parts[-2]
#                     model_name = meta_info_parts[-3]
                    
#                     logging.info(f"Evaluating: {model_name}/{settings}/{timestamp}")

#                     ckpt_to_load = ""
#                     all_state_dict = True

#                     if args.latest:
#                         latest_pt = glob.glob(os.path.join(ckpt_folder, "weights", "*.pt"))
#                         latest_pth = glob.glob(os.path.join(ckpt_folder, "weights", "*.pth"))
#                         if latest_pt:
#                             ckpt_to_load = latest_pt[0]
#                             all_state_dict = True
#                         elif latest_pth:
#                             ckpt_to_load = latest_pth[0]
#                             all_state_dict = False
#                         else:
#                             logging.warning(f"No latest checkpoint found in {os.path.join(ckpt_folder, 'weights')}. Skipping.")
#                             continue
#                     else:
#                         best_acc_pt = os.path.join(ckpt_folder, "weights", "best_acc", "checkpoint_0_0.pt")
#                         best_acc_pth = os.path.join(ckpt_folder, "weights", "best_acc", "checkpoint_0.pth")
#                         if os.path.exists(best_acc_pt):
#                             ckpt_to_load = best_acc_pt
#                             all_state_dict = True
#                         elif os.path.exists(best_acc_pth):
#                             ckpt_to_load = best_acc_pth
#                             all_state_dict = False
#                         else:
#                             logging.warning(f"Best_acc checkpoint not found in {os.path.join(ckpt_folder, 'weights/best_acc')}. Skipping.")
#                             continue

#                     cfg = Config()
#                     cfg.load(cfg_path)
#                     cfg.data_valid = test_set # Apply test set from argument
#                     if args.data_root is not None:
#                         cfg.data_root = args.data_root
#                         cfg.data_name = args.data_name

#                     results = evaluate_model(cfg, ckpt_to_load, all_state_dict=all_state_dict, cm_output=args.confusion_matrix)
                    
#                     row_data = {
#                         "BACC": round(results["bacc"] * 100, 2),
#                         "ACC": round(results["acc"] * 100, 2),
#                         "MACRO_F1": round(results["macro_f1"] * 100, 2),
#                         "WEIGHTED_F1": round(results["weighted_f1"] * 100, 2),
#                         "TOTAL_INFERENCE_TIME_SECONDS": round(results["total_inference_time_seconds"], 4),
#                         "AVG_INFERENCE_TIME_PER_SAMPLE_SECONDS": round(results["avg_inference_time_per_sample_seconds"], 6),
#                         "NUM_SAMPLES_TESTED": results["num_samples_tested"],
#                         "MODEL_NAME": model_name,
#                         "SETTINGS": settings,
#                         "TIMESTAMP": timestamp,
#                         "CHECKPOINT_PATH_RELATIVE": os.path.relpath(ckpt_folder, start=args.checkpoint_path) # Relative path
#                     }
#                     writer.writerow(row_data)
#                     all_results_for_json[os.path.relpath(ckpt_folder, start=args.checkpoint_path)] = row_data

#                     logging.info(
#                         f"\n--- Results for {model_name}/{settings}/{timestamp} ---"
#                         f"\nBACC | ACC | MACRO_F1 | WEIGHTED_F1 | Total Inf. Time | Avg. Inf. Time/Sample"
#                         f"\n{row_data['BACC']:.2f} & {row_data['ACC']:.2f} & {row_data['MACRO_F1']:.2f} & {row_data['WEIGHTED_F1']:.2f} & {row_data['TOTAL_INFERENCE_TIME_SECONDS']:.4f} & {row_data['AVG_INFERENCE_TIME_PER_SAMPLE_SECONDS']:.6f}"
#                     )
#                 except Exception as e:
#                     logging.error(f"Error evaluating checkpoint {ckpt_folder}: {e}", exc_info=True)
#                     continue # Continue to next checkpoint even if one fails

#         logging.info(f"All recursive evaluation results saved to {csv_path}")
#         with open(json_summary_path, 'w') as f:
#             json.dump(all_results_for_json, f, indent=4)
#         logging.info(f"All recursive evaluation results (JSON) saved to {json_summary_path}")


# # --- Argument Parser ---
# def arg_parser():
#     parser = argparse.ArgumentParser(description="Evaluate a trained emotion recognition model.")
#     parser.add_argument(
#         "-ckpt", "--checkpoint_path", type=str, required=True, help="Path to a specific checkpoint folder or a root folder for recursive search."
#     )
#     parser.add_argument(
#         "-r", "--recursive", action="store_true", help="Whether to search for checkpoints in child folders recursively."
#     )
#     parser.add_argument(
#         "-l", "--latest", action="store_true", help="Whether to use the latest weight file in the 'weights' folder, instead of 'best_acc/checkpoint_0_0.pt/pth'."
#     )
#     parser.add_argument(
#         "-t", "--test_set", type=str, default="test.pkl", help="Name of the testing dataset file (e.g., 'test.pkl')."
#     )
#     parser.add_argument(
#         "-cm", "--confusion_matrix", action="store_true", help="Whether to export confusion matrix as a PNG image."
#     )
#     parser.add_argument(
#         "--data_root", type=str, default=None, help="Optional: Override the data_root path specified in cfg.log for the validation dataset."
#     )
#     parser.add_argument(
#         "--data_name", type=str, default=None, help="Optional: Override the data_name specified in cfg.log when changing data_root."
#     )
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = arg_parser()
    
#     # If not recursive, we simply pass the args to the main function,
#     # and the main function handles single checkpoint evaluation logic.
#     # The 'if not args.recursive' block within main handles that.
#     main(args)
import logging
import os
import sys

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
import csv
import glob
import argparse
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import svm
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from data.dataloader import build_train_test_dataset
from tqdm.auto import tqdm
from models import networks
from configs.base import Config
from collections import Counter
from typing import Tuple


def calculate_accuracy(y_true, y_pred) -> Tuple[float, float]:
    class_weights = {cls: 1.0 / count for cls, count in Counter(y_true).items()}
    bacc = float(
        balanced_accuracy_score(
            y_true, y_pred, sample_weight=[class_weights[cls] for cls in y_true]
        )
    )
    acc = float(accuracy_score(y_true, y_pred))
    return bacc, acc


def calculate_f1_score(y_true, y_pred) -> Tuple[float, float]:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    return macro_f1, weighted_f1


def eval(cfg, checkpoint_path, all_state_dict=True, cm=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = getattr(networks, cfg.model_type)(cfg)
    network.to(device)

    # Build dataset
    _, test_ds = build_train_test_dataset(cfg)
    weight = torch.load(checkpoint_path, map_location=torch.device(device))
    if all_state_dict:
        weight = weight["state_dict_network"]

    network.load_state_dict(weight)
    network.eval()
    network.to(device)

    y_actu = []
    y_pred = []

    for every_test_list in tqdm(test_ds):
        input_ids, audio, label = every_test_list
        input_ids = input_ids.to(device)
        audio = audio.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = network(input_ids, audio)[0]
            _, preds = torch.max(output, 1)
            y_actu.append(label.detach().cpu().numpy()[0])
            y_pred.append(preds.detach().cpu().numpy()[0])
    bacc, acc = calculate_accuracy(y_actu, y_pred)
    macro_f1, weighted_f1 = calculate_f1_score(y_actu, y_pred)
    if cm:
        cm = confusion_matrix(y_actu, y_pred)
        print("Confusion Matrix: \n", cm)
        cmn = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]) * 100

        ax = plt.subplots(figsize=(8, 5.5))[1]
        sns.heatmap(
            cmn,
            cmap="YlOrBr",
            annot=True,
            square=True,
            linecolor="black",
            linewidths=0.75,
            ax=ax,
            fmt=".2f",
            annot_kws={"size": 16},
        )
        ax.set_xlabel("Predicted", fontsize=18, fontweight="bold")
        ax.xaxis.set_label_position("bottom")
        label_names = ["Anger", "Happiness", "Sadness", "Neutral"]
        if cfg.num_classes != 4:
            with open(os.path.join(cfg.data_root, "classes.json"), "r") as f:
                label_data = json.load(f)
                label_names = label_data.keys()

        ax.xaxis.set_ticklabels(label_names, fontsize=16)
        ax.set_ylabel("Ground Truth", fontsize=18, fontweight="bold")
        ax.yaxis.set_ticklabels(label_names, fontsize=16)
        plt.tight_layout()
        plt.savefig(
            "confusion_matrix_" + cfg.name + cfg.data_valid + ".png",
            format="png",
            dpi=1200,
        )

    return bacc, acc, macro_f1, weighted_f1


def find_checkpoint_folder(path):
    candidate = os.listdir(path)
    if "logs" in candidate and "weights" in candidate and "cfg.log" in candidate:
        return [path]
    list_candidates = []
    for c in candidate:
        list_candidates += find_checkpoint_folder(os.path.join(path, c))
    return list_candidates


def main(args):
    logging.info("Finding checkpoints")
    list_checkpoints = find_checkpoint_folder(args.checkpoint_path)
    test_set = args.test_set if args.test_set is not None else "test.pkl"
    csv_path = os.path.basename(args.checkpoint_path) + "{}.csv".format(test_set)
    # field names
    fields = ["BACC", "ACC", "MACRO_F1", "WEIGHTED_F1", "Time", "Model", "Settings"]
    with open(csv_path, "a") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for ckpt in list_checkpoints:
            meta_info = ckpt.split("/")
            time = meta_info[-1]
            settings = meta_info[-2]
            model_name = meta_info[-3]
            logging.info("Evaluating: {}/{}/{}".format(model_name, settings, time))
            cfg_path = os.path.join(ckpt, "cfg.log")
            if args.latest:
                ckpt_path = glob.glob(os.path.join(ckpt, "weights", "*.pt"))
                if len(ckpt_path) != 0:
                    ckpt_path = ckpt_path[0]
                    all_state_dict = True
                else:
                    ckpt_path = glob.glob(os.path.join(ckpt, "weights", "*.pth"))[0]
                    all_state_dict = False

            else:
                ckpt_path = os.path.join(ckpt, "weights/best_acc/checkpoint_0_0.pt")
                all_state_dict = True
                if not os.path.exists(ckpt_path):
                    ckpt_path = os.path.join(ckpt, "weights/best_acc/checkpoint_0.pth")
                    all_state_dict = False

            cfg = Config()
            cfg.load(cfg_path)
            # Change to test set
            cfg.data_valid = test_set
            if args.data_root is not None:
                assert (
                    args.data_name is not None
                ), "Change validation dataset requires data_name"
                cfg.data_root = args.data_root
                cfg.data_name = args.data_name

            bacc, acc, macro_f1, weighted_f1 = eval(
                cfg, ckpt_path, all_state_dict=all_state_dict, cm=args.confusion_matrix
            )
            writer.writerows(
                [
                    {
                        "BACC": round(bacc * 100, 2),
                        "ACC": round(acc * 100, 2),
                        "MACRO_F1": round(macro_f1 * 100, 2),
                        "WEIGHTED_F1": round(weighted_f1 * 100, 2),
                        "Time": time,
                        "Model": model_name,
                        "Settings": settings,
                    }
                ]
            )
            logging.info(
                "\nBACC | ACC | MACRO_F1 | WEIGHTED_F1 \n{:.2f} & {:.2f} & {:.2f} & {:.2f}".format(
                    round(bacc * 100, 2),
                    round(acc * 100, 2),
                    round(macro_f1 * 100, 2),
                    round(weighted_f1 * 100, 2),
                )
            )


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ckpt", "--checkpoint_path", type=str, help="path to checkpoint folder"
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="whether to travel child folder or not",
    )

    parser.add_argument(
        "-l",
        "--latest",
        action="store_true",
        help="whether to use latest weight or best weight",
    )

    parser.add_argument(
        "-t",
        "--test_set",
        type=str,
        default=None,
        help="name of testing set. Ex: test.pkl",
    )

    parser.add_argument(
        "-cm",
        "--confusion_matrix",
        action="store_true",
        help="whether to export consution matrix or not",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="If want to change the validation dataset",
    )
    parser.add_argument(
        "--data_name", type=str, default=None, help="for changing validation dataset"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    if not args.recursive:
        cfg_path = os.path.join(args.checkpoint_path, "cfg.log")
        all_state_dict = True
        ckpt_path = os.path.join(
            args.checkpoint_path, "weights/best_acc/checkpoint_0_0.pt"
        )
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(
                args.checkpoint_path, "weights/best_acc/checkpoint_0.pth"
            )
            all_state_dict = False

        cfg = Config()
        cfg.load(cfg_path)
        # Change to test set
        test_set = args.test_set if args.test_set is not None else "test.pkl"
        cfg.data_valid = test_set
        if args.data_root is not None:
            assert (
                args.data_name is not None
            ), "Change validation dataset requires data_name"
            cfg.data_root = args.data_root
            cfg.data_name = args.data_name

        bacc, acc, macro_f1, weighted_f1 = eval(
            cfg,
            ckpt_path,
            cm=args.confusion_matrix,
            all_state_dict=all_state_dict,
        )
        logging.info(
            "\nBACC | ACC | MACRO_F1 | WEIGHTED_F1 \n{:.2f} & {:.2f} & {:.2f} & {:.2f}".format(
                round(bacc * 100, 2),
                round(acc * 100, 2),
                round(macro_f1 * 100, 2),
                round(weighted_f1 * 100, 2),
            )
        )

    else:
        main(args)
