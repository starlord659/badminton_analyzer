import os
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# --- DYNAMICALLY IMPORT THE CLASSIFIER AND DATASET CLASSES ---
try:
    from train_transformer_pose_only import TransformerClassifier, KeypointDataset
except ImportError:
    from train_transformer_classifier import TransformerClassifier, KeypointDataset

# --- Configuration (Defaults can be overridden by CLI args) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Hyperparameters (Must match training scripts) ---
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.4
BATCH_SIZE = 64

def main(model_path, data_path, input_size):
    print(f"--- Starting Evaluation ---")
    print(f"Model Path: {model_path}")
    print(f"Data Path: {data_path}")
    print(f"Expected Input Size: {input_size}")
    print(f"Using device: {DEVICE}")

    # --- Load Mappings and Data ---
    class_map_path = "models/class_map.json"
    with open(class_map_path, 'r', encoding='utf-8') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = list(idx_to_class.values())
    num_classes = len(class_names)

    # --- Load the validation set from the specified data path ---
    file_paths, labels = [], []
    for class_name in class_names:
        class_dir = os.path.join(data_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Class directory not found, skipping: {class_dir}")
            continue
        for data_file in os.listdir(class_dir):
            if data_file.endswith('.npy'):
                file_paths.append(os.path.join(class_dir, data_file))
                labels.append(class_name)
    
    if not file_paths:
        print(f"Error: No .npy files found in {data_path}. Aborting.")
        return

    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Calculate normalization stats from the corresponding training set
    print("Calculating normalization stats from training split...")
    all_train_data = np.vstack([np.load(f) for f in tqdm(train_files, desc="Loading train files for norm")])
    mean = np.mean(all_train_data, axis=0)
    std = np.std(all_train_data, axis=0)
    std[std == 0] = 1.0

    val_dataset = KeypointDataset(val_files, val_labels, class_to_idx, mean=mean, std=std)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Load Transformer Model ---
    model = TransformerClassifier(input_size, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, num_classes, DROPOUT)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Transformer model loaded successfully from {model_path}.")

    # --- Run Evaluation ---
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sequences, lbs in tqdm(val_loader, desc="Evaluating model"):
            sequences, lbs = sequences.to(DEVICE), lbs.to(DEVICE)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(lbs.cpu().numpy())

    # --- Display Results ---
    model_name = os.path.basename(model_path)
    print("\n" + "="*20 + f" EVALUATION REPORT: {model_name} " + "="*20)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3, zero_division=0))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer classifier.")
    parser.add_argument("--model_path", type=str, default="models/transformer_pose_only_v4.pth",
                        help="Path to the trained model file (.pth).")
    parser.add_argument("--data_path", type=str, default="preprocessed_data/",
                       help="Path to the preprocessed dataset used for training the model.")
    parser.add_argument("--input_size", type=int, default=82,
                        help="The input feature size the model expects (e.g., 68 for pose-only, 73 for multi-modal).")
    
    args = parser.parse_args()
    main(args.model_path, args.data_path, args.input_size)
