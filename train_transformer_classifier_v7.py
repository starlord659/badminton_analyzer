import os
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import Counter
import math
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Any

# --- Logger Setup ---
def setup_logger(log_path: str):
    """Initializes a logger to save output to both console and a file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

# --- Normalization ---
def calculate_normalization_params_incremental(file_paths: List[str], input_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean and std without loading all data into memory."""
    logging.info("Calculating normalization parameters incrementally...")
    n_features = input_size
    running_sum = np.zeros(n_features, dtype=np.float64)
    running_sq_sum = np.zeros(n_features, dtype=np.float64)
    n_frames = 0
    
    for filepath in tqdm(file_paths, desc="Computing statistics"):
        data = np.load(filepath)
        running_sum += data.sum(axis=0)
        running_sq_sum += (data ** 2).sum(axis=0)
        n_frames += data.shape[0]
    
    mean = running_sum / n_frames
    variance = (running_sq_sum / n_frames) - (mean ** 2)
    std = np.sqrt(np.maximum(variance, 1e-7))
    std[std < 1e-7] = 1.0
    
    return mean.astype(np.float32), std.astype(np.float32)

# --- Dataset & Collation ---
class KeypointDataset(Dataset):
    def __init__(self, file_paths: List[str], labels: List[str], class_to_idx: Dict[str, int], 
                 mean: torch.Tensor, std: torch.Tensor, augment: bool = False, noise_std: float = 0.01):
        self.file_paths = file_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.mean = mean
        self.std = std
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        keypoints = np.load(self.file_paths[idx])
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
        
        if self.augment and self.noise_std > 0:
            noise = torch.randn_like(keypoints_tensor) * self.noise_std
            keypoints_tensor += noise
            
        keypoints_tensor = (keypoints_tensor - self.mean) / self.std
        label = self.class_to_idx[self.labels[idx]]
        return keypoints_tensor, label

def pad_collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads sequences in a batch and creates a padding mask."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # Pad sequences with zeros
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    # Create padding mask (True for padded values)
    max_len = padded_sequences.size(1)
    mask = torch.arange(max_len)[None, :] >= lengths[:, None]
    
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, labels, mask

# --- Model Architecture ---
class PositionalEncoding(nn.Module):
    """Correct PositionalEncoding module for batch_first tensors."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: Tensor, shape [batch_size, seq_len, embedding_dim]"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, input_size: int, d_model: int, nhead: int, num_encoder_layers: int, 
                 dim_feedforward: int, num_classes: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.decoder = nn.Linear(d_model, num_classes)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        src = self.input_projection(src) * math.sqrt(self.d_model)
        
        # Prepend CLS token
        batch_size = src.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        src = torch.cat((cls_tokens, src), dim=1)
        
        src = self.pos_encoder(src)

        # Adjust padding mask for CLS token
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=src.device)
        src_key_padding_mask = torch.cat((cls_mask, src_key_padding_mask), dim=1)
        
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        # Extract CLS token output for classification
        cls_output = output[:, 0, :]
        return self.decoder(cls_output)

# --- Visualization ---
def plot_results(history: Dict[str, List], cm: np.ndarray, class_names: List[str], save_dir: str):
    """Plot training history and confusion matrix."""
    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss vs. Epochs'); axes[0, 0].legend(); axes[0, 0].grid(True)
    axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy', color='green')
    axes[0, 1].set_title('Validation Accuracy vs. Epochs'); axes[0, 1].legend(); axes[0, 1].grid(True)
    axes[1, 0].plot(history['val_f1_macro'], label='F1 Macro', color='orange')
    axes[1, 0].plot(history['val_f1_weighted'], label='F1 Weighted', color='red')
    axes[1, 0].set_title('F1 Scores vs. Epochs'); axes[1, 0].legend(); axes[1, 0].grid(True)
    axes[1, 1].plot(history['learning_rate'], label='Learning Rate', color='purple')
    axes[1, 1].set_title('Learning Rate vs. Epochs'); axes[1, 1].set_yscale('log'); axes[1, 1].legend(); axes[1, 1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Final Confusion Matrix')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_final.png'), dpi=150)
    plt.close()

# --- Quantization ---
def quantize_model(model: nn.Module, model_path: str, quantized_model_path: str):
    """Quantize model for on-device deployment."""
    logging.info("Quantizing model...")
    model.load_state_dict(torch.load(model_path))
    model.eval().cpu()
    quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    torch.save(quantized_model.state_dict(), quantized_model_path)
    
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(quantized_model_path) / (1024 * 1024)
    logging.info(f"Quantized model saved to {quantized_model_path}")
    logging.info(f"  Original size: {original_size:.2f} MB | Quantized size: {quantized_size:.2f} MB")
    logging.info(f"  Size reduction: {original_size/quantized_size:.2f}x")

# --- Main Training ---
def main(config: Dict[str, Any]):
    # Setup experiment directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config['output_dir'], f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    setup_logger(os.path.join(run_dir, 'training.log'))
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Device
    if config['training']['device'] == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config['training']['device'])
    logging.info(f"Using device: {device}")

    # Load data
    class_names = sorted([d for d in os.listdir(config['data_path']) if os.path.isdir(os.path.join(config['data_path'], d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    with open(os.path.join(run_dir, config['class_map_filename']), 'w') as f:
        json.dump(class_to_idx, f, indent=4)
    
    file_paths, labels = [], []
    for class_name in class_names:
        class_dir = os.path.join(config['data_path'], class_name)
        for data_file in os.listdir(class_dir):
            if data_file.endswith('.npy'):
                file_paths.append(os.path.join(class_dir, data_file))
                labels.append(class_name)

    # Train/val split
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels)

    # Calculate or load normalization params
    input_size = np.load(train_files[0]).shape[1] # Infer input size from first file
    norm_params_path = os.path.join(run_dir, config['normalization_params_filename'])
    mean, std = calculate_normalization_params_incremental(train_files, input_size)
    np.savez(norm_params_path, mean=mean, std=std)
    logging.info(f"Normalization params saved to {norm_params_path}")
    
    mean_tensor, std_tensor = torch.tensor(mean), torch.tensor(std) 

    # Datasets and DataLoaders
    train_dataset = KeypointDataset(train_files, train_labels, class_to_idx, mean_tensor, std_tensor,
                                    augment=config['augmentation']['enabled'], noise_std=config['augmentation']['noise_std'])
    val_dataset = KeypointDataset(val_files, val_labels, class_to_idx, mean_tensor, std_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], collate_fn=pad_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], collate_fn=pad_collate_fn, pin_memory=True)

    # Class weights for imbalanced datasets
    class_counts = Counter(train_labels)
    weights = torch.tensor([len(train_labels) / class_counts[c] for c in class_names], dtype=torch.float32).to(device)
    
    # Model, Loss, Optimizer, Scheduler
    model_cfg = config['model']
    model = TransformerClassifier(input_size, model_cfg['d_model'], model_cfg['nhead'], model_cfg['num_encoder_layers'],
                                  model_cfg['dim_feedforward'], len(class_names), model_cfg['dropout']).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler_cfg = config['scheduler']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_cfg['mode'], patience=scheduler_cfg['patience'],
                                                           factor=scheduler_cfg['factor'], verbose=True)
    
    # Training Loop
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1_macro': [], 'val_f1_weighted': [], 'learning_rate': []}
    best_val_accuracy = 0.0
    
    logging.info("Starting training...")
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        for seq, lbs, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Train]"):
            seq, lbs, mask = seq.to(device), lbs.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(seq, mask)
            loss = criterion(outputs, lbs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['gradient_clip_norm'])
            optimizer.step()
            train_loss += loss.item()
        
        history['train_loss'].append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for seq, lbs, mask in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Val]"):
                seq, lbs, mask = seq.to(device), lbs.to(device), mask.to(device)
                outputs = model(seq, mask)
                val_loss += criterion(outputs, lbs).item()
                preds = torch.max(outputs, 1)[1]
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbs.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * f1_score(all_labels, all_preds, average='micro')
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1_macro'].append(f1_macro)
        history['val_f1_weighted'].append(f1_weighted)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        scheduler.step(avg_val_loss)

        logging.info(f"\nEpoch {epoch+1}/{config['training']['epochs']} | Train Loss: {history['train_loss'][-1]:.4f} | "
                     f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | F1 Macro: {f1_macro:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_save_path = os.path.join(run_dir, "best_model.pth")
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"✓ New best model saved to {model_save_path} (Accuracy: {best_val_accuracy:.2f}%)")

    logging.info(f"Training complete. Best validation accuracy: {best_val_accuracy:.2f}%")
    
    # Final Evaluation & Artifacts
    model_save_path = os.path.join(run_dir, "best_model.pth")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seq, lbs, mask in val_loader:
            seq, lbs, mask = seq.to(device), lbs.to(device), mask.to(device)
            outputs = model(seq, mask)
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            all_labels.extend(lbs.cpu().numpy())
            
    logging.info("\n--- Final Classification Report ---\n" + 
                 classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    cm = confusion_matrix(all_labels, all_preds)
    plot_results(history, cm, class_names, run_dir)
    logging.info(f"✓ Training plots saved in {run_dir}")
    
    # Quantize best model
    quantized_model_path = os.path.join(run_dir, "best_model_quantized.pth")
    quantize_model(model, model_save_path, quantized_model_path)
    
    logging.info("="*60)
    logging.info(f"All artifacts saved successfully in:\n{run_dir}")
    logging.info("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer Classifier for keypoint sequences.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)