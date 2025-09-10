import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
import math

# --- Constants & Configuration ---
PREPROCESSED_DATASET_PATH = "preprocessed_data/"
# --- NEW: Saving to a v4 model to reflect the new feature set ---
MODEL_SAVE_PATH = "models/transformer_pose_only_v4.pth"
CLASS_MAP_SAVE_PATH = "models/class_map.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_KEYPOINTS = 17

# --- UPDATED: Input size now reflects all new features ---
# (Pose XY + Pose Vel) + Angles + Accel = ((17*2)*2) + 6 + 8 = 82
INPUT_SIZE = 82

# --- Model Hyperparameters ---
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.4

# --- Training Hyperparameters ---
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 2e-5

# --- Data Augmentation and Dataset Classes ---
def augment_keypoints(keypoints):
    """
    Augments only the pose *position* data (first 34 features),
    leaving velocity, angles, and acceleration untouched.
    """
    seq_len, num_features = keypoints.shape
    num_pos_features = NUM_KEYPOINTS * 2
    
    pose_position_features = keypoints[:, :num_pos_features]
    other_features = keypoints[:, num_pos_features:] # Velocity, angles, accel
    
    pose_keypoints_reshaped = pose_position_features.reshape(seq_len, NUM_KEYPOINTS, 2)
    
    # Augment only the position keypoints
    noise = np.random.normal(0, 0.01, pose_keypoints_reshaped.shape)
    augmented_kpts = pose_keypoints_reshaped + noise
    shift = np.random.uniform(-0.05, 0.05, (1, 1, 2))
    augmented_kpts += shift
    augmented_kpts = np.clip(augmented_kpts, 0, 1)
    
    augmented_pose = augmented_kpts.reshape(seq_len, num_pos_features)
    
    # Recombine with original dynamic features
    return np.hstack([augmented_pose, other_features])

class KeypointDataset(Dataset):
    def __init__(self, file_paths, labels, class_to_idx, augment=False, mean=None, std=None):
        self.file_paths = file_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.augment = augment
        self.mean = torch.tensor(mean, dtype=torch.float32) if mean is not None else None
        self.std = torch.tensor(std, dtype=torch.float32) if std is not None else None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        class_name = self.labels[idx]
        keypoints = np.load(file_path)
        label = self.class_to_idx[class_name]
        
        if self.augment:
            keypoints = augment_keypoints(keypoints)
        
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
        
        if self.mean is not None and self.std is not None:
            keypoints_tensor = (keypoints_tensor - self.mean) / self.std
            
        return keypoints_tensor, torch.tensor(label, dtype=torch.long)

# --- Transformer Model (PositionalEncoding and TransformerClassifier classes remain the same) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, num_classes, dropout):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.decoder(output)
        return output

# --- Main Training Function ---
def main():
    print(f"Using device: {DEVICE}")
    class_names = sorted([d for d in os.listdir(PREPROCESSED_DATASET_PATH) if os.path.isdir(os.path.join(PREPROCESSED_DATASET_PATH, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(CLASS_MAP_SAVE_PATH, 'w') as f: json.dump(class_to_idx, f, indent=4)
    print(f"Found {len(class_names)} classes. Model will be saved to {MODEL_SAVE_PATH}")

    file_paths, labels = [], []
    for class_name in class_names:
        class_dir = os.path.join(PREPROCESSED_DATASET_PATH, class_name)
        for data_file in os.listdir(class_dir):
            if data_file.endswith('.npy'):
                file_paths.append(os.path.join(class_dir, data_file))
                labels.append(class_name)

    train_files, val_files, train_labels, val_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42, stratify=labels)
    
    print("Calculating normalization statistics from training data...")
    all_train_data = np.vstack([np.load(f) for f in tqdm(train_files, desc="Loading train files for norm")])
    mean = np.mean(all_train_data, axis=0)
    std = np.std(all_train_data, axis=0)
    std[std == 0] = 1.0
    print("Normalization stats calculated.")

    train_dataset = KeypointDataset(train_files, train_labels, class_to_idx, augment=True, mean=mean, std=std)
    val_dataset = KeypointDataset(val_files, val_labels, class_to_idx, augment=False, mean=mean, std=std)
    
    class_counts = Counter(train_labels)
    weights = torch.tensor([len(train_labels) / class_counts[c] for c in class_names], dtype=torch.float32).to(DEVICE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = TransformerClassifier(INPUT_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, len(class_names), DROPOUT).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    best_val_accuracy = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for sequences, lbs in train_pbar:
            sequences, lbs = sequences.to(DEVICE), lbs.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, lbs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
            for sequences, lbs in val_pbar:
                sequences, lbs = sequences.to(DEVICE), lbs.to(DEVICE)
                outputs = model(sequences)
                loss = criterion(outputs, lbs)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += lbs.size(0)
                val_correct += (predicted == lbs).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        
        scheduler.step(avg_val_loss)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with accuracy: {best_val_accuracy:.2f}%")

    print("Training finished.")

if __name__ == "__main__":
    main()
