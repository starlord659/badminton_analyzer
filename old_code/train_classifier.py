# ==============================================================================
# CORRECTED train_classifier.py
# COPY AND PASTE THIS ENTIRE BLOCK INTO YOUR FILE
# ==============================================================================
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- Constants & Configuration ---
SEQUENCE_LENGTH = 30
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 0.001
DATASET_PATH = "data/"
MODEL_SAVE_PATH = "models/shot_classifier.pth"
CLASS_MAP_SAVE_PATH = "models/class_map.json"

# --- Pose Estimation Model ---
# Using the correct name for the standard pose model.
# Change it back if you have a custom model named 'yolo11n-pose.pt'.
POSE_MODEL_PATH = 'models/yolov8n-pose.pt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Keypoint Constants ---
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
NUM_KEYPOINTS = 17
KEYPOINT_DIM = 2

# --- Helper function to get keypoint ---
def get_keypoint(kpt_tensor, index):
    if kpt_tensor is not None and index < kpt_tensor.shape[0]:
        kp = kpt_tensor[index]
        if len(kp) >= 2:
            return kp[0].item(), kp[1].item()
    return None

# --- Dataset Class ---
class ShotDataset(Dataset):
    """Dataset for loading video clips and extracting pose keypoint sequences."""
    def __init__(self, file_paths, labels, class_to_idx, sequence_length, pose_model):
        self.file_paths = file_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.sequence_length = sequence_length
        self.pose_model = pose_model

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        video_path = self.file_paths[idx]
        class_name = self.labels[idx]
        label = self.class_to_idx[class_name]
        keypoints_sequence = self.extract_keypoints(video_path)
        processed_sequence = self.pad_or_truncate(keypoints_sequence)
        return torch.tensor(processed_sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def normalize_keypoints(self, keypoints):
        """
        Normalize keypoints to be relative to the torso.
        This function is now correctly a method of this class.
        """
        l_shoulder = get_keypoint(keypoints, KP_LEFT_SHOULDER)
        r_shoulder = get_keypoint(keypoints, KP_RIGHT_SHOULDER)
        
        if l_shoulder and r_shoulder:
            anchor_point = np.array([(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2])
        elif l_shoulder:
            anchor_point = np.array(l_shoulder)
        elif r_shoulder:
            anchor_point = np.array(r_shoulder)
        else:
            anchor_point = np.array([0, 0])
            
        kpts_xy = keypoints.cpu().numpy()[:, :KEYPOINT_DIM]
        normalized_kpts = kpts_xy - anchor_point
        return normalized_kpts

    def extract_keypoints(self, video_path):
        """Extracts pose keypoints from a video."""
        cap = cv2.VideoCapture(video_path)
        sequence = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            results = self.pose_model(frame, verbose=False, device=DEVICE)
            
            if len(results) > 0 and results[0].keypoints and results[0].boxes:
                largest_area, best_kpts = 0, None
                for i, box in enumerate(results[0].boxes):
                    area = (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
                    if area > largest_area:
                        largest_area = area
                        best_kpts = results[0].keypoints[i]
                
                if best_kpts is not None:
                    # Correctly call the method using 'self'
                    normalized_kpts = self.normalize_keypoints(best_kpts.data[0])
                    sequence.append(normalized_kpts.flatten())

        cap.release()
        return np.array(sequence, dtype=np.float32)

    def pad_or_truncate(self, sequence):
        """Ensure all sequences have the same length."""
        if len(sequence) == 0:
             return np.zeros((self.sequence_length, NUM_KEYPOINTS * KEYPOINT_DIM), dtype=np.float32)

        if len(sequence) > self.sequence_length:
            return sequence[-self.sequence_length:]
        else:
            padding_len = self.sequence_length - len(sequence)
            padding = np.zeros((padding_len, sequence.shape[1]), dtype=np.float32)
            return np.concatenate([padding, sequence], axis=0)

# --- Model Architecture ---
class ShotClassifierLSTM(nn.Module):
    """LSTM model for shot classification."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ShotClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# --- Main Training Function ---
def main():
    print(f"Using device: {DEVICE}")
    print("Loading YOLOv8 pose model for feature extraction...")
    pose_model = YOLO(POSE_MODEL_PATH)

    print("Scanning dataset directory...")
    class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(CLASS_MAP_SAVE_PATH, 'w') as f:
        json.dump(class_to_idx, f)
    print(f"Found {len(class_names)} classes. Class map saved to {CLASS_MAP_SAVE_PATH}")

    file_paths, labels = [], []
    for class_name in class_names:
        class_dir = os.path.join(DATASET_PATH, class_name)
        for video_file in os.listdir(class_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                file_paths.append(os.path.join(class_dir, video_file))
                labels.append(class_name)

    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels)
    
    print(f"Training set size: {len(train_files)}, Validation set size: {len(val_files)}")

    train_dataset = ShotDataset(train_files, train_labels, class_to_idx, SEQUENCE_LENGTH, pose_model)
    val_dataset = ShotDataset(val_files, val_labels, class_to_idx, SEQUENCE_LENGTH, pose_model)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = ShotClassifierLSTM(NUM_KEYPOINTS * KEYPOINT_DIM, 128, 2, len(class_names)).to(DEVICE)
    
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Found existing model at {MODEL_SAVE_PATH}. Loading weights to resume training.")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    else:
        print("No existing model found. Starting training from scratch.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for sequences, lbs in pbar:
            sequences, lbs = sequences.to(DEVICE), lbs.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, lbs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += lbs.size(0)
            train_correct += (predicted == lbs).sum().item()
            pbar.set_postfix({'Loss': f"{running_loss/len(train_loader):.4f}", 'Acc': f"{100*train_correct/train_total:.2f}%"})

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
            for sequences, lbs in pbar_val:
                sequences, lbs = sequences.to(DEVICE), lbs.to(DEVICE)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                val_total += lbs.size(0)
                val_correct += (predicted == lbs).sum().item()
                val_accuracy = 100 * val_correct / val_total
                pbar_val.set_postfix({'Acc': f"{val_accuracy:.2f}%"})

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved to {MODEL_SAVE_PATH} with accuracy: {best_val_accuracy:.2f}%")

    print("Training finished.")

if __name__ == "__main__":
    main()