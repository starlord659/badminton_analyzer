# ==============================================================================
# train_posec3d.py (Version 3 - Corrected Helper Function)
#
# This version fixes the ImportError by making the pose video creation
# a standalone, importable helper function.
# ==============================================================================
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- Constants & Configuration ---
SEQUENCE_LENGTH = 32
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 0.0005
DATASET_PATH = "data/"
MODEL_SAVE_PATH = "models/posec3d_classifier.pth"
CLASS_MAP_SAVE_PATH = "models/class_map.json"
POSE_MODEL_PATH = 'models/yolo11s-pose.pt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_KEYPOINTS = 17
HEATMAP_SIZE = (64, 64)

# --- Helper Functions ---
def get_keypoint(kpt_tensor, index):
    if kpt_tensor is not None and index < kpt_tensor.shape[0]:
        kp = kpt_tensor[index]
        if len(kp) >= 2: return kp[0].item(), kp[1].item()
    return None

def create_weighted_sampler(dataset_labels, class_to_idx):
    class_counts = {c: 0 for c in class_to_idx.keys()}
    for label in dataset_labels: class_counts[label] += 1
    num_samples = len(dataset_labels)
    weights_per_class = {c: num_samples / count for c, count in class_counts.items()}
    weights = [weights_per_class[label] for label in dataset_labels]
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples)

# --- NEW Standalone Helper Function ---
def create_pose_video_from_kpts(kpts_sequence, sequence_length, num_keypoints, heatmap_size):
    """Turns a sequence of keypoints into a stack of heatmaps (a pose video)."""
    pose_video = np.zeros((num_keypoints, sequence_length, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
    for t, kpts_frame in enumerate(kpts_sequence):
        for k, (x, y) in enumerate(kpts_frame):
            if x > 0 and y > 0:
                x_hm, y_hm = int(x * heatmap_size[1]), int(y * heatmap_size[0])
                if 0 <= x_hm < heatmap_size[1] and 0 <= y_hm < heatmap_size[0]:
                    cv2.circle(pose_video[k, t], (x_hm, y_hm), radius=2, color=1, thickness=-1)
    return pose_video

# --- Dataset Class ---
class PoseC3DDataset(Dataset):
    def __init__(self, file_paths, labels, class_to_idx, sequence_length, pose_model):
        self.file_paths, self.labels, self.class_to_idx, self.sequence_length, self.pose_model = \
            file_paths, labels, class_to_idx, sequence_length, pose_model

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        video_path, class_name = self.file_paths[idx], self.labels[idx]
        label = self.class_to_idx[class_name]
        keypoints_sequence = self.extract_keypoints_from_video(video_path)
        padded_kpts = self.pad_or_truncate_kpts(keypoints_sequence)
        
        # Call the new helper function
        pose_video = create_pose_video_from_kpts(padded_kpts, self.sequence_length, NUM_KEYPOINTS, HEATMAP_SIZE)
        
        return torch.tensor(pose_video, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        
    def extract_keypoints_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        sequence, frame_width, frame_height = [], cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = self.pose_model(frame, verbose=False, device=DEVICE)
            if len(results) > 0 and results[0].keypoints and results[0].boxes:
                largest_area, best_kpts = 0, None
                for i, box in enumerate(results[0].boxes):
                    area = (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
                    if area > largest_area:
                        largest_area, best_kpts = area, results[0].keypoints[i]
                if best_kpts is not None:
                    kpts_xy = best_kpts.data[0][:, :2].cpu().numpy()
                    kpts_xy[:, 0] /= frame_width
                    kpts_xy[:, 1] /= frame_height
                    sequence.append(kpts_xy)
        cap.release()
        return sequence

    def pad_or_truncate_kpts(self, sequence):
        if not sequence: return np.zeros((self.sequence_length, NUM_KEYPOINTS, 2), dtype=np.float32)
        if len(sequence) > self.sequence_length:
            return np.array(sequence[-self.sequence_length:], dtype=np.float32)
        else:
            padding = np.zeros((self.sequence_length - len(sequence), NUM_KEYPOINTS, 2), dtype=np.float32)
            return np.concatenate([padding, np.array(sequence, dtype=np.float32)], axis=0)

# --- Model Architecture ---
class LitePoseC3D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_block1 = nn.Sequential(nn.Conv3d(NUM_KEYPOINTS, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d((1, 2, 2)))
        self.conv_block2 = nn.Sequential(nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d((2, 2, 2)))
        self.conv_block3 = nn.Sequential(nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(), nn.MaxPool3d((2, 2, 2)))
        self.adaptive_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, num_classes))
    def forward(self, x):
        return self.classifier(self.adaptive_pool(self.conv_block3(self.conv_block2(self.conv_block1(x)))))

# --- Main Training Function ---
def main():
    print(f"Using device: {DEVICE}")
    pose_model = YOLO(POSE_MODEL_PATH)
    class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(CLASS_MAP_SAVE_PATH, 'w') as f: json.dump(class_to_idx, f)
    print(f"Found {len(class_names)} classes.")
    file_paths, labels = [], []
    for class_name in class_names:
        class_dir = os.path.join(DATASET_PATH, class_name)
        for video_file in os.listdir(class_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                file_paths.append(os.path.join(class_dir, video_file))
                labels.append(class_name)
    train_files, val_files, train_labels, val_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42, stratify=labels)
    train_dataset = PoseC3DDataset(train_files, train_labels, class_to_idx, SEQUENCE_LENGTH, pose_model)
    val_dataset = PoseC3DDataset(val_files, val_labels, class_to_idx, SEQUENCE_LENGTH, pose_model)
    print("Creating weighted sampler to handle class imbalance...")
    train_sampler = create_weighted_sampler(train_labels, class_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    model = LitePoseC3D(num_classes=len(class_names)).to(DEVICE)
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Resuming from saved model: {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_val_accuracy = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for pose_videos, lbs in pbar:
            pose_videos, lbs = pose_videos.to(DEVICE), lbs.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(pose_videos)
            loss = criterion(outputs, lbs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        avg_train_loss = running_loss / len(train_loader)
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
            for pose_videos, lbs in pbar_val:
                pose_videos, lbs = pose_videos.to(DEVICE), lbs.to(DEVICE)
                outputs = model(pose_videos)
                loss = criterion(outputs, lbs)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += lbs.size(0)
                val_correct += (predicted == lbs).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        scheduler.step(avg_val_loss)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved to {MODEL_SAVE_PATH} with accuracy: {best_val_accuracy:.2f}%")
    print("Training finished.")

if __name__ == "__main__":
    main()