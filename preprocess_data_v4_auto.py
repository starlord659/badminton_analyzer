# preprocess_data_v4_auto.py
import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import torch
import sys

# Assume court_detect.py is in the project root or accessible via PYTHONPATH
from court_detect import CourtDetect

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# --- Configuration ---
POSE_MODEL_PATH = 'models/yolo11s-pose.pt'
NUM_KEYPOINTS = 17
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

KEYPOINT_MAP = { 'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4, 'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8, 'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12, 'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16 }
ANGLE_DEFINITIONS = { "R Elbow": ("right_shoulder", "right_elbow", "right_wrist"), "L Elbow": ("left_shoulder", "left_elbow", "left_wrist"), "R Shoulder": ("right_elbow", "right_shoulder", "right_hip"), "L Shoulder": ("left_elbow", "left_shoulder", "left_hip"), "R Knee": ("right_hip", "right_knee", "right_ankle"), "L Knee": ("left_hip", "left_knee", "left_ankle") }
ACCEL_KEYPOINTS = ['right_wrist', 'right_elbow', 'left_wrist', 'left_elbow']
ACCEL_KEYPOINT_INDICES = [KEYPOINT_MAP[k] for k in ACCEL_KEYPOINTS]

NUM_ANGLES = len(ANGLE_DEFINITIONS)
NUM_ACCEL_FEATURES = len(ACCEL_KEYPOINTS) * 2
# (Pose XY+Vel) + Angles + Accel + Court XY = ((17*2)*2) + 6 + 8 + 2 = 84
NUM_FEATURES_PER_FRAME = ((NUM_KEYPOINTS * 2) * 2) + NUM_ANGLES + NUM_ACCEL_FEATURES + 2

# --- Global variables for multiprocessing workers ---
pose_model = None
court_detector = None
homography_matrix = None

def init_worker(model_path, device):
    """Initializes models for each worker process."""
    global pose_model, court_detector
    pose_model = YOLO(model_path)
    pose_model.to("cpu" if device == "cuda" else device)
    # The CourtDetect model is loaded within its class constructor
    court_detector = CourtDetect()

def calculate_angle(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    if np.all(a==0) or np.all(b==0) or np.all(c==0): return 0.0
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return angle / np.pi

def get_homography_from_corners(corners):
    """Calculates homography matrix from 4 court corners."""
    src_points = np.array(corners, dtype=np.float32)
    # Map to a standard 1x1 square for normalized coordinates
    dst_points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return H

def pixel_to_court(p, H):
    """Converts pixel coordinates to normalized court coordinates using homography."""
    if H is None: return np.array([0, 0], dtype=np.float32)
    p_hom = np.array([p[0], p[1], 1], dtype=np.float32)
    p_court_hom = H @ p_hom
    return (p_court_hom / p_court_hom[2])[:2] if p_court_hom[2] != 0 else np.array([0, 0])

def extract_features_from_video(video_path):
    """Main feature extraction function for a single video."""
    global homography_matrix
    
    # --- Step 1: Automatically detect the court and get a stable set of corners ---
    court_detector.pre_process(video_path)
    if court_detector.normal_court_info is None:
        print(f"Warning: Could not find a stable court in {video_path}. Skipping.")
        return None
    
    # Extract the 4 outer corners for homography (TopLeft, TopRight, BottomLeft, BottomRight)
    # NOTE: The indices might need adjustment depending on your CourtDetect model's output order
    corners = [
        court_detector.normal_court_info[0], # Top-Left
        court_detector.normal_court_info[1], # Top-Right
        court_detector.normal_court_info[4], # Bottom-Left
        court_detector.normal_court_info[5]  # Bottom-Right
    ]
    homography_matrix = get_homography_from_corners(corners)
    
    # --- Step 2: Process video frame-by-frame to extract features ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None

    features_sequence = []
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width == 0 or frame_height == 0: return None

    previous_kpts_norm = np.zeros((NUM_KEYPOINTS, 2))
    previous_velocity = np.zeros((NUM_KEYPOINTS, 2))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = pose_model(frame, verbose=False)
        current_kpts_raw = np.zeros((NUM_KEYPOINTS, 2))
        if len(results) > 0 and results[0].keypoints and results[0].boxes and len(results[0].boxes.xyxy) > 0:
            best_kpts_obj = results[0].keypoints[results[0].boxes.xywh.prod(dim=1).argmax()]
            current_kpts_raw = best_kpts_obj.data[0][:, :2].cpu().numpy()
        
        current_kpts_norm = current_kpts_raw.copy()
        current_kpts_norm[:, 0] /= frame_width
        current_kpts_norm[:, 1] /= frame_height
        
        # Biomechanical features
        current_velocity = current_kpts_norm - previous_kpts_norm
        acceleration = current_velocity - previous_velocity
        accel_features = acceleration[ACCEL_KEYPOINT_INDICES].flatten()
        angles = [calculate_angle(current_kpts_raw[KEYPOINT_MAP[p1]], current_kpts_raw[KEYPOINT_MAP[p2]], current_kpts_raw[KEYPOINT_MAP[p3]]) for p1,p2,p3 in ANGLE_DEFINITIONS.values()]
        
        # Court position feature
        player_centroid_pixel = np.mean(current_kpts_raw[11:13], axis=0) # Avg of hips
        player_court_coords = pixel_to_court(player_centroid_pixel, homography_matrix)

        combined_features = np.concatenate([current_kpts_norm.flatten(), current_velocity.flatten(), np.array(angles), accel_features, player_court_coords.flatten()])
        features_sequence.append(combined_features)

        previous_kpts_norm = current_kpts_norm
        previous_velocity = current_velocity

    cap.release()
    return np.array(features_sequence, dtype=np.float32)

def process_video_wrapper(args):
    """Wrapper function for multiprocessing."""
    video_path, class_name, processed_data_path = args
    full_features = extract_features_from_video(video_path)
    if full_features is not None and len(full_features) > 0:
        save_dir = os.path.join(processed_data_path, class_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.npy")
        np.save(save_path, full_features)

def main(raw_data_path, processed_data_path, num_workers):
    print("Starting auto-preprocessing with CourtDetect and full sequence extraction...")
    tasks = []
    for class_name in sorted(os.listdir(raw_data_path)):
        class_dir = os.path.join(raw_data_path, class_name)
        if not os.path.isdir(class_dir): continue
        for video_file in os.listdir(class_dir):
            if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                tasks.append((os.path.join(class_dir, video_file), class_name, processed_data_path))

    with Pool(processes=num_workers, initializer=init_worker, initargs=(POSE_MODEL_PATH, DEVICE)) as pool:
        list(tqdm(pool.imap_unordered(process_video_wrapper, tasks), total=len(tasks), desc="Processing videos"))
    print(f"\nPreprocessing complete! Full sequence data saved in: {processed_data_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, default="data/", help="Path to raw videos.")
    parser.add_argument("--processed_data_path", type=str, default="preprocessed_data_v4_auto/", help="Path to save processed .npy files.")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2), help="Number of worker processes.")
    args = parser.parse_args()
    main(args.raw_data_path, args.processed_data_path, args.workers)