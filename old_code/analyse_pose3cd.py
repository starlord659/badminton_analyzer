# ==============================================================================
# analyze_video_with_classifier.py (V2 - Motion Energy Trigger)
#
# This version uses a more robust "peak motion energy" trigger to detect
# a wider variety of shots, not just high-velocity ones.
# ==============================================================================
import os
import time
import json
import argparse
from collections import defaultdict, deque
import torch
import torch.nn as nn
import cv2
import numpy as np
from ultralytics import YOLO

from train_posec3d import LitePoseC3D, get_keypoint

# --- Constants & Configuration ---
SEQUENCE_LENGTH = 32
NUM_KEYPOINTS = 17
HEATMAP_SIZE = (64, 64)

# --- NEW Trigger Constants ---
MOTION_ENERGY_THRESHOLD = 150  # Tune this based on your videos. Higher values are less sensitive.
MOTION_PEAK_WINDOW = 5         # How many frames to check for a peak.
CLASSIFICATION_COOLDOWN = 45

# --- Helper Function for Inference ---
def create_pose_video_from_buffer(keypoints_buffer):
    pose_video = np.zeros((NUM_KEYPOINTS, SEQUENCE_LENGTH, HEATMAP_SIZE[0], HEATMAP_SIZE[1]), dtype=np.float32)
    for t, kpts_frame in enumerate(keypoints_buffer):
        for k, (x, y) in enumerate(kpts_frame):
            if x > 0 and y > 0:
                x_hm, y_hm = int(x * HEATMAP_SIZE[1]), int(y * HEATMAP_SIZE[0])
                if 0 <= x_hm < HEATMAP_SIZE[1] and 0 <= y_hm < HEATMAP_SIZE[0]:
                    cv2.circle(pose_video[k, t], (x_hm, y_hm), radius=2, color=1, thickness=-1)
    return pose_video

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose-weights', type=str, default='models/yolov8s-pose.pt', help='Pose model path')
    parser.add_argument('--classifier-weights', type=str, default='models/posec3d_classifier.pth', help='PoseC3D classifier model path')
    parser.add_argument('--class-map', type=str, default='models/class_map.json', help='Path to class-to-index JSON map')
    parser.add_argument('--source', type=str, default='video_to_analyze.mp4', help='Video path or 0 for webcam')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    opt, _ = parser.parse_known_args()
    return opt

def run(opt):
    device = torch.device(opt.device if torch.cuda.is_available() and opt.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    pose_model = YOLO(opt.pose_weights)
    with open(opt.class_map, 'r') as f: class_to_idx = json.load(f)
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    
    classifier_model = LitePoseC3D(len(idx_to_class)).to(device)
    classifier_model.load_state_dict(torch.load(opt.classifier_weights, map_location=device))
    classifier_model.eval()
    print("Models loaded successfully.")

    cap = cv2.VideoCapture(opt.source)
    video_fps, frame_width, frame_height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    
    out_video_name = f"{os.path.splitext(os.path.basename(opt.source))[0]}_classified_output.mp4"
    out = cv2.VideoWriter(out_video_name, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (frame_width, frame_height))
    
    # --- Data Buffers ---
    keypoint_buffers = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
    prev_kpts = {}
    last_classification = defaultdict(lambda: {'name': '', 'frame_countdown': 0})
    cooldown_timers = defaultdict(int)
    # NEW buffer for motion energy
    motion_energy_buffers = defaultdict(lambda: deque(maxlen=MOTION_PEAK_WINDOW))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        annotated_frame = frame.copy()
        for track_id in list(cooldown_timers.keys()):
            cooldown_timers[track_id] -= 1
            if cooldown_timers[track_id] <= 0: del cooldown_timers[track_id]
        
        pose_results = pose_model.track(frame, persist=True, verbose=False, device=device)
        if pose_results and pose_results[0].boxes and pose_results[0].boxes.id is not None:
            annotated_frame = pose_results[0].plot(img=annotated_frame)
            track_ids = pose_results[0].boxes.id.int().cpu().tolist()
            all_keypoints = pose_results[0].keypoints.data
            
            new_prev_kpts = {}
            for i, track_id in enumerate(track_ids):
                kpts = all_keypoints[i]
                
                kpts_xy_normalized = kpts[:, :2].cpu().numpy()
                kpts_xy_normalized[:, 0] /= frame_width
                kpts_xy_normalized[:, 1] /= frame_height
                keypoint_buffers[track_id].append(kpts_xy_normalized)
                
                # --- NEW Motion Energy Trigger Logic ---
                motion_energy, shot_detected = 0, False
                if track_id in prev_kpts:
                    current_kpts_np = kpts[:, :2].cpu().numpy()
                    prev_kpts_np = prev_kpts[track_id][:, :2].cpu().numpy()
                    valid_indices = (kpts[:, 2] > 0.1) & (prev_kpts[track_id][:, 2] > 0.1)
                    if np.any(valid_indices):
                        distances = np.linalg.norm(current_kpts_np[valid_indices] - prev_kpts_np[valid_indices], axis=1)
                        motion_energy = np.sum(distances)
                
                motion_energy_buffers[track_id].append(motion_energy)
                
                if len(motion_energy_buffers[track_id]) == MOTION_PEAK_WINDOW:
                    energies = list(motion_energy_buffers[track_id])
                    # A peak is when the middle frame's energy is the max in its window
                    # and it's above a minimum noise threshold.
                    peak_index = MOTION_PEAK_WINDOW // 2
                    if energies[peak_index] > MOTION_ENERGY_THRESHOLD and energies[peak_index] == max(energies):
                        shot_detected = True

                if shot_detected and track_id not in cooldown_timers:
                    if len(keypoint_buffers[track_id]) == SEQUENCE_LENGTH:
                        pose_video_tensor = create_pose_video_from_buffer(keypoint_buffers[track_id])
                        input_tensor = torch.tensor(pose_video_tensor).unsqueeze(0).to(device)

                        with torch.no_grad():
                            output = classifier_model(input_tensor)
                            _, predicted_idx = torch.max(output, 1)
                            predicted_class_name = idx_to_class[predicted_idx.item()]
                        
                        last_classification[track_id] = {
                            'name': predicted_class_name.replace('0','').replace('1','').replace('_',' ').strip().title(),
                            'frame_countdown': int(video_fps * 2)
                        }
                        cooldown_timers[track_id] = CLASSIFICATION_COOLDOWN

                new_prev_kpts[track_id] = kpts.clone()
            prev_kpts = new_prev_kpts

        for track_id, classification in list(last_classification.items()):
            if classification['frame_countdown'] > 0:
                # This check needs to be robust against frames where no poses are detected
                if pose_results and pose_results[0].boxes and pose_results[0].boxes.id is not None and track_id in pose_results[0].boxes.id.int().cpu().tolist():
                    box_idx = pose_results[0].boxes.id.int().cpu().tolist().index(track_id)
                    box_coords = pose_results[0].boxes.xyxy[box_idx].cpu().numpy()
                    x1, y1 = int(box_coords[0]), int(box_coords[1])
                    cv2.putText(annotated_frame, classification['name'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)
                
                last_classification[track_id]['frame_countdown'] -= 1

        out.write(annotated_frame)
        cv2.imshow("AI Badminton Coach", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_idx += 1
        print(f"Processed frame {frame_idx}", end='\r')

    cap.release(), out.release(), cv2.destroyAllWindows()
    print(f"\nAnalysis complete. Output video saved to {out_video_name}")

if __name__ == "__main__":
    run(parse_opt())