# ==============================================================================
# CORRECTED analyze_video_with_classifier.py
# COPY AND PASTE THIS ENTIRE BLOCK INTO YOUR FILE
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

# --- Import only what's needed from the training script ---
from old_Code.train_classifier import ShotClassifierLSTM, get_keypoint

# --- Constants & Configuration ---
SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 17
KEYPOINT_DIM = 2

# --- Keypoint Constants ---
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_RIGHT_WRIST = 10
WRIST_VEL_THRESHOLD = 700
CLASSIFICATION_COOLDOWN = 45

# --- New Standalone Helper Function for Inference ---
def normalize_keypoints_for_inference(keypoints_tensor):
    """
    Normalizes a single set of keypoints for inference.
    `keypoints_tensor` is a tensor for one person, e.g., from results[0].keypoints.data[i]
    """
    l_shoulder = get_keypoint(keypoints_tensor, KP_LEFT_SHOULDER)
    r_shoulder = get_keypoint(keypoints_tensor, KP_RIGHT_SHOULDER)
    
    if l_shoulder and r_shoulder:
        anchor_point = np.array([(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2])
    elif l_shoulder:
        anchor_point = np.array(l_shoulder)
    elif r_shoulder:
        anchor_point = np.array(r_shoulder)
    else:
        anchor_point = np.array([0, 0])
        
    kpts_xy = keypoints_tensor.cpu().numpy()[:, :KEYPOINT_DIM]
    normalized_kpts = kpts_xy - anchor_point
    return normalized_kpts

# --- Argparse function ---
def parse_opt():
    parser = argparse.ArgumentParser()
    # Using the correct model name
    parser.add_argument('--pose-weights', type=str, default='models/yolov11n-pose.pt', help='Pose model path')
    parser.add_argument('--classifier-weights', type=str, default='models/shot_classifier.pth', help='Shot classifier model path')
    parser.add_argument('--class-map', type=str, default='models/class_map.json', help='Path to class-to-index JSON map')
    parser.add_argument('--source', type=str, default='video_to_analyze.mp4', help='Video path or 0 for webcam')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    opt, _ = parser.parse_known_args()
    return opt

# --- Main Analysis Function ---
def run(opt):
    device = torch.device(opt.device if torch.cuda.is_available() and opt.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    print("Loading YOLOv11 pose model...")
    pose_model = YOLO(opt.pose_weights)

    print("Loading shot classification model...")
    with open(opt.class_map, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    num_classes = len(idx_to_class)
    
    classifier_model = ShotClassifierLSTM(
        NUM_KEYPOINTS * KEYPOINT_DIM, 128, 2, num_classes
    ).to(device)
    classifier_model.load_state_dict(torch.load(opt.classifier_weights, map_location=device))
    classifier_model.eval()
    print("Models loaded successfully.")

    cap = cv2.VideoCapture(opt.source)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    time_diff = 1.0 / video_fps if video_fps > 0 else 1/30.0
    out_video_name = f"{os.path.splitext(os.path.basename(opt.source))[0]}_classified_output.mp4"
    out = cv2.VideoWriter(out_video_name, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (int(cap.get(3)), int(cap.get(4))))
    
    keypoint_buffers = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
    prev_kpts = {}
    last_classification = defaultdict(lambda: {'name': '', 'frame_countdown': 0})
    cooldown_timers = defaultdict(int)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        annotated_frame = frame.copy()
        
        for track_id in list(cooldown_timers.keys()):
            cooldown_timers[track_id] -= 1
            if cooldown_timers[track_id] <= 0:
                del cooldown_timers[track_id]
        
        pose_results = pose_model.track(frame, persist=True, verbose=False, device=device)
        if pose_results and pose_results[0].boxes:
            annotated_frame = pose_results[0].plot(img=annotated_frame)
            if pose_results[0].boxes.id is not None:
                track_ids = pose_results[0].boxes.id.int().cpu().tolist()
                all_keypoints = pose_results[0].keypoints.data
                
                new_prev_kpts = {}
                for i, track_id in enumerate(track_ids):
                    kpts = all_keypoints[i]
                    
                    # --- Correctly call the new local helper function ---
                    normalized_kpts = normalize_keypoints_for_inference(kpts).flatten()
                    keypoint_buffers[track_id].append(normalized_kpts)
                    
                    vel_r_wrist = 0.0
                    if track_id in prev_kpts:
                        kp_r_wrist_now = get_keypoint(kpts, KP_RIGHT_WRIST)
                        kp_r_wrist_prev = get_keypoint(prev_kpts[track_id], KP_RIGHT_WRIST)
                        if kp_r_wrist_now and kp_r_wrist_prev:
                            dist = np.linalg.norm(np.array(kp_r_wrist_now) - np.array(kp_r_wrist_prev))
                            vel_r_wrist = dist / time_diff

                    if vel_r_wrist > WRIST_VEL_THRESHOLD and track_id not in cooldown_timers:
                        if len(keypoint_buffers[track_id]) == SEQUENCE_LENGTH:
                            sequence_to_classify = torch.tensor(
                                np.array(keypoint_buffers[track_id]), dtype=torch.float32
                            ).unsqueeze(0).to(device)

                            with torch.no_grad():
                                output = classifier_model(sequence_to_classify)
                                _, predicted_idx = torch.max(output, 1)
                                predicted_class_name = idx_to_class[predicted_idx.item()]
                            
                            last_classification[track_id] = {
                                'name': predicted_class_name.replace('0','').replace('1','').replace('_',' ').strip(),
                                'frame_countdown': int(video_fps * 2)
                            }
                            cooldown_timers[track_id] = CLASSIFICATION_COOLDOWN

                    new_prev_kpts[track_id] = kpts.clone()
                prev_kpts = new_prev_kpts

        for track_id, classification in list(last_classification.items()):
            if classification['frame_countdown'] > 0:
                if pose_results and pose_results[0].boxes and track_id in pose_results[0].boxes.id.int().cpu().tolist():
                    box_idx = pose_results[0].boxes.id.int().cpu().tolist().index(track_id)
                    box_coords = pose_results[0].boxes.xyxy[box_idx].cpu().numpy()
                    x1, y1 = int(box_coords[0]), int(box_coords[1])
                    cv2.putText(annotated_frame, classification['name'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                
                last_classification[track_id]['frame_countdown'] -= 1

        out.write(annotated_frame)
        cv2.imshow("Badminton Shot Classification", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1
        print(f"Processed frame {frame_idx}", end='\r')

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nAnalysis complete. Output video saved to {out_video_name}")

if __name__ == "__main__":
    cli_opts = parse_opt()
    run(cli_opts)