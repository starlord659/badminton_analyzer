# ==============================================================================
# coach_analyzer.py (Version 6 - Final with select_roi function)
#
# This version includes the missing select_roi helper function and is the
# complete, working version.
# ==============================================================================
import os
import time
import json
import argparse
from collections import defaultdict, deque
import torch
import cv2
import numpy as np
from ultralytics import YOLO

# --- Import from the training script ---
from train_posec3d import LitePoseC3D, create_pose_video_from_kpts, \
                          SEQUENCE_LENGTH, NUM_KEYPOINTS, HEATMAP_SIZE

# --- Trigger & Feedback Constants ---
# --- YOUR TUNED CONSTANTS ---
MOTION_ENERGY_THRESHOLD = 27
MOTION_PEAK_WINDOW = 5
CLASSIFICATION_COOLDOWN = 36

# --- Keypoint Constants (from COCO dataset) ---
KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6
KP_LEFT_ELBOW, KP_RIGHT_ELBOW = 7, 8
KP_LEFT_WRIST, KP_RIGHT_WRIST = 9, 10
KP_LEFT_HIP, KP_RIGHT_HIP = 11, 12
KP_LEFT_KNEE, KP_RIGHT_KNEE = 13, 14
KP_LEFT_ANKLE, KP_RIGHT_ANKLE = 15, 16

# --- NEW: Define the keypoints to use for motion calculation ---
ACTION_KEYPOINT_INDICES = [
    KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
    KP_LEFT_ELBOW, KP_RIGHT_ELBOW,
    KP_LEFT_WRIST, KP_RIGHT_WRIST,
    KP_LEFT_HIP, KP_RIGHT_HIP,
    KP_LEFT_KNEE, KP_RIGHT_KNEE,
    KP_LEFT_ANKLE, KP_RIGHT_ANKLE
]
# --- HELPER FUNCTIONS (LOCAL TO THIS SCRIPT) ---

def select_roi(video_source):
    """
    Allows the user to select a Region of Interest (ROI) from the first frame of the video.
    Returns the ROI coordinates (x, y, w, h) or None if not selected.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video to select ROI.")
        return None
    
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame to select ROI.")
        cap.release()
        return None
    
    cap.release()
    
    print("\n" + "="*50)
    print("ROI SELECTION INSTRUCTIONS:")
    print(" - Click and drag a rectangle around the court area.")
    print(" - Press ENTER or SPACE to confirm.")
    print(" - Press 'c' to cancel and analyze the full frame.")
    print("="*50 + "\n")

    roi = cv2.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    
    if roi[2] > 0 and roi[3] > 0:
        print(f"ROI selected at: {roi}")
        return roi
    else:
        print("No ROI selected. Analyzing full frame.")
        return None

def get_keypoint(kpts_array, index):
    """Safely retrieves a keypoint (x, y, conf) from a numpy array."""
    if kpts_array is not None and index < len(kpts_array):
        return kpts_array[index]
    return (0, 0, 0)

def calculate_angle(p1, p2, p3):
    if p1[2] < 0.1 or p2[2] < 0.1 or p3[2] < 0.1: return None
    v1 = np.array(p1[:2]) - np.array(p2[:2])
    v2 = np.array(p3[:2]) - np.array(p2[:2])
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return None
    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

# def get_form_feedback(shot_name, kpts_at_impact, frame_height, rules_db):
#     if shot_name not in rules_db or rules_db[shot_name].get('ideal_elbow_angle') is None:
#         return []
#     rules, feedback = rules_db[shot_name], []
#     r_shoulder_y, l_shoulder_y = get_keypoint(kpts_at_impact, 6)[1], get_keypoint(kpts_at_impact, 5)[1]
#     shoulder, elbow, wrist = (6, 8, 10) if r_shoulder_y > 0 and r_shoulder_y < l_shoulder_y else (5, 7, 9)
#     user_elbow_angle = calculate_angle(get_keypoint(kpts_at_impact, shoulder), get_keypoint(kpts_at_impact, elbow), get_keypoint(kpts_at_impact, wrist))
#     if user_elbow_angle and rules.get('ideal_elbow_angle'):
#         if user_elbow_angle < rules['ideal_elbow_angle'] - 15:
#             feedback.append(f"Elbow: {int(user_elbow_angle)}d (Too bent)")
#             if 'feedback_low_elbow' in rules: feedback.append(rules['feedback_low_elbow'])
#     return feedback

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose-weights', type=str, default='models/yolov8s-pose.pt')
    parser.add_argument('--classifier-weights', type=str, default='models/posec3d_classifier.pth')
    parser.add_argument('--class-map', type=str, default='models/class_map.json')
    parser.add_argument('--rules-db', type=str, default='models/form_rules.json')
    parser.add_argument('--source', type=str, default='test_video.mp4')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

# --- Main Application Logic ---
def run_coach(opt):
    # This call will now work correctly
    roi = select_roi(opt.source)
    
    device = torch.device(opt.device)
    pose_model = YOLO(opt.pose_weights)
    with open(opt.class_map, 'r') as f: class_to_idx = json.load(f)
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    with open(opt.rules_db, 'r') as f: rules_db = json.load(f)
    classifier_model = LitePoseC3D(len(idx_to_class)).to(device)
    classifier_model.load_state_dict(torch.load(opt.classifier_weights, map_location=device))
    classifier_model.eval()
    print("All models and databases loaded successfully.")

    cap = cv2.VideoCapture(opt.source)
    video_fps, frame_width, frame_height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    out_video_name = f"{os.path.splitext(os.path.basename(opt.source))[0]}_coaching_output.mp4"
    out = cv2.VideoWriter(out_video_name, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (frame_width, frame_height))
    
    keypoint_buffers = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
    prev_kpts, cooldown_timers = {}, defaultdict(int)
    feedback_display = defaultdict(lambda: {'text': [], 'countdown': 0})
    motion_energy_buffers = defaultdict(lambda: deque(maxlen=MOTION_PEAK_WINDOW))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        annotated_frame = frame.copy()

        for track_id in list(cooldown_timers.keys()):
            cooldown_timers[track_id] -= 1
            if cooldown_timers[track_id] <= 0: del cooldown_timers[track_id]
        for track_id in list(feedback_display.keys()):
            feedback_display[track_id]['countdown'] -= 1
            if feedback_display[track_id]['countdown'] <= 0: del feedback_display[track_id]
            
        if roi: cv2.rectangle(annotated_frame, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 1)
        pose_results = pose_model.track(frame, persist=True, verbose=False, device=device)
        if pose_results and pose_results[0].boxes and pose_results[0].boxes.id is not None:
            annotated_frame = pose_results[0].plot(img=annotated_frame, line_width=1)
            track_ids, all_keypoints = pose_results[0].boxes.id.int().cpu().tolist(), pose_results[0].keypoints.data
            
            new_prev_kpts = {}
            for i, track_id in enumerate(track_ids):
                box_xywh = pose_results[0].boxes.xywh[i].cpu().numpy()
                center_x, center_y = int(box_xywh[0]), int(box_xywh[1])
                if roi and not (roi[0] < center_x < roi[0] + roi[2] and roi[1] < center_y < roi[1] + roi[3]):
                    continue
                
                kpts = all_keypoints[i]
                kpts_xy_normalized = kpts[:, :2].cpu().numpy()
                kpts_xy_normalized[:, 0] /= frame_width
                kpts_xy_normalized[:, 1] /= frame_height
                keypoint_buffers[track_id].append(kpts_xy_normalized)
                
                motion_energy, shot_detected = 0, False
                if track_id in prev_kpts:
                    # --- MODIFIED: Select only the action keypoints ---
                    current_action_kpts = kpts[ACTION_KEYPOINT_INDICES]
                    prev_action_kpts = prev_kpts[track_id][ACTION_KEYPOINT_INDICES]

                    # Check for confidence in this subset of keypoints
                    valid_indices = ((current_action_kpts[:, 2] > 0.1) & (prev_action_kpts[:, 2] > 0.1)).cpu().numpy()

                    if np.any(valid_indices):
                        # Get the coordinates for the valid action keypoints
                        current_kpts_np = current_action_kpts[:, :2].cpu().numpy()
                        prev_kpts_np = prev_action_kpts[:, :2].cpu().numpy()

                        # The rest of the calculation is the same, but now only operates on the subset
                        motion_energy = np.sum(np.linalg.norm(current_kpts_np[valid_indices] - prev_kpts_np[valid_indices], axis=1))
                motion_energy_buffers[track_id].append(motion_energy)
                cv2.putText(annotated_frame, f"Motion: {int(motion_energy)}", (center_x, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                if len(motion_energy_buffers[track_id]) == MOTION_PEAK_WINDOW:
                    energies = list(motion_energy_buffers[track_id])
                    peak_idx = MOTION_PEAK_WINDOW // 2
                    if energies[peak_idx] == max(energies):
                        if energies[peak_idx] > 1: print(f"Peak detected for player {track_id} with energy: {energies[peak_idx]:.2f}")
                        if energies[peak_idx] > MOTION_ENERGY_THRESHOLD: shot_detected = True
                
                if shot_detected and track_id not in cooldown_timers:
                    if len(keypoint_buffers[track_id]) == SEQUENCE_LENGTH:
                        pose_video = create_pose_video_from_kpts(keypoint_buffers[track_id], SEQUENCE_LENGTH, NUM_KEYPOINTS, HEATMAP_SIZE)
                        input_tensor = torch.tensor(pose_video).unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = classifier_model(input_tensor)
                            shot_name = idx_to_class[torch.argmax(output).item()]
                        
                        # impact_frame_index = - (MOTION_PEAK_WINDOW // 2) - 1
                        # kpts_at_impact_unnorm = np.hstack([keypoint_buffers[track_id][impact_frame_index] * [frame_width, frame_height], kpts[:, 2:].cpu().numpy()])
                        # feedback_text = get_form_feedback(shot_name, kpts_at_impact_unnorm, frame_height, rules_db)
                        
                        display_text = [f"Shot: {shot_name.replace('_', ' ')}"] # + feedback_text
                        feedback_display[track_id] = {'text': display_text, 'countdown': int(video_fps * 4)}
                        cooldown_timers[track_id] = CLASSIFICATION_COOLDOWN

                new_prev_kpts[track_id] = kpts.clone()
            prev_kpts = new_prev_kpts

        # In coach_analyzer.py -> run_coach()

        for track_id, fb in feedback_display.items():
            # --- ADDED `pose_results[0].boxes.id is not None` ---
            if pose_results and pose_results[0].boxes and pose_results[0].boxes.id is not None and track_id in pose_results[0].boxes.id.int().cpu().tolist():
                box_idx = pose_results[0].boxes.id.int().cpu().tolist().index(track_id)
                x1, y1 = int(pose_results[0].boxes.xyxy[box_idx][0]), int(pose_results[0].boxes.xyxy[box_idx][1])
                for j, line in enumerate(fb['text']):
                    cv2.putText(annotated_frame, line, (x1, y1 - 10 - (j*25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(annotated_frame)
        cv2.imshow("AI Badminton Coach", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(), out.release(), cv2.destroyAllWindows()
    print(f"\nAnalysis complete. Output video saved to {out_video_name}")

if __name__ == "__main__":
    run_coach(parse_opt())