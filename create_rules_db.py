# ==============================================================================
# create_rules_db.py
#
# Analyzes the "ideal" shot videos to create a JSON database of
# biomechanical rules for form correction.
# ==============================================================================
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# --- Configuration ---
IDEAL_VIDEOS_PATH = "ideal_shots/"
RULES_DB_SAVE_PATH = "models/form_rules.json"
POSE_MODEL_PATH = 'models/yolo11s-pose.pt' # Using a high-quality model for analysis
DEVICE = "cuda"

# --- Keypoint Constants ---
KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6
KP_LEFT_ELBOW, KP_RIGHT_ELBOW = 7, 8
KP_LEFT_WRIST, KP_RIGHT_WRIST = 9, 10
KP_LEFT_HIP, KP_RIGHT_HIP = 11, 12

# --- Helper Functions ---
def get_keypoint(kpts, index):
    if kpts is not None and index < len(kpts):
        return kpts[index]
    return (0, 0, 0) # Return with 0 confidence if not found

def calculate_angle(p1, p2, p3):
    # p1, p2, p3 are full keypoint tuples (x, y, conf)
    if p1[2] == 0 or p2[2] == 0 or p3[2] == 0: return None
    v1 = np.array(p1[:2]) - np.array(p2[:2])
    v2 = np.array(p3[:2]) - np.array(p2[:2])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def main():
    print(f"Loading pose model: {POSE_MODEL_PATH}")
    pose_model = YOLO(POSE_MODEL_PATH)
    rules_db = {}

    video_files = [f for f in os.listdir(IDEAL_VIDEOS_PATH) if f.endswith('.mp4')]
    
    for video_file in tqdm(video_files, desc="Analyzing Ideal Shots"):
        class_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(IDEAL_VIDEOS_PATH, video_file)
        
        cap = cv2.VideoCapture(video_path)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        all_kpts_in_video = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = pose_model(frame, verbose=False, device=DEVICE)
            if results[0].keypoints and len(results[0].keypoints.data) > 0:
                # Assume the most prominent player is the subject
                all_kpts_in_video.append(results[0].keypoints.data[0].cpu().numpy())
        cap.release()

        if not all_kpts_in_video:
            print(f"Warning: No keypoints found in {video_file}. Skipping.")
            continue

        # Find the frame with the highest wrist position (likely impact frame)
        highest_wrist_frame_idx = -1
        max_wrist_y = float('inf')
        
        for i, kpts in enumerate(all_kpts_in_video):
            r_wrist = get_keypoint(kpts, KP_RIGHT_WRIST)
            l_wrist = get_keypoint(kpts, KP_LEFT_WRIST)
            # Use the higher of the two wrists, y is smaller when higher
            wrist_y = min(r_wrist[1], l_wrist[1]) if r_wrist[2] > 0 or l_wrist[2] > 0 else float('inf')
            if wrist_y < max_wrist_y:
                max_wrist_y = wrist_y
                highest_wrist_frame_idx = i

        if highest_wrist_frame_idx == -1:
            print(f"Warning: No valid wrist keypoints in {video_file}. Skipping.")
            continue
            
        # Analyze the "impact frame"
        impact_kpts = all_kpts_in_video[highest_wrist_frame_idx]
        
        # Determine dominant arm (arm that is higher)
        r_shoulder_y = get_keypoint(impact_kpts, KP_RIGHT_SHOULDER)[1]
        l_shoulder_y = get_keypoint(impact_kpts, KP_LEFT_SHOULDER)[1]
        
        if r_shoulder_y < l_shoulder_y: # Right arm is dominant
            shoulder, elbow, wrist, hip = KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW, KP_RIGHT_WRIST, KP_RIGHT_HIP
        else: # Left arm is dominant
            shoulder, elbow, wrist, hip = KP_LEFT_SHOULDER, KP_LEFT_ELBOW, KP_LEFT_WRIST, KP_LEFT_HIP

        # Calculate metrics for the dominant arm
        p_shoulder = get_keypoint(impact_kpts, shoulder)
        p_elbow = get_keypoint(impact_kpts, elbow)
        p_wrist = get_keypoint(impact_kpts, wrist)
        p_hip = get_keypoint(impact_kpts, hip)

        elbow_angle = calculate_angle(p_shoulder, p_elbow, p_wrist)
        shoulder_angle = calculate_angle(p_hip, p_shoulder, p_elbow)
        
        # Normalized height of the wrist at impact
        wrist_contact_height = p_wrist[1] / frame_height if p_wrist[2] > 0 and frame_height > 0 else None

        # Store the rules
        # In create_rules_db.py -> main()

        # Store the rules
        rules_db[class_name] = {
            # Convert numpy.float32 to standard Python float
            'ideal_elbow_angle': float(round(elbow_angle, 1)) if elbow_angle is not None else None,
            'ideal_shoulder_angle': float(round(shoulder_angle, 1)) if shoulder_angle is not None else None,
            'ideal_wrist_contact_height': float(round(wrist_contact_height, 2)) if wrist_contact_height is not None else None,
            
            # Add simple feedback text
            'feedback_low_elbow': "Try to extend your arm more at contact.",
            'feedback_high_contact': "Good height! You're hitting on top of the shuttle."
        }
                
    # Save the database
    os.makedirs(os.path.dirname(RULES_DB_SAVE_PATH), exist_ok=True)
    with open(RULES_DB_SAVE_PATH, 'w') as f:
        json.dump(rules_db, f, indent=4)
        
    print(f"Successfully created form rules database at {RULES_DB_SAVE_PATH}")
    print(json.dumps(rules_db, indent=4))

if __name__ == "__main__":
    main()