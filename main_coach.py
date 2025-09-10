# ==============================================================================
# main_coach.py (Refactored for Dual-Model System)
#
# USAGE:
# python main_coach.py --video "path/to/your/test_video.mp4"
#
# This updated script dynamically loads the correct model and calculates
# all necessary features (including velocity) on-the-fly.
# ==============================================================================
import os
import json
import torch
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import deque
import argparse
from sklearn.model_selection import train_test_split
import subprocess
import sys
from tqdm import tqdm

# --- DYNAMICALLY IMPORT THE CLASSIFIER CLASS ---
# We can import from either training script as the class definition is the same.
try:
    from train_transformer_pose_only import TransformerClassifier
except ImportError:
    from train_transformer_classifier import TransformerClassifier

# --- Configuration ---
# --- UPDATED: Model paths for the two new models ---
POSE_ONLY_MODEL_PATH = "models/transformer_pose_only_v4.pth"
MULTIMODAL_MODEL_PATH = "models/transformer_multimodal_classifier_v4.pth" # Using the latest v3 model

CLASS_MAP_PATH = "models/class_map.json"
RULES_PATH = "models/form_rules.json"
POSE_MODEL_PATH = 'models/yolo11s-pose.pt'
TRACKNET_PREDICTION_DIR = "TrackNetV3/prediction"

# --- Model Hyperparameters (Must match training scripts) ---
SEQUENCE_LENGTH = 32
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.4
NUM_KEYPOINTS = 17

# --- Tunable Thresholds ---
CONFIDENCE_THRESHOLD = 0.5
PREDICTION_BUFFER_FRAMES = 5

class UnifiedCoach:
    # --- MAJOR REFACTOR: __init__ now takes the mode to load the correct assets ---
    def __init__(self, mode):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing coach in '{mode}' mode...")
        print(f"Using device: {self.device}")
        
        self.mode = mode

        # --- DYNAMIC MODEL LOADING ---
        if self.mode == 'real_time':
            self.model_path = POSE_ONLY_MODEL_PATH
            self.input_size = ((NUM_KEYPOINTS * 2) * 2) + 6 + 8  # 68 features (Pose XY + Pose Velocity)
            self.preprocessed_path = "preprocessed_data/"
        elif self.mode == 'high_accuracy':
            self.model_path = MULTIMODAL_MODEL_PATH
            self.input_size = ((NUM_KEYPOINTS * 2) * 2) + 6 + 8 + 7# 73 features (Pose XY+Vel + Shuttle XY+Vel+Angle)
            self.preprocessed_path = "preprocessed_multimodal_data/"
        else:
            raise ValueError(f"Invalid mode specified: {self.mode}")

        # --- Load All Models, Mappings, and Rules ---
        with open(CLASS_MAP_PATH, 'r', encoding='utf-8') as f: self.class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        with open(RULES_PATH, 'r', encoding='utf-8') as f: self.rules = json.load(f)
        
        self.pose_model = YOLO(POSE_MODEL_PATH)
        
        # Load the selected classifier model
        self.classifier_model = TransformerClassifier(self.input_size, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, self.num_classes, DROPOUT)
        self.classifier_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.classifier_model.to(self.device)
        self.classifier_model.eval()
        print(f"Successfully loaded classifier model from: {self.model_path}")

        # Load the correct normalization stats for the selected model
        self.mean, self.std = self._get_norm_stats()

        self.keypoint_map = { 'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12, 'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16 }
        self.angle_definitions = { "R Elbow": ("right_shoulder", "right_elbow", "right_wrist"), "L Elbow": ("left_shoulder", "left_elbow", "left_wrist"), "R Shoulder": ("right_elbow", "right_shoulder", "right_hip"), "L Shoulder": ("left_elbow", "left_shoulder", "left_hip"), "R Knee": ("right_hip", "right_knee", "right_ankle"), "L Knee": ("left_hip", "left_knee", "left_ankle") }

    def _get_norm_stats(self):
        print(f"Loading normalization stats from: {self.preprocessed_path}")
        file_paths, labels = [], []
        class_names = list(self.idx_to_class.values())
        for class_name in class_names:
            class_dir = os.path.join(self.preprocessed_path, class_name)
            if not os.path.isdir(class_dir): continue
            for data_file in os.listdir(class_dir):
                if data_file.endswith('.npy'):
                    file_paths.append(os.path.join(class_dir, data_file))
                    labels.append(class_name)
        
        if not file_paths:
            print("Warning: No training files found. Using zero mean and unit variance.")
            return torch.zeros(self.input_size).to(self.device), torch.ones(self.input_size).to(self.device)
            
        train_files, _, _, _ = train_test_split(file_paths, labels, test_size=0.2, random_state=42, stratify=labels)
        all_train_data = np.vstack([np.load(f) for f in train_files])
        mean, std = np.mean(all_train_data, axis=0), np.std(all_train_data, axis=0)
        std[std == 0] = 1.0
        print("Normalization stats loaded.")
        return torch.tensor(mean, dtype=torch.float32).to(self.device), torch.tensor(std, dtype=torch.float32).to(self.device)

    def _run_tracknet_on_video(self, video_path):
        print("\nTrackNet data not found. Processing video... (This may take several minutes)")
        tracknet_dir = os.path.join(os.path.dirname(__file__), 'TrackNetV3')
        predict_script = os.path.join(tracknet_dir, 'predict.py')
        tracknet_model = os.path.join(tracknet_dir, 'ckpts', 'TrackNet_best.pt')

        if not os.path.exists(predict_script) or not os.path.exists(tracknet_model):
            print(f"ERROR: TrackNet scripts or model not found in {tracknet_dir}")
            return False

        command = [sys.executable, predict_script, '--video_file', video_path, '--tracknet_file', tracknet_model, '--save_dir', TRACKNET_PREDICTION_DIR]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            print("TrackNet processing complete.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: TrackNet processing failed for {video_path}:\n{e.stderr}")
            return False

    def _calculate_angle(self, p1, p2, p3):
        a, b, c = np.array(p1), np.array(p2), np.array(p3)
        if np.all(b==0) or np.all(a==0) or np.all(c==0): return None
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _analyze_frame_angles(self, kpts):
        angles = {}
        for name, (p1_n, p2_n, p3_n) in self.angle_definitions.items():
            try:
                p1, p2, p3 = kpts[self.keypoint_map[p1_n]], kpts[self.keypoint_map[p2_n]], kpts[self.keypoint_map[p3_n]]
                angle = self._calculate_angle(p1, p2, p3)
                if angle is not None: angles[name] = angle
            except (KeyError, IndexError): continue
        return angles

    def _apply_coaching_rules(self, shot_name, angles_at_shot):
        if shot_name not in self.rules: return f"No rules for {shot_name}."
        feedback = [f"{shot_name}:"]
        for angle_name, rule in self.rules[shot_name].items():
            if angle_name in angles_at_shot:
                angle_val = angles_at_shot[angle_name]
                if "ideal_min" in rule and angle_val < rule["ideal_min"]: feedback.append(rule.get("feedback_low", ""))
                elif "ideal_max" in rule and angle_val > rule["ideal_max"]: feedback.append(rule.get("feedback_high", ""))
        
        clean_feedback = [msg for msg in feedback if msg and msg != "N/A"]
        return " ".join(clean_feedback) if len(clean_feedback) > 1 else f"{shot_name}: Good form!"

    def _draw_text_box(self, frame, text_lines):
        x, y, padding = 10, 10, 5
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        line_height = cv2.getTextSize("S", font, font_scale, thickness)[0][1] + padding
        
        box_width = 0
        if text_lines:
            try:
                box_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in text_lines)
            except (ValueError, TypeError): # Handle empty or invalid lines
                pass

        box_height = len(text_lines) * line_height
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + box_width + 2*padding, y + box_height + padding), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        for i, line in enumerate(text_lines):
            cv2.putText(frame, line, (x + padding, y + (i*line_height) + (line_height - padding)), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def run_analysis(self, video_path):
        shuttle_df = None
        if self.mode == 'high_accuracy':
            output_suffix = '_high_accuracy_analysis.mp4'
            csv_path = os.path.join(TRACKNET_PREDICTION_DIR, os.path.splitext(os.path.basename(video_path))[0] + '_ball.csv')
            if not os.path.exists(csv_path):
                if not self._run_tracknet_on_video(video_path): return
            shuttle_df = pd.read_csv(csv_path).set_index('Frame')
        else:
            output_suffix = '_realtime_analysis.mp4'
        
        cap = cv2.VideoCapture(video_path)
        width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FPS))
        output_filename = os.path.splitext(video_path)[0] + output_suffix
        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        # --- ON-THE-FLY FEATURE CALCULATION VARIABLES ---
        feature_sequence = deque(maxlen=SEQUENCE_LENGTH)
        raw_keypoints_buffer = deque(maxlen=SEQUENCE_LENGTH)
        previous_kpts_norm = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        previous_shuttle_coords = np.array([-1.0, -1.0], dtype=np.float32)

        current_prediction, feedback, conf = "Idle", "", 0.0
        prediction_buffer = deque(maxlen=PREDICTION_BUFFER_FRAMES)
        shot_locked = False
        
        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"Analyzing ({self.mode.replace('_', ' ')} mode)") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                pbar.update(1)
                frame_copy = frame.copy()
                frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                results = self.pose_model(frame, verbose=False)
                raw_kpts = np.zeros((NUM_KEYPOINTS, 2))
                current_kpts_norm = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
                
                if results and results[0].boxes and len(results[0].boxes.xyxy) > 0:
                    largest_person_idx = results[0].boxes.xywh.prod(dim=1).argmax()
                    box = results[0].boxes.xyxy[largest_person_idx].cpu().numpy().astype(int)
                    cv2.rectangle(frame_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    raw_kpts = results[0].keypoints[largest_person_idx].data[0][:, :2].cpu().numpy()
                    current_kpts_norm = raw_kpts.copy()
                    current_kpts_norm[:, 0] /= width
                    current_kpts_norm[:, 1] /= height
                
                # --- ON-THE-FLY FEATURE ENGINEERING ---
                pose_velocity = current_kpts_norm - previous_kpts_norm
                
                if self.mode == 'real_time':
                    features_for_frame = np.concatenate([current_kpts_norm.flatten(), pose_velocity.flatten()])
                else: # high_accuracy mode
                    current_shuttle_coords = np.array([-1.0, -1.0], dtype=np.float32)
                    if shuttle_df is not None:
                        try:
                            if frame_num in shuttle_df.index and shuttle_df.loc[frame_num]['Visibility'] == 1:
                                current_shuttle_coords = shuttle_df.loc[frame_num][['X', 'Y']].to_numpy(dtype=np.float32)
                        except (KeyError, IndexError): pass
                    
                    shuttle_velocity = current_shuttle_coords - previous_shuttle_coords
                    shuttle_angle = np.arctan2(shuttle_velocity[1], shuttle_velocity[0]) * (180. / np.pi) if np.any(shuttle_velocity) else 0.0
                    
                    shuttle_features = np.concatenate([current_shuttle_coords, shuttle_velocity, [shuttle_angle]])
                    features_for_frame = np.concatenate([current_kpts_norm.flatten(), pose_velocity.flatten(), shuttle_features])
                    previous_shuttle_coords = current_shuttle_coords

                # --- NORMALIZATION & SEQUENCE APPENDING ---
                normalized_features = (torch.tensor(features_for_frame, dtype=torch.float32).to(self.device) - self.mean) / self.std
                feature_sequence.append(normalized_features.cpu().numpy())
                raw_keypoints_buffer.append(raw_kpts)
                previous_kpts_norm = current_kpts_norm

                # --- PREDICTION LOGIC ---
                if len(feature_sequence) == SEQUENCE_LENGTH:
                    seq_tensor = torch.tensor(np.array(feature_sequence), dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        outputs = self.classifier_model(seq_tensor.unsqueeze(0))
                        probs = torch.softmax(outputs, dim=1)
                        c, idx = torch.max(probs, 1)
                        pred_class = self.idx_to_class.get(idx.item()) if c.item() > CONFIDENCE_THRESHOLD else "Idle"
                        prediction_buffer.append(pred_class)
                        conf = c.item()

                    if len(prediction_buffer) == PREDICTION_BUFFER_FRAMES:
                        first_pred = prediction_buffer[0]
                        if first_pred != "Idle" and all(p == first_pred for p in prediction_buffer):
                            if not shot_locked:
                                current_prediction, shot_locked = first_pred, True
                                angles = self._analyze_frame_angles(raw_keypoints_buffer[SEQUENCE_LENGTH // 2])
                                feedback = self._apply_coaching_rules(current_prediction, angles)
                        elif "Idle" in prediction_buffer and shot_locked:
                            current_prediction, shot_locked, feedback, conf = "Idle", False, "", 0.0

                display_conf = conf if shot_locked else 0.0
                info_lines = [f"Mode: {self.mode.replace('_', ' ').title()}", f"Shot: {current_prediction} ({display_conf:.2f})"]
                if feedback: info_lines.append(feedback)
                frame_copy = self._draw_text_box(frame_copy, info_lines)
                
                out.write(frame_copy)
                cv2.imshow("AI Coach", frame_copy)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\nAnalysis complete. Video saved to: {output_filename}")

def main_menu():
    parser = argparse.ArgumentParser(description="AI Badminton Coach")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file to analyze.")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
        return

    while True:
        print("\n" + "="*25)
        print("  AI Badminton Coach Menu")
        print("="*25)
        print("1. Real-Time Analysis (Fast, Pose-Only Model)")
        print("2. High-Accuracy Analysis (Slow, Multi-Modal Model)")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            print("\nStarting Real-Time Analysis...")
            coach = UnifiedCoach(mode='real_time')
            coach.run_analysis(args.video)
            break
        elif choice == '2':
            print("\nStarting High-Accuracy Analysis...")
            coach = UnifiedCoach(mode='high_accuracy')
            coach.run_analysis(args.video)
            break
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main_menu()
