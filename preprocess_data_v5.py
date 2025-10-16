import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import torch
from court_detect import CourtDetect
import warnings
from collections import deque
import json

# Suppress all OpenCV/GStreamer warnings
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)
cv2.setLogLevel(0)

# --- Configuration ---
POSE_MODEL_PATH = 'models/yolo11s-pose.pt'
NUM_KEYPOINTS = 17
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURES_PER_PLAYER = 36
COURT_CORNER_INDICES = [0, 4, 21, 17]
METADATA_DIR = "preprocessed_metadata/"

# --- Globals for multiprocessing model ---
model = None

def init_worker(model_path):
    """Initializer for each worker process to load the model once."""
    global model
    model = YOLO(model_path)
    print(f"Worker {os.getpid()} initialized YOLO model.")
# -----------------------------------------------

def get_homography_from_corners(corners):
    src_points = np.array(corners, dtype=np.float32)
    dst_points = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    H, _ = cv2.findHomography(src_points, dst_points)
    return H

def pixel_to_court(p, H):
    if H is None or np.all(p == 0) or np.isnan(p).any():
        return np.array([np.nan, np.nan], dtype=np.float32)
    p_hom = np.array([p[0], p[1], 1], dtype=np.float32)
    p_court_hom = H @ p_hom
    if abs(p_court_hom[2]) < 1e-6:
        return np.array([np.nan, np.nan], dtype=np.float32)
    coords = (p_court_hom / p_court_hom[2])[:2]
    return np.clip(coords, 0.0, 1.0)

def get_stable_position(kpts):
    priority_indices = [[11, 12], [5, 6], [13, 14], list(range(17))]
    for indices in priority_indices:
        points = kpts[indices]
        valid_points = points[~np.isnan(points).any(axis=1)]
        if len(valid_points) > 0:
            return np.mean(valid_points, axis=0)
    return np.array([np.nan, np.nan])

def detect_court_robust(video_path, court_detector):
    """Enhanced court detection with validation"""
    cap = cv2.VideoCapture(video_path)
    test_frame_indices = [0, 30, 60, 90, 150]

    for frame_idx in test_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        if court_detector.detect_court(frame):
            all_kps = court_detector.keypoints
            if all_kps is not None and len(all_kps) > max(COURT_CORNER_INDICES):
                corners = np.array([all_kps[i][:2] for i in COURT_CORNER_INDICES], dtype=np.int32)
                area = cv2.contourArea(corners)
                frame_area = frame.shape[0] * frame.shape[1]

                if 0.1 < (area / frame_area) < 0.9:
                    cap.release()
                    print(f"✓ Court detected at frame {frame_idx} for {os.path.basename(video_path)}")
                    return corners

    cap.release()
    return None

def is_on_court_floor(kpts, court_polygon):
    """Checks if any part of the player is inside the court polygon."""
    if court_polygon is None:
        return False
    valid_kpts = kpts[~np.isnan(kpts).any(axis=1)]
    for pt in valid_kpts:
        if cv2.pointPolygonTest(court_polygon, (float(pt[0]), float(pt[1])), False) >= 0:
            return True
    return False

def is_player_not_referee(kpts, court_polygon):
    """Checks if player's feet are below the net line to filter out the referee."""
    if court_polygon is None:
        return False
    net_y = np.min(court_polygon[:, 1])

    left_ankle_y = kpts[15, 1]
    right_ankle_y = kpts[16, 1]

    if not np.isnan(left_ankle_y) and left_ankle_y > net_y:
        return True
    if not np.isnan(right_ankle_y) and right_ankle_y > net_y:
        return True

    return False

def calculate_iou(box1, box2):
    """Calculate intersection over union for two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def match_players_to_history(current_players, player_history):
    """Match current players to previous frame using IoU"""
    if len(player_history) == 0 or len(current_players) != 2:
        return current_players

    prev_frame = player_history[-1]
    if 'top_box' not in prev_frame or 'bottom_box' not in prev_frame:
        return current_players
    
    iou_00 = calculate_iou(current_players[0]['box'], prev_frame['top_box'])
    iou_01 = calculate_iou(current_players[0]['box'], prev_frame['bottom_box'])
    iou_10 = calculate_iou(current_players[1]['box'], prev_frame['top_box'])
    iou_11 = calculate_iou(current_players[1]['box'], prev_frame['bottom_box'])
    
    score1 = iou_00 + iou_11
    score2 = iou_01 + iou_10
    
    if score2 > score1:
        return [current_players[1], current_players[0]]
    
    return current_players

def extract_features_from_video(video_path):
    """MODIFIED: Uses worker-initialized model and better initial player assignment"""
    court_detector = CourtDetect()
    court_polygon = detect_court_robust(video_path, court_detector)
    
    if court_polygon is None:
        print(f"Warning: No court found in {video_path}. Skipping.")
        return None, None
    
    homography_matrix = get_homography_from_corners(court_polygon)
    
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    metadata = {
        'frame_width': frame_width,
        'frame_height': frame_height,
        'num_frames': num_frames,
        'fps': fps,
        'court_polygon': court_polygon.tolist()
    }
    
    results_generator = model.track(
        source=video_path, tracker="botsort.yaml", conf=0.15, persist=True, stream=True, verbose=False
    )

    video_features = np.zeros((num_frames, FEATURES_PER_PLAYER * 2))
    player_history = deque(maxlen=5)
    
    print(f"Processing {os.path.basename(video_path)} ({frame_width}x{frame_height})...")
    
    for frame_idx, results in enumerate(tqdm(results_generator, total=num_frames)):
        valid_player_candidates = []
        
        if results.boxes.id is not None:
            tracker_ids = results.boxes.id.int().cpu().tolist()
            all_keypoints = results.keypoints.xy.cpu().numpy()
            boxes = results.boxes.xywh.cpu().numpy()

            for i, track_id in enumerate(tracker_ids):
                kpts_raw_pixels = all_keypoints[i]
                
                if is_player_not_referee(kpts_raw_pixels, court_polygon) and \
                   is_on_court_floor(kpts_raw_pixels, court_polygon):
                    
                    bbox_area = boxes[i][2] * boxes[i][3]
                    valid_player_candidates.append({
                        "keypoints": kpts_raw_pixels, "area": bbox_area, "box": boxes[i]
                    })
        
        top_player_features = np.zeros(FEATURES_PER_PLAYER)
        bottom_player_features = np.zeros(FEATURES_PER_PLAYER)
        
        if len(valid_player_candidates) > 0:
            sorted_candidates = sorted(valid_player_candidates, key=lambda x: x['area'], reverse=True)
            best_players = sorted_candidates[:2]
            
            if len(best_players) == 2:
                if not player_history:
                    y_pos_0 = get_stable_position(best_players[0]['keypoints'])[1]
                    y_pos_1 = get_stable_position(best_players[1]['keypoints'])[1]
                    if y_pos_0 > y_pos_1:
                        best_players = [best_players[1], best_players[0]]
                else:
                    best_players = match_players_to_history(best_players, player_history)
            
            for i, player in enumerate(best_players):
                kpts_raw_pixels = player['keypoints']
                
                kpts_norm_screen = kpts_raw_pixels.copy()
                kpts_norm_screen[:, 0] /= frame_width
                kpts_norm_screen[:, 1] /= frame_height

                player_centroid_pixel = get_stable_position(kpts_raw_pixels)
                player_court_coords = pixel_to_court(player_centroid_pixel, homography_matrix)
                
                player_features = np.concatenate([kpts_norm_screen.flatten(), player_court_coords])

                if len(best_players) == 1:
                    bottom_player_features = player_features
                else:
                    if i == 0:
                        top_player_features = player_features
                    else:
                        bottom_player_features = player_features
            
            if len(best_players) == 2:
                player_history.append({
                    'top_box': best_players[0]['box'], 'bottom_box': best_players[1]['box']
                })

        video_features[frame_idx] = np.concatenate([top_player_features, bottom_player_features])

    return video_features, metadata

def is_video_readable(video_path):
    cap = cv2.VideoCapture(video_path)
    ret = cap.isOpened()
    cap.release()
    return ret

def process_video_wrapper(args):
    video_path, class_name, processed_data_path = args
    
    if not is_video_readable(video_path):
        return ("skipped", video_path, "Unsupported video codec")
    
    try:
        full_features, metadata = extract_features_from_video(video_path)
        
        if full_features is not None and len(full_features) > 0:
            save_dir = os.path.join(processed_data_path, class_name)
            os.makedirs(save_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            save_path = os.path.join(save_dir, f"{base_name}.npy")
            metadata_path = os.path.join(METADATA_DIR, class_name, f"{base_name}.json")
            
            np.save(save_path, full_features)
            
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return ("success", video_path)
        else:
            return ("skipped", video_path, "No valid features extracted")
    except Exception as e:
        return ("failure", video_path, str(e))

def main(args):
    """MODIFIED: Handles both single-video and batch processing modes."""
    print("=" * 70)
    print("Auto-Preprocessing v5.2 with Single Video Mode")
    print("=" * 70)

    os.makedirs(METADATA_DIR, exist_ok=True)
    init_worker(POSE_MODEL_PATH) # Initialize model once for either mode

    # --- SINGLE VIDEO MODE ---
    # This is the new logic that will be triggered by main_coach.py
    if args.video and args.output_dir:
        video_path = args.video
        output_dir = args.output_dir
        
        print(f"Running in single-video mode for: {video_path}")
        os.makedirs(output_dir, exist_ok=True)
        
        if not is_video_readable(video_path):
            print(f"Error: Unsupported video codec or file not found at {video_path}")
            return

        try:
            full_features, metadata = extract_features_from_video(video_path)
            
            if full_features is not None and len(full_features) > 0:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                save_path = os.path.join(output_dir, f"{base_name}.npy")
                metadata_path = os.path.join(METADATA_DIR, f"{base_name}.json")
                
                np.save(save_path, full_features)
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"✓ Success! Processed data saved to {save_path}")
                print(f"✓ Metadata saved to {metadata_path}")
            else:
                print("Warning: No valid features were extracted from the video.")
        except Exception as e:
            print(f"✗ Failure during single video processing: {e}")
        
        return # Exit after processing the single video

    # --- BATCH PROCESSING MODE (Original Logic) ---
    print("Running in batch mode...")
    raw_data_path = args.raw_data_path
    processed_data_path = args.processed_data_path
    num_workers = args.workers

    tasks = []
    total_videos_found = 0
    already_processed_count = 0
    
    print("Scanning for videos and checking for existing processed files...")
    for class_name in sorted(os.listdir(raw_data_path)):
        class_dir = os.path.join(raw_data_path, class_name)
        if not os.path.isdir(class_dir): 
            continue
        
        for video_file in os.listdir(class_dir):
            if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                total_videos_found += 1
                base_name = os.path.splitext(video_file)[0]
                
                save_path = os.path.join(processed_data_path, class_name, f"{base_name}.npy")
                metadata_path = os.path.join(METADATA_DIR, class_name, f"{base_name}.json")
                
                if os.path.exists(save_path) and os.path.exists(metadata_path):
                    already_processed_count += 1
                    continue
                
                tasks.append((os.path.join(class_dir, video_file), class_name, processed_data_path))
    
    print(f"Scan complete. Found {total_videos_found} total videos.")
    print(f"  - {already_processed_count} videos are already processed (skipping).")
    print(f"  - {len(tasks)} videos remaining to be processed.")
    print(f"Using {num_workers} worker(s)\n")
    
    if not tasks:
        print("No new videos to process. Exiting.")
        return

    results = []
    if num_workers > 1:
        # Note: Multiprocessing is complex with CUDA models, running sequentially is safer.
        # We will use the single initialized model for simplicity and robustness.
        print("Processing videos sequentially (num_workers > 1 is for batch mode without CUDA).")
        for task in tqdm(tasks, desc="Processing videos"):
            results.append(process_video_wrapper(task))
    else:
        for task in tqdm(tasks, desc="Processing videos"):
            results.append(process_video_wrapper(task))
    
    successes = [r for r in results if r and r[0] == "success"]
    skipped_runtime = [r for r in results if r and r[0] == "skipped"]
    failures = [r for r in results if r and r[0] == "failure"]
    
    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total videos scanned:        {total_videos_found}")
    print(f"Already processed (skipped): {already_processed_count}")
    print(f"Videos processed this run:   {len(tasks)}")
    print(f"  ✓ Successful:              {len(successes)}")
    print(f"  ⊘ Skipped (no court/player): {len(skipped_runtime)}")
    print(f"  ✗ Failed (errors):         {len(failures)}")
    
    if failures:
        print(f"\n⚠️  First 5 failures:")
        for _, path, error in failures[:5]:
            print(f"  - {os.path.basename(path)}: {error}")
    
    print(f"\n✓ Data saved to: {processed_data_path}")
    print(f"✓ Metadata saved to: {METADATA_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess badminton videos for pose data.")
    
    # Arguments for single video mode
    parser.add_argument("--video", type=str, help="Path to a single video to process.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the output .npy file for a single video.")
    
    # Arguments for batch mode
    parser.add_argument("--raw_data_path", type=str, default="data/", help="Path to raw videos directory (for batch mode).")
    parser.add_argument("--processed_data_path", type=str, default="preprocessed_data_v5_two_player/", 
                       help="Path to save processed .npy files (for batch mode).")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (recommend 1 if using GPU).")
    
    args = parser.parse_args()
    main(args)

