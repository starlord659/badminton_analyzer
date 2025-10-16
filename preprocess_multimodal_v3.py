import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from hit_detect_v2 import detect_and_attribute_hits

# --- CONFIGURATION ---
POSE_DATA_DIR = Path('preprocessed_data_v5_two_player/')
METADATA_DIR = Path('preprocessed_metadata/')
SHUTTLE_DATA_DIR = Path(r'D:\capstone\badminton_analyzer\TrackNetV3\prediction')
OUTPUT_DIR = Path('preprocessed_multimodal_v5_engineered/')
SEQUENCE_LENGTH = 16
FEATURES_PER_PLAYER = 36
VERBOSE = True
AUGMENT_UNDERREPRESENTED = True
MIN_SAMPLES_THRESHOLD = 800
# ---------------------

logging.basicConfig(level=logging.DEBUG if VERBOSE else logging.INFO,
                    format="[%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- HELPER FUNCTIONS FOR FEATURE ENGINEERING ---
def normalize_shuttle_features(shuttle_df, frame_w, frame_h):
    """
    Normalize shuttle positions and velocities to [0,1] range.
    This ensures feature scales match the normalized pose keypoints.
    """
    shuttle_normalized = shuttle_df.copy()
    
    # Normalize positions to [0, 1]
    shuttle_normalized['X'] = shuttle_df['X'] / frame_w
    shuttle_normalized['Y'] = shuttle_df['Y'] / frame_h
    
    # Calculate velocities in normalized space
    shuttle_normalized['vel_X'] = shuttle_normalized['X'].diff().fillna(0)
    shuttle_normalized['vel_Y'] = shuttle_normalized['Y'].diff().fillna(0)
    
    # Clip velocities to reasonable range (prevents extreme outliers)
    # Max velocity of 0.5 means shuttle can move half the screen per frame
    shuttle_normalized['vel_X'] = shuttle_normalized['vel_X'].clip(-0.5, 0.5)
    shuttle_normalized['vel_Y'] = shuttle_normalized['vel_Y'].clip(-0.5, 0.5)
    
    return shuttle_normalized[['X', 'Y', 'vel_X', 'vel_Y']].to_numpy(dtype=np.float32)

def calculate_angle(p1, p2, p3):
    """Calculates the angle at p2 and normalizes it to [0, 1]. Returns np.nan on failure."""
    if np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any():
        return np.nan # MODIFIED
    v1, v2 = p1 - p2, p3 - p2
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return np.nan # MODIFIED
    cos_angle = np.dot(v1, v2) / norm_product
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle / 180.0

def calculate_distance(p1, p2):
    """Calculates normalized Euclidean distance. Returns np.nan on failure."""
    if np.isnan(p1).any() or np.isnan(p2).any():
        return np.nan # MODIFIED
    # MODIFIED: Normalize distance by the diagonal of the unit square
    # This makes the feature more robust to different aspect ratios.
    return np.linalg.norm(p1 - p2) / np.sqrt(2.0)

def extract_engineered_features(player_pose_kpts, shuttle_pos):
    """Dynamically extracts features, using np.nan for invalid data."""
    RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 6, 8, 10
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 5, 7, 9
    RIGHT_ANKLE, LEFT_ANKLE = 16, 15

    p_right_wrist = player_pose_kpts[RIGHT_WRIST]
    p_left_wrist = player_pose_kpts[LEFT_WRIST]

    dist_right = calculate_distance(p_right_wrist, shuttle_pos)
    dist_left = calculate_distance(p_left_wrist, shuttle_pos)

    # Use 'and not np.isnan()' to handle nan values correctly
    if not np.isnan(dist_right) and (np.isnan(dist_left) or dist_right < dist_left):
        racket_shoulder_idx, racket_elbow_idx, racket_wrist_idx = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
        opposite_ankle_idx = LEFT_ANKLE
    elif not np.isnan(dist_left):
        racket_shoulder_idx, racket_elbow_idx, racket_wrist_idx = LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
        opposite_ankle_idx = RIGHT_ANKLE
    else:
        # MODIFIED: Return an array of NaNs on complete failure
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    p_shoulder = player_pose_kpts[racket_shoulder_idx]
    p_elbow = player_pose_kpts[racket_elbow_idx]
    p_wrist = player_pose_kpts[racket_wrist_idx]
    p_opposite_ankle = player_pose_kpts[opposite_ankle_idx]

    elbow_angle = calculate_angle(p_shoulder, p_elbow, p_wrist)
    posture_proxy = calculate_distance(p_shoulder, p_opposite_ankle)
    
    # MODIFIED: Handle nan values when finding the minimum distance
    valid_dists = [d for d in [dist_right, dist_left] if not np.isnan(d)]
    wrist_shuttle_dist = min(valid_dists) if valid_dists else np.nan

    return np.array([elbow_angle, posture_proxy, wrist_shuttle_dist], dtype=np.float32)

def load_metadata(video_filename: str, label_dir: str) -> Optional[Dict]:
    """Load video metadata (frame dimensions, fps, etc.)"""
    metadata_path = METADATA_DIR / label_dir / (os.path.splitext(video_filename)[0] + '.json')
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    else:
        logger.warning(f"Metadata not found for {video_filename}, using defaults (1920x1080)")
        return { 'frame_width': 1920, 'frame_height': 1080, 'fps': 30.0 }

def load_and_combine_data(video_filename: str, pose_data: np.ndarray, 
                          label_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict]:
    """Loads and aligns shuttle and pose data."""
    metadata = load_metadata(video_filename, label_dir)
    
    csv_path = SHUTTLE_DATA_DIR / (os.path.splitext(video_filename)[0] + '_ball.csv')
    if not csv_path.exists(): 
        logger.warning(f"Shuttle CSV not found for {video_filename}")
        return None, None, metadata
    
    try:
        df_shuttle = pd.read_csv(csv_path)
        if df_shuttle.empty:
            logger.warning(f"Shuttle CSV is empty for {video_filename}")
            return None, None, metadata
    except pd.errors.EmptyDataError:
        logger.warning(f"Shuttle CSV is empty for {video_filename}")
        return None, None, metadata

    max_frames = min(len(df_shuttle), len(pose_data))
    if max_frames == 0:
        logger.warning(f"Frame length mismatch results in 0 frames for {video_filename}")
        return None, None, metadata

    df_shuttle_aligned = df_shuttle.iloc[:max_frames].reset_index(drop=True)
    df_combined = pd.DataFrame(index=range(max_frames))
    
    top_poses = [pose_data[i, :FEATURES_PER_PLAYER] for i in range(max_frames)]
    bottom_poses = [pose_data[i, FEATURES_PER_PLAYER:] for i in range(max_frames)]
    df_combined['top'] = pd.Series(top_poses)
    df_combined['bottom'] = pd.Series(bottom_poses)

    return df_combined, df_shuttle_aligned, metadata

def augment_window(window: np.ndarray, augmentation_type: str = 'time_shift') -> np.ndarray:
    """Data augmentation for underrepresented classes"""
    if augmentation_type == 'time_shift':
        shift = np.random.randint(-2, 3)
        return np.roll(window, shift, axis=0)
    elif augmentation_type == 'noise':
        noise = np.random.normal(0, 0.005, window.shape)
        return window + noise
    elif augmentation_type == 'speed':
        factor = np.random.uniform(0.9, 1.1)
        indices = np.linspace(0, len(window)-1, int(len(window) * factor))
        indices = np.clip(indices, 0, len(window)-1).astype(int)
        augmented = window[indices]
        
        if len(augmented) < len(window):
            pad_width = len(window) - len(augmented)
            augmented = np.vstack([augmented, np.repeat(augmented[-1:], pad_width, axis=0)])
        else:
            augmented = augmented[:len(window)]
        return augmented
    return window

def create_training_windows_from_full_set(full_feature_set: np.ndarray, 
                                         hit_events: List[Dict],
                                         should_augment: bool = False) -> List[np.ndarray]:
    """Create training windows with optional augmentation"""
    num_total_frames = len(full_feature_set)
    all_windows = []
    
    for hit in hit_events:
        center_frame = hit["frame"]
        
        start_frame = center_frame - (SEQUENCE_LENGTH // 2)
        end_frame = start_frame + SEQUENCE_LENGTH
        window = full_feature_set[max(0, start_frame):min(num_total_frames, end_frame)]

        pad_before = max(0, -start_frame)
        pad_after = max(0, end_frame - num_total_frames)

        if pad_before > 0:
            window = np.vstack([np.repeat(window[0:1], pad_before, axis=0), window])
        if pad_after > 0:
            window = np.vstack([window, np.repeat(window[-1:], pad_after, axis=0)])
        
        if window.shape[0] == SEQUENCE_LENGTH:
            all_windows.append(window.astype(np.float32))
            
            if should_augment:
                all_windows.append(augment_window(window, 'time_shift').astype(np.float32))
                all_windows.append(augment_window(window, 'noise').astype(np.float32))
            
    return all_windows

def validate_engineered_features(engineered_features: np.ndarray, video_base: str):
    """Validate quality of engineered features based on NaN ratio"""
    # MODIFIED: Check for np.nan instead of -1.0
    invalid_ratio = np.isnan(engineered_features).mean()
    
    if invalid_ratio > 0.5:
        logger.warning(f"Video {video_base} has {invalid_ratio:.1%} invalid (NaN) engineered features. Skipping.")
        return False
    
    if VERBOSE and invalid_ratio > 0.2:
        logger.info(f"Video {video_base} engineered feature quality: {(1-invalid_ratio)*100:.1f}% valid")
    
    return True

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_files_scanned = 0
    total_files_skipped = 0
    total_windows_saved = 0
    class_statistics = {}

    logger.info("Analyzing class distribution...")
    for label_dir in sorted([d for d in POSE_DATA_DIR.iterdir() if d.is_dir()]):
        npy_files = list(label_dir.glob('*.npy'))
        class_statistics[label_dir.name] = {
            'file_count': len(npy_files),
            'needs_augmentation': len(npy_files) < MIN_SAMPLES_THRESHOLD
        }
    
    logger.info("\nClass Distribution:")
    for class_name, stats in class_statistics.items():
        aug_status = "AUGMENT" if stats['needs_augmentation'] else "OK"
        logger.info(f"  {class_name}: {stats['file_count']} files [{aug_status}]")
    
    logger.info("\nProcessing videos...")
    
    is_training_set = True
    
    for label_dir in sorted([d for d in POSE_DATA_DIR.iterdir() if d.is_dir()]):
        output_label_dir = OUTPUT_DIR / label_dir.name
        output_label_dir.mkdir(exist_ok=True)
        windows_saved_for_label = 0
        npy_files = sorted(label_dir.glob('*.npy'))
        
        should_augment = (is_training_set and 
                          class_statistics[label_dir.name]['needs_augmentation'] and 
                          AUGMENT_UNDERREPRESENTED)
        
        for pose_path in tqdm(npy_files, desc=f"Processing {label_dir.name}"):
            total_files_scanned += 1
            video_base = pose_path.stem
            pose_data_raw = np.load(pose_path)

            if pose_data_raw.shape[1] != FEATURES_PER_PLAYER * 2:
                logger.warning(f"Unexpected pose data shape in {video_base}: {pose_data_raw.shape}. Skipping.")
                total_files_skipped += 1
                continue

            combined_df, shuttle_df, metadata = load_and_combine_data(
                video_base + '.mp4', pose_data_raw, label_dir.name
            )
            
            if combined_df is None:
                total_files_skipped += 1
                continue

            pose_data = pose_data_raw[:len(combined_df)]
            
            frame_w = metadata['frame_width']
            frame_h = metadata['frame_height']
            fps = metadata.get('fps', 30.0)
            
            # ========== MODIFIED SECTION ==========
            all_engineered_features = []
            
            # STEP 1: Normalize shuttle positions FIRST
            shuttle_features = normalize_shuttle_features(shuttle_df, frame_w, frame_h)
            shuttle_positions_norm = shuttle_features[:, :2]  # Already normalized X, Y
            
            # STEP 2: Extract engineered features using normalized shuttle positions
            for i in range(len(pose_data)):
                top_player_kpts_norm = pose_data[i, :34].reshape(17, 2)
                bottom_player_kpts_norm = pose_data[i, 36:70].reshape(17, 2)
                
                # Use already-normalized shuttle position
                shuttle_pos_norm = shuttle_positions_norm[i]
                
                top_features = extract_engineered_features(top_player_kpts_norm, shuttle_pos_norm)
                bottom_features = extract_engineered_features(bottom_player_kpts_norm, shuttle_pos_norm)
                all_engineered_features.append(np.concatenate([top_features, bottom_features]))
            # ========== END MODIFIED SECTION ==========
            
            engineered_features_np = np.array(all_engineered_features)
            
            if not validate_engineered_features(engineered_features_np, video_base):
                total_files_skipped += 1
                continue
            
            # Detect hits using PIXEL coordinates (hit detector expects this)
            shuttle_df_pixel = shuttle_df.copy()
            hit_events = detect_and_attribute_hits(
                shuttle_df_pixel, fps=fps, frame_w=frame_w, frame_h=frame_h, verbose=False
            )
            
            if not hit_events:
                if VERBOSE: logger.info(f"No hits detected in {video_base}")
                continue

            # Use the normalized shuttle features we calculated earlier
            if len(pose_data) != len(shuttle_features) or len(pose_data) != len(engineered_features_np):
                logger.warning(f"Data length inconsistency in {video_base}. Skipping.")
                total_files_skipped += 1
                continue
            
            engineered_features_filled = np.nan_to_num(engineered_features_np, nan=0.0)
            full_feature_set = np.concatenate([pose_data, engineered_features_filled, shuttle_features], axis=1)
            
            feature_windows = create_training_windows_from_full_set(
                full_feature_set, hit_events, should_augment=should_augment
            )
            
            if feature_windows:
                windows_saved_for_label += len(feature_windows)
                for i, window in enumerate(feature_windows):
                    sample_name = f"{video_base}_hit_{i+1}.npy"
                    save_path = output_label_dir / sample_name
                    np.save(save_path, window)
        
        total_windows_saved += windows_saved_for_label
        logger.info(f"Finished label '{label_dir.name}'. Windows saved: {windows_saved_for_label}")
    
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total video files scanned: {total_files_scanned}")
    logger.info(f"Total windows generated:   {total_windows_saved}")
    if total_files_scanned > 0:
        logger.info(f"Files skipped due to data issues: {total_files_skipped} ({total_files_skipped/total_files_scanned:.2%})")
    logger.info(f"Sequence length: {SEQUENCE_LENGTH} frames")
    logger.info(f"Final dataset saved to: {OUTPUT_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    main()