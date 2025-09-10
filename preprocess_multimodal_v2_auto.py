# preprocess_multimodal_v2_auto.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# Assume hit_detect.py is in the project root or accessible via PYTHONPATH
from hit_detect import event_detect

# --- Configuration ---
POSE_DATA_DIR = 'preprocessed_data_v4_auto/' 
SHUTTLE_DATA_DIR = os.path.join('TrackNetV3', 'prediction') 
OUTPUT_DIR = 'preprocessed_multimodal_data_v2_auto/'
SEQUENCE_LENGTH = 32

def load_shuttle_data(video_filename):
    """Loads shuttlecock CSV data into a DataFrame required by event_detect."""
    csv_path = os.path.join(SHUTTLE_DATA_DIR, os.path.splitext(video_filename)[0] + '_ball.csv')
    if not os.path.exists(csv_path): return None
    
    # The event_detect function expects a specific DataFrame structure.
    # We will create a dummy DataFrame that matches it.
    df = pd.read_csv(csv_path)
    df_formatted = pd.DataFrame()
    df_formatted['ball'] = df[['X', 'Y']].values.tolist()
    # Add other columns if your event_detect requires them, otherwise fill with placeholders
    df_formatted['top'] = [[] for _ in range(len(df))]
    df_formatted['bottom'] = [[] for _ in range(len(df))]
    df_formatted['court'] = [[] for _ in range(len(df))]
    df_formatted['net'] = [[] for _ in range(len(df))]
    
    return df, df_formatted # Return both original for feature fusion and formatted for hit detection

def combine_and_window_features(pose_data, shuttle_df_original, hit_frames):
    """Slices data around hit frames and combines pose and shuttle features."""
    num_total_frames = pose_data.shape[0]
    
    # Pre-calculate shuttle features (velocity, acceleration, angle)
    shuttle_df_original['vel_X'] = shuttle_df_original['X'].diff().fillna(0)
    shuttle_df_original['vel_Y'] = shuttle_df_original['Y'].diff().fillna(0)
    shuttle_df_original['accel_X'] = shuttle_df_original['vel_X'].diff().fillna(0)
    shuttle_df_original['accel_Y'] = shuttle_df_original['vel_Y'].diff().fillna(0)
    shuttle_df_original['angle'] = np.arctan2(shuttle_df_original['vel_Y'], shuttle_df_original['vel_X']).fillna(0) * (180. / np.pi)
    shuttle_features_df = shuttle_df_original[['X', 'Y', 'vel_X', 'vel_Y', 'accel_X', 'accel_Y', 'angle']]

    windows = []
    for hit_frame in hit_frames:
        start_frame = hit_frame - (SEQUENCE_LENGTH // 2)
        end_frame = hit_frame + (SEQUENCE_LENGTH // 2)

        if start_frame < 0 or end_frame > num_total_frames: continue

        pose_window = pose_data[start_frame:end_frame]
        shuttle_window = shuttle_features_df.iloc[start_frame:end_frame].to_numpy(dtype=np.float32)
        
        # Final feature vector: 84 (pose+court) + 7 (shuttle) = 91 features
        combined_window = np.concatenate([pose_window, shuttle_window], axis=1)
        windows.append(combined_window)
        
    return windows

def main():
    print("Starting hit-based multimodal data preprocessing with event_detect...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for label in sorted(os.listdir(POSE_DATA_DIR)):
        label_dir = os.path.join(POSE_DATA_DIR, label)
        if not os.path.isdir(label_dir): continue

        output_label_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(output_label_dir, exist_ok=True)
        
        for pose_filename in tqdm(os.listdir(label_dir), desc=f"Processing {label}"):
            if not pose_filename.endswith('.npy'): continue
            
            video_filename = os.path.splitext(pose_filename)[0] + '.mp4'
            shuttle_df_original, shuttle_df_formatted = load_shuttle_data(video_filename)

            if shuttle_df_formatted is not None and not shuttle_df_formatted.empty:
                hit_frames = event_detect(shuttle_df_formatted)
                
                if hit_frames:
                    pose_data = np.load(os.path.join(label_dir, pose_filename))
                    feature_windows = combine_and_window_features(pose_data, shuttle_df_original, hit_frames)
                    
                    for i, window in enumerate(feature_windows):
                        base_name = os.path.splitext(pose_filename)[0]
                        output_filename = f"{base_name}_hit_{i+1}.npy"
                        np.save(os.path.join(output_label_dir, output_filename), window)

    print(f"\nPreprocessing complete! Hit-centered data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()