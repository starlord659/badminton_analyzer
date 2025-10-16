import cv2
import numpy as np
import pandas as pd
import time
from collections import deque # <-- Import deque for the trail
from hit_detect_v2 import detect_and_attribute_hits, plot_impact_scores

# --- Paths and Constants ---
VIDEO_PATH = "my_video.mp4"
POSE_PATH = "preprocessed_data_v5_two_player/NewVideos/my_video.npy"
SHUTTLE_CSV = r"D:\capstone\badminton_analyzer\TrackNetV3\prediction\my_video_ball.csv"
TRAIL_LENGTH = 15 # How long the shuttle's tail is

# --- Robust Load and Prepare Data ---
print("Loading and merging data sources...")
pose_data = np.load(POSE_PATH, allow_pickle=True)
pose_df = pd.DataFrame(pose_data[:, :72]); pose_df['frame'] = pose_df.index
shuttle_df = pd.read_csv(SHUTTLE_CSV).rename(columns={'Frame': 'frame', 'X': 'shuttle_x', 'Y': 'shuttle_y', 'Visibility': 'visibility'})
df = pd.merge(pose_df, shuttle_df[['frame', 'shuttle_x', 'shuttle_y', 'visibility']], on='frame', how='inner')
df = df.set_index('frame').sort_index().rename(columns={'shuttle_x': 'X', 'shuttle_y': 'Y', 'visibility': 'Visibility'})
print(f"Successfully merged data. Found {len(df)} aligned frames.")

# --- Video Capture ---
cap = cv2.VideoCapture(VIDEO_PATH)
print(f"Video: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_duration_ms = 1000 / fps

# --- Step 1: Interactively Tune Hit Detection ---
initial_hit_params = {
    'peak_prominence': 0.6, 
    'peak_height': 0.01,
    'slope_thresh': 30,
    'min_hit_separation_seconds': 0.25
}
# The plot function now returns the best parameters you chose
final_hit_params = plot_impact_scores(df.copy(), fps=fps, initial_params=initial_hit_params)

# --- Step 2: Run Final Detection with Tuned Parameters ---
print("\nRunning final hit detection with your tuned parameters...")
hit_events = detect_and_attribute_hits(df, fps=fps, verbose=True, **final_hit_params)
hit_frames_set = {hit['frame'] for hit in hit_events}
print(f"✅ Detection complete. Found {len(hit_frames_set)} hits: {sorted(list(hit_frames_set))}\n")

# --- Convert data to NumPy for performance ---
shuttle_pos = df[['X', 'Y']].to_numpy(dtype=float)
visibility_data = df['Visibility'].to_numpy(dtype=int)
valid_frames = df.index.to_numpy()

frame_idx = 0
pose_map_idx = 0
shuttle_trail = deque(maxlen=TRAIL_LENGTH) # <-- Initialize the trail deque

# -----------------------------------------------------------------
# FINAL: Synchronized Main Video Loop with Shuttle Trail
# -----------------------------------------------------------------
while True:
    loop_start_time = time.time()
    
    ret, frame = cap.read()
    if not ret: break
    canvas = frame.copy()

    current_shuttle_pos = None
    if pose_map_idx < len(valid_frames) and frame_idx == valid_frames[pose_map_idx]:
        vis_status = visibility_data[pose_map_idx]
        current_shuttle_pos = shuttle_pos[pose_map_idx]
        pose_map_idx += 1
    else:
        vis_status = 0
    
    # --- NEW: Add current position to trail and draw it ---
    if current_shuttle_pos is not None and not np.isnan(current_shuttle_pos).any():
        shuttle_trail.append(current_shuttle_pos.astype(int))
    
    for i, pos in enumerate(shuttle_trail):
        alpha = (i + 1) / len(shuttle_trail)
        radius = int(1 + 3 * alpha)
        color = (0, int(200 * alpha + 55), int(240 * alpha + 15))
        cv2.circle(canvas, tuple(pos), radius, color, -1)
    
    # Draw hit animation
    if frame_idx in hit_frames_set:
        shuttle_idx_list = np.where(valid_frames == frame_idx)[0]
        if shuttle_idx_list.size > 0:
            sx, sy = shuttle_pos[shuttle_idx_list[0]].astype(int)
            cv2.circle(canvas, (sx, sy), 25, (0, 100, 255), 3, cv2.LINE_AA)

    # Display status text
    vis_text = "Shuttle Visible" if vis_status == 1 else "Shuttle NOT Visible"; vis_color = (0,255,0) if vis_status==1 else (0,0,255)
    cv2.putText(canvas, vis_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, vis_color, 2)
    cv2.putText(canvas, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    cv2.imshow("Debug - Hit Analysis Mode", canvas)

    # Synchronized waitKey
    elapsed_time_ms = (time.time() - loop_start_time) * 1000
    time_to_wait_ms = frame_duration_ms - elapsed_time_ms
    wait_key = max(1, int(time_to_wait_ms))
    if cv2.waitKey(wait_key) & 0xFF == ord('q'): break
    
    frame_idx += 1

cap.release(); cv2.destroyAllWindows()