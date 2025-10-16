# hit_detect_fixed.py
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import warnings

# --- Pose class for attribution ---
class Pose:
    def __init__(self):
        self.kp = np.full((17, 2), np.nan)

    def init_from_kparray(self, kparray, frame_w=640, frame_h=360):
        arr = np.array(kparray, dtype=float).flatten()[:34].reshape(-1, 2)
        if arr.shape[0] < 17:
            pad_width = ((0, 17 - arr.shape[0]), (0, 0))
            arr = np.pad(arr, pad_width, constant_values=np.nan)
        if not np.isnan(arr).all() and np.nanmax(arr) <= 1.05:
            arr[:, 0] *= frame_w
            arr[:, 1] *= frame_h
        self.kp[:arr.shape[0]] = arr

    def get_hand_position(self):
        if not np.isnan(self.kp[10]).any():
            return self.kp[10]
        if not np.isnan(self.kp[9]).any():
            return self.kp[9]
        valid_points = self.kp[~np.isnan(self.kp).any(axis=1)]
        return np.mean(valid_points, axis=0) if len(valid_points) > 0 else np.array([np.nan, np.nan])

    def get_centroid(self):
        valid_points = self.kp[~np.isnan(self.kp).any(axis=1)]
        return np.mean(valid_points, axis=0) if len(valid_points) > 0 else np.array([np.nan, np.nan])


class HitDetector:
    def __init__(self, fps=30,
                 smoothing_window=5, polyorder=2, peak_prominence=0.15, peak_height=0.1,
                 slope_thresh=10, slope_window=7,
                 min_hit_separation_seconds=0.25,
                 frame_resolution=(1920, 1080)):
        
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.polyorder = polyorder
        self.peak_prominence = peak_prominence
        self.peak_height = peak_height
        self.slope_thresh = slope_thresh
        self.slope_window = slope_window
        self.min_hit_separation_frames = int(fps * min_hit_separation_seconds)
        self.frame_resolution = frame_resolution
        
        # FIXED: Adaptive jump threshold based on resolution
        diagonal = np.sqrt(frame_resolution[0]**2 + frame_resolution[1]**2)
        self.jump_threshold = diagonal * 0.10  # 10% of screen diagonal

    def _get_continuous_segments(self, df: pd.DataFrame) -> list:
        """FIXED: Uses resolution-aware camera cut detection"""
        df = df.copy()
        df['prev_X'] = df['X'].shift(1)
        df['prev_Y'] = df['Y'].shift(1)
        df['jump_dist'] = np.sqrt((df['X'] - df['prev_X'])**2 + (df['Y'] - df['prev_Y'])**2)
        
        # Use adaptive threshold
        df['new_segment_start'] = ((df['Visibility'] == 0) & (df['Visibility'].shift(1) == 1)) | \
                                  (df['jump_dist'] > self.jump_threshold)
        df['segment'] = df['new_segment_start'].cumsum()
        
        visible_df = df[df['Visibility'] == 1].copy()
        return [g for _, g in visible_df.groupby('segment') if len(g) > self.smoothing_window]

    def _calculate_kinematics(self, segment: pd.DataFrame) -> pd.DataFrame:
        segment = segment.copy()
        segment['x_smooth'] = savgol_filter(segment['X'], self.smoothing_window, self.polyorder)
        segment['y_smooth'] = savgol_filter(segment['Y'], self.smoothing_window, self.polyorder)
        dt = 1.0 / self.fps
        segment['vx'] = segment['x_smooth'].diff().fillna(0) / dt
        segment['vy'] = segment['y_smooth'].diff().fillna(0) / dt
        segment['acceleration'] = np.abs(np.sqrt(segment['vx']**2 + segment['vy']**2).diff().fillna(0) / dt)
        angles = np.unwrap(np.arctan2(segment['vy'], segment['vx']))
        segment['angular_change_rad'] = np.abs(np.diff(angles, prepend=angles[0]))
        return segment

    def _normalize_signal(self, signal: pd.Series) -> pd.Series:
        min_val, max_val = signal.min(), signal.max()
        return (signal - min_val) / (max_val - min_val) if (max_val - min_val) > 1e-6 else pd.Series(np.zeros_like(signal), index=signal.index)

    def _find_slope_peaks(self, z: np.ndarray) -> list:
        w = self.slope_window
        peaks = []
        for i in range(w + 1, len(z) - w - 1):
            if (z[i] - z[i-1]) * (z[i] - z[i+1]) < 0:
                continue
            left_slope = abs(np.median(z[i-w+1:i+1] - z[i-w:i]))
            right_slope = abs(np.median(z[i+1:i+w+1] - z[i:i+w]))
            if max(left_slope, right_slope) > self.slope_thresh:
                peaks.append(i)
        return peaks

    def detect_from_df(self, df: pd.DataFrame) -> list:
        all_hit_frames = []
        df = df.copy()
        df.index.name = 'Frame'
        segments = self._get_continuous_segments(df)
        
        for segment in segments:
            kinematics = self._calculate_kinematics(segment.copy())
            accel_norm = self._normalize_signal(kinematics['acceleration'])
            angle_norm = self._normalize_signal(kinematics['angular_change_rad'])
            impact_score = (accel_norm * angle_norm).fillna(0)
            peaks1, _ = find_peaks(impact_score, height=self.peak_height, prominence=self.peak_prominence)
            peaks_x = self._find_slope_peaks(kinematics['x_smooth'].values)
            peaks_y = self._find_slope_peaks(kinematics['y_smooth'].values)
            combined_peaks = set(peaks1) | set(peaks_x) | set(peaks_y)
            hit_frames = kinematics.index[list(combined_peaks)].tolist()
            all_hit_frames.extend(hit_frames)
            
        final_hits = []
        if not all_hit_frames:
            return []
        
        sorted_hits = sorted(list(set(all_hit_frames)))
        last_hit = -9999
        for frame in sorted_hits:
            if frame - last_hit > self.min_hit_separation_frames:
                final_hits.append(frame)
                last_hit = frame
        return final_hits


# FIXED: Efficient pose caching to avoid redundant lookups
def get_closest_valid_pose(df, shuttle_pos, search_frames, frame_w, frame_h, pose_cache=None):
    """Optimized with pose caching"""
    if pose_cache is None:
        pose_cache = {}
    
    min_dist_p0, min_dist_p1 = float('inf'), float('inf')
    
    for frame in search_frames:
        if frame not in df.index:
            continue
        
        # Check cache first
        if frame not in pose_cache:
            try:
                pose_cache[frame] = (
                    df.loc[frame].iloc[0:36].values,
                    df.loc[frame].iloc[36:72].values
                )
            except Exception:
                continue
        
        p0_data, p1_data = pose_cache[frame]
        
        p0, p1 = Pose(), Pose()
        p0.init_from_kparray(p0_data, frame_w, frame_h)
        p1.init_from_kparray(p1_data, frame_w, frame_h)
        
        for player_idx, pose in enumerate([p0, p1]):
            hand = pose.get_hand_position()
            pos = pose.get_centroid() if np.isnan(hand).any() else hand
            
            if not np.isnan(pos).any():
                dist = np.linalg.norm(shuttle_pos - pos)
                if player_idx == 0 and dist < min_dist_p0:
                    min_dist_p0 = dist
                if player_idx == 1 and dist < min_dist_p1:
                    min_dist_p1 = dist
    
    return min_dist_p0, min_dist_p1


def detect_and_attribute_hits(df, fps=30, max_reach_threshold=850, frame_w=640, frame_h=360, verbose=False, **kwargs):
    """FIXED: Now accepts frame_w and frame_h as parameters with proper resolution handling"""
    
    # Extract frame resolution if provided
    frame_resolution = kwargs.pop('frame_resolution', (frame_w, frame_h))
    
    detector = HitDetector(fps=fps, frame_resolution=frame_resolution, **kwargs)
    candidate_frames = detector.detect_from_df(df[['Visibility', 'X', 'Y']])
    
    if verbose:
        print(f"  > Found {len(candidate_frames)} candidate hits from ensemble detector.")
        print(f"  > Using resolution: {frame_w}x{frame_h}")
        print(f"  > Jump threshold: {detector.jump_threshold:.1f} pixels")
    
    hit_events = []
    pose_cache = {}  # Shared cache for efficiency
    
    for frame in candidate_frames:
        try:
            shuttle_pos = np.array([df.loc[frame, 'X'], df.loc[frame, 'Y']], dtype=float)
            if np.isnan(shuttle_pos).any():
                continue
            
            search_window = range(frame - 3, frame + 4)
            dist_p0, dist_p1 = get_closest_valid_pose(df, shuttle_pos, search_window, frame_w, frame_h, pose_cache)
            
            # Adaptive reach threshold based on resolution
            diagonal = np.sqrt(frame_w**2 + frame_h**2)
            reach_thresh = min(max_reach_threshold, diagonal * 0.4)  # Max 40% of screen diagonal
            
            if verbose:
                print(f"  Frame {frame}: shuttle={shuttle_pos}, dist_p0={dist_p0:.1f}, dist_p1={dist_p1:.1f}, threshold={reach_thresh:.1f}")
            
            if dist_p0 < dist_p1 and dist_p0 < reach_thresh:
                hit_events.append({'frame': frame, 'player': 'Top'})
            elif dist_p1 < dist_p0 and dist_p1 < reach_thresh:
                hit_events.append({'frame': frame, 'player': 'Bottom'})
        except Exception as e:
            if verbose:
                warnings.warn(f"[Attribution] Frame {frame} skipped: {e}")
    
    if verbose:
        print(f"  > Attributed {len(hit_events)} hits to players.")
    
    return hit_events


def plot_impact_scores(df, fps, initial_params, frame_resolution=None):
    """Interactive debugging function to plot impact score and tune parameters with sliders."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
    except ImportError:
        print("Please install matplotlib: pip install matplotlib")
        return initial_params

    print("--- Generating plot data. This may take a moment. ---")
    
    # Get frame resolution - use provided value or detect from df
    if frame_resolution is None:
        frame_w = df.get('frame_width', 1920) if hasattr(df, 'get') else 1920
        frame_h = df.get('frame_height', 1080) if hasattr(df, 'get') else 1080
        frame_resolution = (frame_w, frame_h)
    
    # Create detector with frame_resolution but WITHOUT it in initial_params
    detector = HitDetector(fps=fps, frame_resolution=frame_resolution, **initial_params)
    
    all_scores = pd.Series(dtype=np.float64)
    segments = detector._get_continuous_segments(df[['Visibility', 'X', 'Y']])
    
    for segment in segments:
        if segment.empty:
            continue
        kinematics = detector._calculate_kinematics(segment.copy())
        accel_norm = detector._normalize_signal(kinematics['acceleration'])
        angle_norm = detector._normalize_signal(kinematics['angular_change_rad'])
        impact_score = (accel_norm * angle_norm).fillna(0)
        all_scores = pd.concat([all_scores, impact_score])

    if all_scores.empty:
        print("No visible shuttle segments found to plot.")
        return initial_params

    fig, ax = plt.subplots(figsize=(18, 7))
    plt.subplots_adjust(bottom=0.3)
    
    line, = ax.plot(all_scores.index, all_scores.values, label='Impact Score (Accel*Angle)', alpha=0.7)
    scatter = ax.scatter([], [], color='r', zorder=5, s=50, label='Detected Hits')
    ax.set_title('Interactive Hit Detector Tuning')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Normalized Impact Score')
    ax.grid(True)
    ax.legend()

    ax_prom = plt.axes([0.25, 0.18, 0.65, 0.03])
    prom_slider = Slider(ax=ax_prom, label='Prominence', valmin=0.01, valmax=0.5, 
                        valinit=initial_params.get('peak_prominence', 0.15), valstep=0.01)

    ax_height = plt.axes([0.25, 0.12, 0.65, 0.03])
    height_slider = Slider(ax=ax_height, label='Height', valmin=0.01, valmax=0.5, 
                          valinit=initial_params.get('peak_height', 0.1), valstep=0.01)

    ax_slope = plt.axes([0.25, 0.06, 0.65, 0.03])
    slope_slider = Slider(ax=ax_slope, label='Slope Thresh', valmin=1, valmax=50, 
                         valinit=initial_params.get('slope_thresh', 10), valstep=1)

    def update(val):
        params = {
            **initial_params,
            'peak_prominence': prom_slider.val,
            'peak_height': height_slider.val,
            'slope_thresh': slope_slider.val
        }
        
        tuned_detector = HitDetector(fps=fps, frame_resolution=frame_resolution, **params)
        hit_frames = tuned_detector.detect_from_df(df[['Visibility', 'X', 'Y']])
        
        valid_hit_frames = [f for f in hit_frames if f in all_scores.index]
        
        if valid_hit_frames:
            scatter.set_offsets(np.c_[valid_hit_frames, all_scores.loc[valid_hit_frames].values])
        else:
            scatter.set_offsets(np.empty((0, 2)))
            
        ax.legend([line, scatter], ['Impact Score', f'Detected Hits ({len(valid_hit_frames)})'])
        fig.canvas.draw_idle()

    prom_slider.on_changed(update)
    height_slider.on_changed(update)
    slope_slider.on_changed(update)
    update(None)

    print("\n--- INTERACTIVE TUNING ---")
    print("Adjust sliders to select the correct peaks.")
    print("Close the plot window when you are done to continue.")
    plt.show()

    final_params = {
        'peak_prominence': prom_slider.val,
        'peak_height': height_slider.val,
        'slope_thresh': slope_slider.val,
        'min_hit_separation_seconds': initial_params.get('min_hit_separation_seconds', 0.25)
    }
    print(f"\nFinal tuned parameters: {final_params}")
    return final_params


def detect_from_df(self, df: pd.DataFrame, verbose: bool = False) -> list:
    all_hit_frames = []
    df = df.copy()
    df.index.name = 'Frame'
    segments = self._get_continuous_segments(df)
    
    # DEBUG OUTPUT
    if verbose:
        print(f"\n[HIT DETECTOR DEBUG]")
        print(f"  Resolution used: {self.frame_resolution}")
        print(f"  Total frames: {len(df)}")
        print(f"  Visible frames: {(df['Visibility'] == 1).sum()}")
        print(f"  Jump threshold: {self.jump_threshold:.1f} pixels")
        print(f"  Number of continuous segments found: {len(segments)}")
        
        if len(segments) == 0:
            print(f"  ❌ NO SEGMENTS FOUND - This is the problem!")
            print(f"     All shuttlecock motion is being filtered as camera cuts.")
            print(f"     Current jump_threshold: {self.jump_threshold:.1f}")
            print(f"     Try manually setting frame_resolution to match shuttle coords.")
            return []
    
    for i, segment in enumerate(segments):
        if verbose:
            print(f"  Segment {i+1}: {len(segment)} frames (frames {segment.index[0]}-{segment.index[-1]})")
        
        if len(segment) <= self.smoothing_window:
            if verbose:
                print(f"    ⚠️ Segment too short (need >{self.smoothing_window} frames). Skipping.")
            continue
            
        kinematics = self._calculate_kinematics(segment.copy())
        accel_norm = self._normalize_signal(kinematics['acceleration'])
        angle_norm = self._normalize_signal(kinematics['angular_change_rad'])
        impact_score = (accel_norm * angle_norm).fillna(0)
        peaks1, _ = find_peaks(impact_score, height=self.peak_height, prominence=self.peak_prominence)
        peaks_x = self._find_slope_peaks(kinematics['x_smooth'].values)
        peaks_y = self._find_slope_peaks(kinematics['y_smooth'].values)
        combined_peaks = set(peaks1) | set(peaks_x) | set(peaks_y)
        
        if verbose:
            print(f"    - Impact peaks: {len(peaks1)}, X-slope peaks: {len(peaks_x)}, Y-slope peaks: {len(peaks_y)}")
            print(f"    - Combined candidate peaks: {len(combined_peaks)}")
        
        hit_frames = kinematics.index[list(combined_peaks)].tolist()
        all_hit_frames.extend(hit_frames)
        
    final_hits = []
    if not all_hit_frames:
        if verbose:
            print(f"  ⚠️ NO PEAKS FOUND in any segment")
            print(f"     Try adjusting: peak_prominence={self.peak_prominence}, peak_height={self.peak_height}, slope_thresh={self.slope_thresh}")
        return []
    
    sorted_hits = sorted(list(set(all_hit_frames)))
    last_hit = -9999
    for frame in sorted_hits:
        if frame - last_hit > self.min_hit_separation_frames:
            final_hits.append(frame)
            last_hit = frame
    
    if verbose:
        print(f"  ✓ Total candidate hits: {len(sorted_hits)}")
        print(f"  ✓ Final hits after {self.min_hit_separation_frames}-frame separation: {len(final_hits)}")
    
    return final_hits