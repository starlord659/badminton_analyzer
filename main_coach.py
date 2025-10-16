#!/usr/bin/env python3
"""
AI Badminton Coach - Production Ready Video Analysis System
===========================================================
Automatically processes badminton videos with:
- Player skeleton tracking
- Shuttlecock detection with hit detection
- Court boundary visualization  
- Real-time shot classification
- Professional visual overlays
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import math
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Dict, List
import logging

# ============================================================================
# TUNABLE PARAMETERS - ADJUST THESE FOR BETTER RESULTS
# ============================================================================

class Config:
    """Central configuration for all tunable parameters"""
    
    # ========================================================================
    # HIT DETECTION PARAMETERS (Most Important for Accuracy)
    # ========================================================================
    HIT_DETECTION = {
        # Smoothing and peak detection
        'smoothing_window': 5,           # Savitzky-Golay window (5-11, odd numbers)
        'polyorder': 2,                  # Polynomial order for smoothing (2-3)
        'peak_prominence': 0.08,         # Min peak prominence (0.1-0.3, higher = fewer hits)
        'peak_height': 0.05,              # Min peak height (0.05-0.2, higher = fewer hits)
        'slope_thresh': 45,              # Slope change threshold (5-20, higher = fewer hits)
        'slope_window': 7,               # Window for slope detection (5-9)
        
        # Temporal filtering
        'min_hit_separation_seconds': 0.25,  # Min time between hits (0.2-0.5s)
        
        # Spatial filtering
        'max_reach_threshold': 850,      # Max player reach in pixels (500-1000)
        
        # Enable verbose output for debugging
        'verbose': False                 # Set to True to see hit detection details
    }
    
    # ========================================================================
    # SHOT CLASSIFICATION FILTERING
    # ========================================================================
    CLASSIFICATION = {
        # Confidence thresholds
        'min_confidence': 0,           # Only show predictions above this (0.0-0.7)
        'high_confidence_threshold': 0.7, # Green border above this
        
        # Temporal smoothing (reduce flickering predictions)
        'use_temporal_smoothing': False,  # Enable prediction smoothing
        'smoothing_window_frames': 5,    # Frames to average over (3-7)
    }
    
    # ========================================================================
    # VISUALIZATION PARAMETERS
    # ========================================================================
    VISUAL = {
        # Animation durations
        'hit_animation_duration': 0.75,   # Seconds (0.5-1.5)
        'prediction_hold_duration': 2.0,  # Seconds (1.0-3.0)
        
        # Visual styles
        'skeleton_thickness': 2,          # Line thickness (1-3)
        'shuttle_radius': 12,             # Shuttle circle size (8-16)
        'court_thickness': 3,             # Court line thickness (2-4)
        
        # Text sizes
        'prediction_font_scale': 1.0,     # Font size (0.7-1.2)
        'hit_text_scale': 0.9,            # Hit text size (0.7-1.1)
    }

# ============================================================================
# IMPORT CUSTOM MODULES
# ============================================================================
try:
    from hit_detect_v2 import detect_and_attribute_hits
except ImportError:
    print("❌ ERROR: 'hit_detect_v2.py' not found in current directory")
    sys.exit(1)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Color Palette (BGR format for OpenCV)
COLORS = {
    'player_top': (255, 165, 0),      # Orange - Top player
    'player_bottom': (0, 0, 255),     # Red - Bottom player  
    'shuttle': (0, 255, 0),           # Green - Shuttlecock
    'court': (0, 255, 255),           # Yellow - Court lines
    'hit_gradient': (0, 200, 0),      # Green - Hit animation
    'prediction_bg': (40, 40, 40),    # Dark gray - Prediction panel
    'prediction_text': (255, 255, 255) # White - Text
}

# Skeleton connections (COCO-17 keypoint format)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head
    (5, 6), (11, 12), (5, 11), (6, 12),       # Torso
    (5, 7), (7, 9), (6, 8), (8, 10),          # Arms
    (11, 13), (13, 15), (12, 14), (14, 16)    # Legs
]

# Animation parameters
HIT_ANIMATION_FRAMES = int(30 * Config.VISUAL['hit_animation_duration'])
PREDICTION_HOLD_FRAMES = int(30 * Config.VISUAL['prediction_hold_duration'])

# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def print_config():
    """Print current configuration for debugging"""
    logging.info("\n" + "="*70)
    logging.info("CURRENT CONFIGURATION")
    logging.info("="*70)
    logging.info("\n🎯 Hit Detection Parameters:")
    for key, value in Config.HIT_DETECTION.items():
        logging.info(f"   {key}: {value}")
    logging.info("\n🧠 Classification Parameters:")
    for key, value in Config.CLASSIFICATION.items():
        logging.info(f"   {key}: {value}")
    logging.info("\n🎨 Visual Parameters:")
    for key, value in Config.VISUAL.items():
        logging.info(f"   {key}: {value}")
    logging.info("="*70 + "\n")


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(verbose: bool = False):
    """Configure logging with proper formatting"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def draw_skeleton(frame: np.ndarray, keypoints_norm: np.ndarray, 
                 frame_dims: Tuple[int, int], color: Tuple[int, int, int],
                 thickness: int = None) -> None:
    """
    Draw player skeleton with proper coordinate handling
    
    Args:
        frame: Image to draw on
        keypoints_norm: Normalized keypoints [17, 2] in range [0, 1]
        frame_dims: (width, height) of frame
        color: BGR color tuple
        thickness: Line thickness (uses Config if None)
    """
    if thickness is None:
        thickness = Config.VISUAL['skeleton_thickness']
    
    w, h = frame_dims
    
    # Convert normalized coords to pixels
    keypoints_px = (keypoints_norm.reshape(17, 2) * np.array([w, h])).astype(int)
    
    # Draw skeleton connections
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        p1 = tuple(keypoints_px[start_idx])
        p2 = tuple(keypoints_px[end_idx])
        
        # Only draw if both points are valid (not at origin)
        if (p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0):
            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
    
    # Draw keypoint circles
    for px, py in keypoints_px:
        if px > 0 and py > 0:
            cv2.circle(frame, (px, py), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)  # White outline


def draw_court(frame: np.ndarray, court_polygon: Optional[np.ndarray], 
              color: Tuple[int, int, int]) -> None:
    """Draw court boundary lines"""
    if court_polygon is not None and len(court_polygon) >= 4:
        thickness = Config.VISUAL['court_thickness']
        cv2.polylines(frame, [court_polygon.astype(int)], True, color, thickness, cv2.LINE_AA)


def draw_shuttle(frame: np.ndarray, shuttle_pos: Tuple[int, int], 
                color: Tuple[int, int, int], radius: int = None) -> None:
    """Draw shuttlecock with red outline"""
    if radius is None:
        radius = Config.VISUAL['shuttle_radius']
    
    x, y = shuttle_pos
    if x > 0 and y > 0:
        # Outer red circle
        cv2.circle(frame, (x, y), radius + 2, (0, 0, 255), 2, cv2.LINE_AA)
        # Inner green circle
        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)


def draw_hit_animation(frame: np.ndarray, pos: Tuple[int, int], 
                      ttl: int, max_ttl: int) -> None:
    """
    Draw animated green gradient on hit with 'HIT' text
    
    Args:
        frame: Image to draw on
        pos: Center position (x, y)
        ttl: Time-to-live remaining frames
        max_ttl: Maximum animation duration
    """
    if ttl <= 0:
        return
    
    progress = ttl / max_ttl  # 1.0 -> 0.0
    
    # Expanding ring effect
    base_radius = 15
    ring_radius = int(base_radius + 30 * (1 - progress))
    
    # Color transitions from bright to dark green
    intensity = int(155 + 100 * progress)
    color = (0, intensity, 0)
    
    # Draw multiple rings for gradient effect
    overlay = frame.copy()
    for i in range(3):
        r = ring_radius - i * 5
        alpha = 0.4 * progress
        cv2.circle(overlay, pos, r, color, 2, cv2.LINE_AA)
    
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw 'HIT' text
    font = cv2.FONT_HERSHEY_DUPLEX  # Use DUPLEX instead of BOLD
    text = "HIT"
    text_scale = Config.VISUAL['hit_text_scale']
    text_size = cv2.getTextSize(text, font, text_scale, 2)[0]
    text_x = pos[0] + ring_radius + 5
    text_y = pos[1] + text_size[1] // 2
    
    # Text shadow for better visibility
    cv2.putText(frame, text, (text_x + 2, text_y + 2), font, text_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (text_x, text_y), font, text_scale, (200, 255, 200), 2, cv2.LINE_AA)


def draw_prediction_panel(frame: np.ndarray, text: str, frame_w: int, 
                         confidence: float = 0.0) -> None:
    """
    Draw semi-transparent prediction panel in top-left corner
    
    Args:
        frame: Image to draw on
        text: Prediction text (e.g., "Clear (0.95)")
        frame_w: Frame width for positioning
        confidence: Confidence score for color coding
    """
    panel_h = 70
    panel_w = min(400, frame_w - 20)
    
    # Create semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), 
                 COLORS['prediction_bg'], -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Add border
    border_color = (0, 255, 0) if confidence > Config.CLASSIFICATION['high_confidence_threshold'] else (0, 165, 255)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), 
                 border_color, 2, cv2.LINE_AA)
    
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = Config.VISUAL['prediction_font_scale']
    cv2.putText(frame, text, (25, 55), font, font_scale, 
               COLORS['prediction_text'], 2, cv2.LINE_AA)

# ============================================================================
# MODEL ARCHITECTURE (Transformer Classifier)
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """Transformer-based shot type classifier"""
    def __init__(self, input_size: int, d_model: int, nhead: int, 
                 num_encoder_layers: int, dim_feedforward: int, 
                 num_classes: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = nn.Linear(d_model, num_classes)
    
    def forward(self, src: torch.Tensor, 
                src_key_padding_mask: Optional[torch.Tensor] = None):
        src = self.input_projection(src) * math.sqrt(self.d_model)
        
        batch_size = src.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        src = torch.cat((cls_tokens, src), dim=1)
        src = self.pos_encoder(src)
        
        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=src.device)
            src_key_padding_mask = torch.cat((cls_mask, src_key_padding_mask), dim=1)
        
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return self.decoder(output[:, 0, :])

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def calculate_angle(p1, p2, p3) -> float:
    """Calculate normalized angle at p2"""
    if np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any():
        return np.nan
    v1, v2 = p1 - p2, p3 - p2
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return np.nan
    angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm_product, -1.0, 1.0)))
    return angle / 180.0


def calculate_distance(p1, p2) -> float:
    """Calculate normalized distance"""
    if np.isnan(p1).any() or np.isnan(p2).any():
        return np.nan
    return np.linalg.norm(p1 - p2) / np.sqrt(2.0)


def extract_engineered_features(player_pose_kpts: np.ndarray, 
                               shuttle_pos: np.ndarray) -> np.ndarray:
    """
    Extract biomechanical features for shot classification
    Returns: [elbow_angle, posture_proxy, wrist_shuttle_dist]
    """
    RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 6, 8, 10
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 5, 7, 9
    RIGHT_ANKLE, LEFT_ANKLE = 16, 15
    
    p_right_wrist = player_pose_kpts[RIGHT_WRIST]
    p_left_wrist = player_pose_kpts[LEFT_WRIST]
    
    dist_right = calculate_distance(p_right_wrist, shuttle_pos)
    dist_left = calculate_distance(p_left_wrist, shuttle_pos)
    
    # Determine active hand
    if not np.isnan(dist_right) and (np.isnan(dist_left) or dist_right < dist_left):
        indices = (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, LEFT_ANKLE)
    elif not np.isnan(dist_left):
        indices = (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, RIGHT_ANKLE)
    else:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    
    p_shoulder = player_pose_kpts[indices[0]]
    p_elbow = player_pose_kpts[indices[1]]
    p_wrist = player_pose_kpts[indices[2]]
    p_ankle = player_pose_kpts[indices[3]]
    
    elbow_angle = calculate_angle(p_shoulder, p_elbow, p_wrist)
    posture_proxy = calculate_distance(p_shoulder, p_ankle)
    valid_dists = [d for d in [dist_right, dist_left] if not np.isnan(d)]
    wrist_shuttle_dist = min(valid_dists) if valid_dists else np.nan
    
    return np.array([elbow_angle, posture_proxy, wrist_shuttle_dist], dtype=np.float32)

# In main_coach.py, near the other helper functions

def get_player_centroid(pose_data_frame: np.ndarray, player: str) -> np.ndarray:
    """Helper to get a player's torso centroid from a single frame of pose data."""
    if player == 'top':
        kpts = pose_data_frame[:34].reshape(17, 2)
    else: # bottom
        kpts = pose_data_frame[36:70].reshape(17, 2)
    
    # Use torso keypoints for stable center (shoulders and hips)
    torso_indices = [5, 6, 11, 12]
    valid_points = kpts[torso_indices]
    
    # Filter out points at (0,0) which indicate no detection
    valid_points = valid_points[np.all(valid_points > 0, axis=1)]
    
    if len(valid_points) > 0:
        return np.mean(valid_points, axis=0)
    return np.array([np.nan, np.nan])


def check_player_stillness(pose_data: np.ndarray, hit_frame: int, player_who_hit: str, 
                           frame_w: int, frame_h: int, check_window_frames: int = 15, 
                           stillness_threshold_px: int = 5) -> bool: # <-- Increased default threshold
    """
    MODIFIED: Checks if players were relatively still before a hit. Handles missing pose data gracefully.
    Returns True if players are still, False otherwise.
    """
    if hit_frame < check_window_frames:
        return True # Not enough history, assume it could be a serve

    start_frame = hit_frame - check_window_frames
    
    # Get centroids for the check window
    top_centroids, bottom_centroids = [], []
    for i in range(start_frame, hit_frame):
        frame_pose = pose_data[i]
        top_centroids.append(get_player_centroid(frame_pose, 'top'))
        bottom_centroids.append(get_player_centroid(frame_pose, 'bottom'))

    # Remove NaN values from lists
    top_centroids = np.array([c for c in top_centroids if not np.isnan(c).any()])
    bottom_centroids = np.array([c for c in bottom_centroids if not np.isnan(c).any()])

    # Convert to pixel coordinates
    if len(top_centroids) > 0: top_centroids *= np.array([frame_w, frame_h])
    if len(bottom_centroids) > 0: bottom_centroids *= np.array([frame_w, frame_h])
    
    # Calculate displacement ONLY if we have enough data points (prevents 'nan' errors)
    top_max_displacement = np.max(np.ptp(top_centroids, axis=0)) if len(top_centroids) > 1 else 0.0
    bottom_max_displacement = np.max(np.ptp(bottom_centroids, axis=0)) if len(bottom_centroids) > 1 else 0.0

    # NEW LOGIC: A player is considered "still" if we have no data OR their movement is below the threshold.
    # This correctly handles the case where one player is not detected.
    top_is_still = len(top_centroids) < 2 or top_max_displacement < stillness_threshold_px
    bottom_is_still = len(bottom_centroids) < 2 or bottom_max_displacement < stillness_threshold_px
    
    # More informative debug log
    logging.info(f"[Stillness Check @ Frame {hit_frame}] Top-Disp: {top_max_displacement:.2f}px (IsStill: {top_is_still}), "
                 f"Bottom-Disp: {bottom_max_displacement:.2f}px (IsStill: {bottom_is_still}) "
                 f"(Threshold: {stillness_threshold_px}px)")

    # For a serve, BOTH players (for whom we have data) must be in a still state.
    return top_is_still and bottom_is_still

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_metadata(video_path: Path, metadata_dir: Path) -> Optional[Dict]:
    """Load video metadata (court polygon, dimensions, fps)"""
    metadata_path = metadata_dir / (video_path.stem + ".json")
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load metadata: {e}")
    return None


def ensure_data_exists(video_path: Path, pose_dir: Path, shuttle_dir: Path) -> Tuple[Path, Path]:
    """
    Ensure pose and shuttle data exist, generate if missing
    Returns: (pose_path, shuttle_path)
    """
    pose_path = pose_dir / "NewVideos" / f"{video_path.stem}.npy"
    shuttle_path = shuttle_dir / f"{video_path.stem}_ball.csv"
    
    # Generate pose data if missing
    if not pose_path.exists():
        logging.info("📊 Pose data missing, running preprocessing...")
        output_pose_dir = pose_dir / "NewVideos"
        output_pose_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            subprocess.run(
                [sys.executable, "preprocess_data_v5.py", 
                 "--video", str(video_path), 
                 "--output_dir", str(output_pose_dir)],
                check=True,
                capture_output=True
            )
            logging.info("✅ Pose data generated successfully")
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Pose preprocessing failed: {e.stderr.decode()}")
            raise
    
    # Generate shuttle data if missing
    if not shuttle_path.exists():
        logging.info("🏸 Shuttle data missing, running TrackNet prediction...")
        shuttle_dir.mkdir(parents=True, exist_ok=True)
        
        # ----- NEW CODE (CORRECTED) -----
        # ----- NEW CODE (CORRECTED) -----
        try:
            # Define the path to your TrackNet model
            tracknet_model_path = r"TrackNetV3\ckpts\TrackNet_best.pt" # IMPORTANT: Verify this path!

            subprocess.run(
                [sys.executable, r"TrackNetV3\predict.py",
                "--video_file", str(video_path),
                "--save_dir", str(shuttle_dir),
                "--tracknet_file", tracknet_model_path], # <-- This line was added
                check=True,
                capture_output=True
            )
            logging.info("✅ Shuttle data generated successfully")
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Shuttle tracking failed: {e.stderr.decode()}")
            raise
    return pose_path, shuttle_path


def load_model_and_classes(run_dir: Path, model_params: Dict, 
                          device: torch.device) -> Tuple:
    """Load trained model, class mappings, and normalization parameters"""
    model_path = run_dir / "best_model.pth"
    class_map_path = run_dir / "class_map.json"
    norm_params_path = run_dir / "normalization_params.npz"
    
    # Validate all files exist
    if not all(p.exists() for p in [model_path, class_map_path, norm_params_path]):
        logging.error(f"❌ Model artifacts missing in {run_dir}")
        return None, None, None, None
    
    # Load class mapping
    with open(class_map_path, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Load normalization parameters
    norm_data = np.load(norm_params_path)
    mean = torch.tensor(norm_data['mean'], dtype=torch.float32, device=device)
    std = torch.tensor(norm_data['std'], dtype=torch.float32, device=device)
    
    # Initialize model
    model = TransformerClassifier(
        input_size=model_params['input_size'],
        d_model=model_params['d_model'],
        nhead=model_params['nhead'],
        num_encoder_layers=model_params['num_encoder_layers'],
        dim_feedforward=model_params['dim_feedforward'],
        num_classes=len(class_to_idx),
        dropout=model_params['dropout']
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    logging.info("✅ Model loaded successfully")
    return model, idx_to_class, mean, std


def create_prediction_window(full_feature_set: np.ndarray, 
                            center_frame_idx: int, 
                            sequence_length: int) -> Optional[np.ndarray]:
    """Create padded sequence window for prediction"""
    num_total_frames = len(full_feature_set)
    start_frame = center_frame_idx - sequence_length // 2
    end_frame = start_frame + sequence_length
    
    window = full_feature_set[max(0, start_frame):min(num_total_frames, end_frame)]
    
    # Pad if necessary
    pad_before = max(0, -start_frame)
    pad_after = max(0, end_frame - num_total_frames)
    
    if pad_before > 0:
        window = np.vstack([np.repeat(window[0:1], pad_before, axis=0), window])
    if pad_after > 0:
        window = np.vstack([window, np.repeat(window[-1:], pad_after, axis=0)])
    
    return window.astype(np.float32) if window.shape[0] == sequence_length else None


# In main_coach.py

def predict_shot_type(model: nn.Module, window: np.ndarray, 
                     idx_to_class: Dict, mean: torch.Tensor, 
                     std: torch.Tensor, device: torch.device) -> List[Tuple[str, float]]:
    """
    MODIFIED: Predict shot type from feature window, returning top 2 predictions.
    """
    if any(x is None for x in [model, window, mean, std]):
        return [("Unknown", 0.0), ("Unknown", 0.0)]
    
    window_tensor = torch.tensor(window, dtype=torch.float32, device=device)
    window_tensor = (window_tensor - mean) / std
    
    with torch.no_grad():
        outputs = model(window_tensor.unsqueeze(0))
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top 2 predictions
        top2_probs, top2_indices = torch.topk(probs, 2, dim=1)
        
        results = []
        for i in range(top2_indices.size(1)):
            idx = top2_indices[0, i].item()
            prob = top2_probs[0, i].item()
            results.append((idx_to_class[idx], prob))
            
    return results

# ============================================================================
# INTERACTIVE TUNING MODE
# ============================================================================

def calculate_impact_scores(shuttle_df: pd.DataFrame, fps: float, 
                            frame_w: int, frame_h: int, 
                            smoothing_window: int, polyorder: int):
    """Calculate impact scores for visualization"""
    from scipy.signal import savgol_filter
    import pandas as pd
    
    # Import the HitDetector class
    from hit_detect_v2 import HitDetector
    
    detector = HitDetector(
        fps=fps,
        smoothing_window=smoothing_window,
        polyorder=polyorder,
        frame_resolution=(frame_w, frame_h)
    )
    
    segments = detector._get_continuous_segments(shuttle_df[['Visibility', 'X', 'Y']])
    
    all_scores = pd.Series(dtype=np.float64)
    
    for segment in segments:
        if segment.empty or len(segment) <= smoothing_window:
            continue
            
        kinematics = detector._calculate_kinematics(segment.copy())
        accel_norm = detector._normalize_signal(kinematics['acceleration'])
        angle_norm = detector._normalize_signal(kinematics['angular_change_rad'])
        impact_score = (accel_norm * angle_norm).fillna(0)
        all_scores = pd.concat([all_scores, impact_score])
    
    return all_scores


def interactive_tuning_mode(video_path: Path, shuttle_df: pd.DataFrame, 
                           fps: float, frame_w: int, frame_h: int):
    """
    Interactive tuning interface with live parameter adjustment
    Shows impact score peaks for precise tuning
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button
        from matplotlib.patches import Rectangle
    except ImportError:
        logging.error("❌ matplotlib required for interactive mode: pip install matplotlib")
        return None
    
    logging.info("\n" + "="*70)
    logging.info("🎛️  INTERACTIVE TUNING MODE - IMPACT SCORE ANALYSIS")
    logging.info("="*70)
    logging.info("🔴 RED DOTS = Detected hits with current parameters")
    logging.info("📊 BLUE LINE = Impact score (peaks indicate potential hits)")
    logging.info("🎯 GOAL: Adjust sliders so red dots align with real hits")
    logging.info("❌ Close window when satisfied to continue rendering")
    logging.info("="*70 + "\n")
    
    # Create figure
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(6, 3, hspace=0.4, wspace=0.3)
    
    # Main impact score plot
    ax_main = fig.add_subplot(gs[0:3, :])
    ax_main.set_title('Impact Score Analysis (Acceleration × Angular Change)', 
                     fontsize=14, fontweight='bold')
    ax_main.set_xlabel('Frame Number', fontsize=11)
    ax_main.set_ylabel('Normalized Impact Score', fontsize=11)
    ax_main.grid(True, alpha=0.3)
    
    # Visibility indicator
    ax_vis = fig.add_subplot(gs[3, :], sharex=ax_main)
    ax_vis.set_title('Shuttlecock Visibility', fontsize=10)
    ax_vis.set_ylabel('Visible', fontsize=9)
    ax_vis.set_ylim(-0.1, 1.1)
    ax_vis.grid(True, alpha=0.3)
    ax_vis.fill_between(shuttle_df.index, 0, shuttle_df['Visibility'], 
                        alpha=0.3, color='green', label='Visible')
    ax_vis.legend(loc='upper right', fontsize=8)
    
    # Calculate initial impact scores
    impact_scores = calculate_impact_scores(
        shuttle_df, fps, frame_w, frame_h,
        Config.HIT_DETECTION['smoothing_window'],
        Config.HIT_DETECTION['polyorder']
    )
    
    if impact_scores.empty:
        logging.error("❌ No impact scores calculated - check shuttle data")
        plt.close()
        return None
    
    # Plot impact scores
    line_impact, = ax_main.plot(impact_scores.index, impact_scores.values, 
                                'b-', alpha=0.7, linewidth=1.5, label='Impact Score')
    
    # Store hit markers
    hit_scatter = ax_main.scatter([], [], c='red', s=100, zorder=5, 
                                 marker='o', label='Detected Hits', edgecolors='darkred')
    
    # Info text box
    info_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes, 
                            fontsize=11, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    ax_main.legend(loc='upper right', fontsize=10)
    
    # Create sliders - ALL tunable parameters
    slider_params = {}
    
    # Row 1: Peak detection
    ax_prom = fig.add_subplot(gs[4, 0])
    slider_params['prominence'] = Slider(
        ax_prom, 'Peak\nProminence', 0.01, 0.50, 
        valinit=Config.HIT_DETECTION['peak_prominence'], 
        valstep=0.01, color='steelblue'
    )
    
    ax_height = fig.add_subplot(gs[4, 1])
    slider_params['height'] = Slider(
        ax_height, 'Peak\nHeight', 0.01, 0.50, 
        valinit=Config.HIT_DETECTION['peak_height'], 
        valstep=0.01, color='steelblue'
    )
    
    ax_slope = fig.add_subplot(gs[4, 2])
    slider_params['slope'] = Slider(
        ax_slope, 'Slope\nThreshold', 1, 50, 
        valinit=Config.HIT_DETECTION['slope_thresh'], 
        valstep=1, color='mediumseagreen'
    )
    
    # Row 2: Additional parameters
    ax_sep = fig.add_subplot(gs[5, 0])
    slider_params['separation'] = Slider(
        ax_sep, 'Min Hit\nSeparation (s)', 0.1, 1.0, 
        valinit=Config.HIT_DETECTION['min_hit_separation_seconds'], 
        valstep=0.05, color='coral'
    )
    
    ax_conf = fig.add_subplot(gs[5, 1])
    slider_params['confidence'] = Slider(
        ax_conf, 'Min\nConfidence', 0.0, 1.0, 
        valinit=Config.CLASSIFICATION['min_confidence'], 
        valstep=0.05, color='orchid'
    )
    
    ax_reach = fig.add_subplot(gs[5, 2])
    slider_params['reach'] = Slider(
        ax_reach, 'Max Player\nReach (px)', 300, 1500, 
        valinit=Config.HIT_DETECTION['max_reach_threshold'], 
        valstep=50, color='goldenrod'
    )
    
    # Update function
    def update(val):
        # Update config
        Config.HIT_DETECTION['peak_prominence'] = slider_params['prominence'].val
        Config.HIT_DETECTION['peak_height'] = slider_params['height'].val
        Config.HIT_DETECTION['slope_thresh'] = int(slider_params['slope'].val)
        Config.HIT_DETECTION['min_hit_separation_seconds'] = slider_params['separation'].val
        Config.HIT_DETECTION['max_reach_threshold'] = int(slider_params['reach'].val)
        Config.CLASSIFICATION['min_confidence'] = slider_params['confidence'].val
        
        # Re-detect hits with new parameters
        try:
            hit_detection_params = {k: v for k, v in Config.HIT_DETECTION.items() if k != 'verbose'}
            hit_events = detect_and_attribute_hits(
                shuttle_df.copy(), fps=fps, frame_w=frame_w, frame_h=frame_h,
                frame_resolution=(frame_w, frame_h), 
                verbose=False,
                **hit_detection_params
            )
            
            # Extract hit frame numbers
            hit_frames = [hit['frame'] for hit in hit_events]
            
            # Get impact scores at hit frames
            valid_hits = [f for f in hit_frames if f in impact_scores.index]
            hit_scores = impact_scores.loc[valid_hits].values if valid_hits else []
            
            # Update scatter plot
            if len(valid_hits) > 0:
                hit_scatter.set_offsets(np.c_[valid_hits, hit_scores])
            else:
                hit_scatter.set_offsets(np.empty((0, 2)))
            
            # Update info text
            info_text.set_text(
                f'🎯 Detected Hits: {len(hit_events)}\n'
                f'Peak Prominence: {slider_params["prominence"].val:.2f}\n'
                f'Peak Height: {slider_params["height"].val:.2f}\n'
                f'Slope Threshold: {slider_params["slope"].val:.0f}\n'
                f'Min Separation: {slider_params["separation"].val:.2f}s\n'
                f'Min Confidence: {slider_params["confidence"].val:.2f}\n'
                f'Max Reach: {slider_params["reach"].val:.0f}px'
            )
            
        except Exception as e:
            info_text.set_text(f'❌ Error: {str(e)[:100]}...')
        
        fig.canvas.draw_idle()
    
    # Connect all sliders
    for slider in slider_params.values():
        slider.on_changed(update)
    
    # Initial update
    update(None)
    
    # Add comprehensive instructions
    instructions = (
        'TUNING GUIDE:\n'
        '• Peak Prominence/Height: ↓ Lower = More sensitive (more hits) | ↑ Higher = Less sensitive (fewer false positives)\n'
        '• Slope Threshold: Controls detection of sudden direction changes\n'
        '• Min Separation: Prevents detecting same hit multiple times\n'
        '• Min Confidence: Filters out low-confidence shot classifications\n'
        '• Max Reach: Maximum distance for hit attribution to player'
    )
    fig.text(0.5, 0.01, instructions, ha='center', fontsize=9, 
            style='italic', color='darkslategray',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()
    
    # Return final parameters
    final_params = {
        'peak_prominence': slider_params['prominence'].val,
        'peak_height': slider_params['height'].val,
        'slope_thresh': int(slider_params['slope'].val),
        'min_hit_separation_seconds': slider_params['separation'].val,
        'max_reach_threshold': int(slider_params['reach'].val),
        'min_confidence': slider_params['confidence'].val
    }
    
    logging.info("\n✅ Interactive tuning complete!")
    logging.info("📋 Final Parameters:")
    for key, value in final_params.items():
        logging.info(f"   {key}: {value}")
    logging.info("")
    
    return final_params


# ============================================================================
# MAIN VIDEO PROCESSING
# ============================================================================

def coach_video(args, device: torch.device):
    """Main video coaching pipeline"""
    video_path = Path(args.video)
    
    logging.info("="*70)
    logging.info(f"🎾 AI BADMINTON COACH - Processing: {video_path.name}")
    logging.info("="*70)
    
    # ========================================================================
    # 1. LOAD MODEL
    # ========================================================================
    model, idx_to_class, mean, std = load_model_and_classes(
        Path(args.run_dir), args.model_params, device
    )
    if model is None:
        logging.error("Failed to load model, aborting")
        return
    
    # ========================================================================
    # 2. ENSURE DATA EXISTS (auto-generate if needed)
    # ========================================================================
    try:
        pose_path, shuttle_path = ensure_data_exists(
            video_path, Path(args.pose_dir), Path(args.shuttle_dir)
        )
    except Exception as e:
        logging.error(f"Failed to prepare data: {e}")
        return
    
    # ========================================================================
    # 3. LOAD DATA
    # ========================================================================
    logging.info("📂 Loading pose and shuttle data...")
    pose_data = np.load(pose_path)
    shuttle_df = pd.read_csv(shuttle_path)
    metadata = load_metadata(video_path, Path(args.metadata_dir))
    
    court_polygon = None
    if metadata and 'court_polygon' in metadata:
        court_polygon = np.array(metadata['court_polygon'])
    
    # ========================================================================
    # 4. VIDEO PROPERTIES
    # ========================================================================
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    max_frames = min(len(pose_data), len(shuttle_df))
    pose_data = pose_data[:max_frames]
    shuttle_df = shuttle_df.iloc[:max_frames]
    
    logging.info(f"📹 Video: {frame_w}x{frame_h} @ {fps:.1f} fps, {max_frames} frames")
    
    # ========================================================================
    # 5. PRECOMPUTE FEATURES
    # ========================================================================
    logging.info("⚙️  Pre-computing features and detecting hits...")
    
    # Shuttle velocities
    shuttle_df['vel_X'] = shuttle_df['X'].diff().fillna(0) / (1 / fps)
    shuttle_df['vel_Y'] = shuttle_df['Y'].diff().fillna(0) / (1 / fps)
    
    # Normalized shuttle features
    shuttle_features_norm = shuttle_df[['X', 'Y']].to_numpy() / np.array([frame_w, frame_h])
    shuttle_vel_norm = shuttle_df[['vel_X', 'vel_Y']].to_numpy() / np.array([frame_w, frame_h]) / (1 / fps)
    
    # Engineered features
    all_engineered_features = []
    shuttle_positions_norm = shuttle_features_norm
    
    for i in range(max_frames):
        top_kpts_norm = pose_data[i, :34].reshape(17, 2)
        bot_kpts_norm = pose_data[i, 36:70].reshape(17, 2)
        
        top_eng = extract_engineered_features(top_kpts_norm, shuttle_positions_norm[i])
        bot_eng = extract_engineered_features(bot_kpts_norm, shuttle_positions_norm[i])
        all_engineered_features.append(np.concatenate([top_eng, bot_eng]))
    
    engineered_features = np.nan_to_num(np.array(all_engineered_features), nan=0.0)
    
    # Combine all features
    full_feature_set = np.concatenate([
        pose_data, engineered_features, shuttle_features_norm, shuttle_vel_norm
    ], axis=1)
    
    # ========================================================================
    # INTERACTIVE TUNING MODE (if enabled)
    # ========================================================================
    if args.interactive:
        tuned_params = interactive_tuning_mode(
            video_path, shuttle_df.copy(), fps, frame_w, frame_h
        )
        if tuned_params:
            # Update both hit detection and classification params
            Config.HIT_DETECTION['peak_prominence'] = tuned_params['peak_prominence']
            Config.HIT_DETECTION['peak_height'] = tuned_params['peak_height']
            Config.CLASSIFICATION['min_confidence'] = tuned_params['min_confidence']
    
    # Detect hits
    hit_detection_params = {k: v for k, v in Config.HIT_DETECTION.items() if k != 'verbose'}
    hit_events = detect_and_attribute_hits(
        shuttle_df, fps=fps, frame_w=frame_w, frame_h=frame_h, 
        frame_resolution=(frame_w, frame_h), 
        verbose=Config.HIT_DETECTION['verbose'],
        **hit_detection_params  # Pass all hit detection parameters except verbose
    )
    logging.info(f"🎯 Detected {len(hit_events)} hits")
    
    # ========================================================================
    # 6. ANALYZE SHOTS (Keep only highest confidence per hit)
    # ========================================================================
    logging.info("🧠 Analyzing shot types...")
    hits_by_frame = {}
    
    # NEW: Track last hit frame to detect serves
    last_classified_hit_frame = -1
    serve_check_pause_seconds = 3 # A rally can't start less than 2s after the last point
    serve_check_pause_frames = int(serve_check_pause_seconds * fps)

    for hit in tqdm(hit_events, desc="Classifying shots"):
        hit_frame = hit['frame']
        player_who_hit = hit['player'] # 'Top' or 'Bottom'
        
        if hit_frame >= max_frames:
            continue
        
        window = create_prediction_window(
            full_feature_set, hit_frame, args.model_params['sequence_length']
        )
        if window is None:
            continue
        
        # MODIFIED: Get top 2 predictions
        predictions = predict_shot_type(
            model, window, idx_to_class, mean, std, device
        )
        (shot_type, confidence), (fallback_shot, fallback_conf) = predictions

        # --- ADD THIS NEW DEBUG LINE ---
        logging.info(f"[Model Prediction @ Frame {hit_frame}] Top Guess: '{shot_type}' ({confidence:.2f}), Fallback: '{fallback_shot}' ({fallback_conf:.2f})")
        # --- END OF NEW DEBUG LINE ---

        # --- NEW SERVE VALIDATION LOGIC ---
        serve_types = ["serve_long", "serve_short", "serve"] # Add any serve variations here
        if (shot_type.lower() in serve_types and 
            last_classified_hit_frame != -1 and 
            (hit_frame - last_classified_hit_frame) < serve_check_pause_frames):
            
            logging.info(f"Frame {hit_frame}: Invalidating '{shot_type}' (mid-rally). Falling back to '{fallback_shot}'.")
            
            # Smart Fallback: If the fallback is also a serve, classify as 'Unknown'.
            if fallback_shot.lower() in serve_types:
                shot_type = "Rally_Shot" # Use a generic but descriptive name
                confidence = (confidence + fallback_conf) / 2
            else:
                shot_type, confidence = fallback_shot, fallback_conf
        # Filter by minimum confidence
        if confidence < Config.CLASSIFICATION['min_confidence']:
            continue
        
        # Keep only best prediction per frame
        if hit_frame not in hits_by_frame or confidence > hits_by_frame[hit_frame]['confidence']:
            hits_by_frame[hit_frame] = {
                'shot_type': shot_type,
                'confidence': confidence,
                'frame': hit_frame
            }
            # Update the last hit frame ONLY if it's a valid, classified shot
            last_classified_hit_frame = hit_frame

    logging.info(f"✅ Classified {len(hits_by_frame)} shots (filtered by confidence >= {Config.CLASSIFICATION['min_confidence']})")
    # ========================================================================
    # 7. VIDEO RENDERING
    # ========================================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / f"{video_path.stem}_coached.mp4"
    
    # Try multiple codecs for better compatibility
    codecs_to_try = [
        ('avc1', '.mp4'),  # H.264 (best compatibility)
        ('mp4v', '.mp4'),  # MPEG-4
        ('XVID', '.avi'),  # Xvid
    ]
    
    out = None
    for codec, ext in codecs_to_try:
        if ext != output_video_path.suffix:
            test_path = output_video_path.with_suffix(ext)
        else:
            test_path = output_video_path
            
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(str(test_path), fourcc, fps, (frame_w, frame_h))
        
        if out.isOpened():
            output_video_path = test_path
            logging.info(f"📹 Using codec: {codec}")
            break
        out.release()
        out = None
    
    if out is None or not out.isOpened():
        logging.error("❌ Failed to initialize video writer with any codec")
        cap.release()
        return
    
    # Animation state
    animation_state = {
        'hit_ttl': 0,
        'hit_pos': (0, 0),
        'prediction_ttl': 0,
        'prediction_text': "Analysis Ready",
        'confidence': 0.0
    }
    
    logging.info("🎬 Rendering coached video...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
    
    for frame_idx in tqdm(range(max_frames), desc="Rendering"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update animation state on hit
        if frame_idx in hits_by_frame:
            hit_info = hits_by_frame[frame_idx]
            shuttle_pos_at_hit = shuttle_df.loc[frame_idx, ['X', 'Y']].astype(int).values
            
            animation_state['hit_ttl'] = HIT_ANIMATION_FRAMES
            animation_state['hit_pos'] = tuple(shuttle_pos_at_hit)
            animation_state['prediction_ttl'] = PREDICTION_HOLD_FRAMES
            animation_state['prediction_text'] = f"{hit_info['shot_type']} ({hit_info['confidence']:.2f})"
            animation_state['confidence'] = hit_info['confidence']
        
        # Draw court
        draw_court(frame, court_polygon, COLORS['court'])
        
        # Draw player skeletons
        top_kpts_norm = pose_data[frame_idx, :34]
        bot_kpts_norm = pose_data[frame_idx, 36:70]
        draw_skeleton(frame, top_kpts_norm, (frame_w, frame_h), COLORS['player_top'])
        draw_skeleton(frame, bot_kpts_norm, (frame_w, frame_h), COLORS['player_bottom'])
        
        # Draw shuttlecock
        shuttle_pos = shuttle_df.loc[frame_idx, ['X', 'Y']].astype(int).values
        draw_shuttle(frame, tuple(shuttle_pos), COLORS['shuttle'])
        
        # Draw hit animation
        if animation_state['hit_ttl'] > 0:
            draw_hit_animation(
                frame, animation_state['hit_pos'], 
                animation_state['hit_ttl'], HIT_ANIMATION_FRAMES
            )
            animation_state['hit_ttl'] -= 1
        
        # Draw prediction panel
        if animation_state['prediction_ttl'] > 0:
            draw_prediction_panel(
                frame, animation_state['prediction_text'], 
                frame_w, animation_state['confidence']
            )
            animation_state['prediction_ttl'] -= 1
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    logging.info("="*70)
    logging.info(f"✅ SUCCESS! Coached video saved to:")
    logging.info(f"   {output_video_path}")
    logging.info("="*70)

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Badminton Coach - Production Ready Video Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_coach.py --video my_match.mp4
  python main_coach.py --video game.mp4 --verbose
  python main_coach.py --video rally.mp4 --run_dir custom_model/
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--video', type=str, required=True,
        help='Path to the badminton video to analyze'
    )
    
    # Model configuration
    parser.add_argument(
        '--run_dir', type=str, 
        default='training_runs/run_20251015_131418',
        help='Directory containing trained model artifacts'
    )
    
    # Data directories
    parser.add_argument(
        '--pose_dir', type=str,
        default='preprocessed_data_v5_two_player',
        help='Directory for pose data (NewVideos subfolder auto-created)'
    )
    
    parser.add_argument(
        '--shuttle_dir', type=str,
        default=r'TrackNetV3/prediction',
        help='Directory for shuttle tracking data'
    )
    
    parser.add_argument(
        '--metadata_dir', type=str,
        default='preprocessed_metadata',
        help='Directory containing video metadata (court info, dimensions)'
    )
    
    parser.add_argument(
        '--output_dir', type=str,
        default='output_videos',
        help='Directory to save the coached video'
    )
    
    # Processing options
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )
    
    # Interactive tuning mode
    parser.add_argument(
        '--interactive', action='store_true',
        help='Enable interactive tuning mode with real-time preview'
    )
    
    # Hit detection tuning
    parser.add_argument(
        '--peak-prominence', type=float,
        help=f'Peak prominence for hit detection (default: {Config.HIT_DETECTION["peak_prominence"]})'
    )
    parser.add_argument(
        '--peak-height', type=float,
        help=f'Peak height threshold (default: {Config.HIT_DETECTION["peak_height"]})'
    )
    parser.add_argument(
        '--min-confidence', type=float,
        help=f'Minimum classification confidence (default: {Config.CLASSIFICATION["min_confidence"]})'
    )
    parser.add_argument(
        '--show-config', action='store_true',
        help='Print current configuration and exit'
    )
    
    args = parser.parse_args()
    
    # Apply command-line overrides
    if args.peak_prominence is not None:
        Config.HIT_DETECTION['peak_prominence'] = args.peak_prominence
    if args.peak_height is not None:
        Config.HIT_DETECTION['peak_height'] = args.peak_height
    if args.min_confidence is not None:
        Config.CLASSIFICATION['min_confidence'] = args.min_confidence
    
    args = parser.parse_args()
    
    # Model parameters (must match training configuration)
    args.model_params = {
        'input_size': 82,           # 72 (pose) + 6 (engineered) + 4 (shuttle)
        'sequence_length': 16,
        'd_model': 128,
        'nhead': 4,
        'num_encoder_layers': 3,
        'dim_feedforward': 256,
        'dropout': 0.4
    }
    
    return args

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Show config and exit if requested
    if args.show_config:
        print_config()
        sys.exit(0)
    
    # Validate video exists
    if not Path(args.video).exists():
        logging.error(f"❌ Video file not found: {args.video}")
        sys.exit(1)
    
    # Print configuration
    if args.verbose:
        print_config()
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🖥️  Using device: {device}")
    
    if torch.cuda.is_available():
        logging.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Run coaching pipeline
    try:
        coach_video(args, device)
    except KeyboardInterrupt:
        logging.warning("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Error during processing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()