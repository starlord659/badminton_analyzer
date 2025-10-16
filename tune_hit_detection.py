"""
Interactive hit detection tuning tool
Use this to find optimal parameters for your specific videos
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from hit_detect_v2 import plot_impact_scores

def tune_hit_detection(shuttle_csv_path, fps=30):
    """
    Load shuttle data and launch interactive tuning interface
    """
    print("="*70)
    print("INTERACTIVE HIT DETECTION TUNING")
    print("="*70)
    
    # Load shuttle data
    if not Path(shuttle_csv_path).exists():
        print(f"❌ ERROR: Shuttle CSV not found: {shuttle_csv_path}")
        return None
    
    df = pd.read_csv(shuttle_csv_path)
    
    # Validate required columns
    required_cols = ['Visibility', 'X', 'Y']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ ERROR: Missing columns in CSV: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    print(f"\n✓ Loaded shuttle data: {len(df)} frames")
    print(f"  Visible frames: {(df['Visibility'] == 1).sum()}")
    print(f"  Hidden frames: {(df['Visibility'] == 0).sum()}")
    
    # Detect resolution from X, Y ranges
    max_x = df[df['Visibility'] == 1]['X'].max()
    max_y = df[df['Visibility'] == 1]['Y'].max()
    
    # Guess resolution
    if max_x > 1280 or max_y > 720:
        frame_resolution = (1920, 1080)
    else:
        frame_resolution = (1280, 720)
    
    print(f"  Detected resolution: {frame_resolution[0]}x{frame_resolution[1]}")
    
    # Initial parameters (conservative defaults)
    # NOTE: frame_resolution will be added separately to avoid duplication
    initial_params = {
        'peak_prominence': 0.15,
        'peak_height': 0.1,
        'slope_thresh': 10,
        'min_hit_separation_seconds': 0.25
    }
    
    print(f"\nInitial parameters:")
    for key, value in initial_params.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("Starting interactive tuning...")
    print("Adjust the sliders to match expected hit moments")
    print("Close the plot window when satisfied with the parameters")
    print("="*70 + "\n")
    
    # Launch interactive plot with frame_resolution passed separately
    final_params = plot_impact_scores(df, fps, initial_params, frame_resolution=frame_resolution)
    
    return final_params


def save_parameters(params, output_path='hit_detection_params.json'):
    """Save tuned parameters to JSON file"""
    import json
    
    # Remove frame_resolution (not JSON serializable as tuple)
    params_to_save = {k: v for k, v in params.items() if k != 'frame_resolution'}
    
    with open(output_path, 'w') as f:
        json.dump(params_to_save, f, indent=2)
    
    print(f"\n✓ Parameters saved to: {output_path}")
    print("You can load these in your preprocessing script:")
    print(f"""
import json
with open('{output_path}', 'r') as f:
    hit_params = json.load(f)

hit_events = detect_and_attribute_hits(
    df, 
    fps=30,
    frame_w=frame_w,
    frame_h=frame_h,
    **hit_params
)
""")


def batch_tune_videos(video_dir, num_videos=3):
    """
    Tune parameters on multiple videos to find robust settings
    """
    from pathlib import Path
    
    shuttle_dir = Path(r'D:\capstone\badminton_analyzer\TrackNetV3\prediction')
    
    if not shuttle_dir.exists():
        print(f"❌ ERROR: Shuttle prediction directory not found: {shuttle_dir}")
        return
    
    # Find shuttle CSV files
    csv_files = list(shuttle_dir.glob('*_ball.csv'))
    
    if not csv_files:
        print(f"❌ ERROR: No shuttle CSV files found in: {shuttle_dir}")
        return
    
    print(f"\nFound {len(csv_files)} shuttle CSV files")
    print(f"Will tune on {min(num_videos, len(csv_files))} videos\n")
    
    all_params = []
    
    for i, csv_path in enumerate(csv_files[:num_videos], 1):
        print(f"\n{'='*70}")
        print(f"TUNING VIDEO {i}/{min(num_videos, len(csv_files))}: {csv_path.name}")
        print(f"{'='*70}")
        
        params = tune_hit_detection(csv_path)
        
        if params:
            # Remove frame_resolution before storing (it's video-specific)
            params_without_resolution = {k: v for k, v in params.items() if k != 'frame_resolution'}
            all_params.append(params_without_resolution)
    
    if not all_params:
        print("\n❌ No successful tuning runs")
        return
    
    # Calculate average parameters
    print(f"\n{'='*70}")
    print("RECOMMENDED PARAMETERS (averaged across videos)")
    print(f"{'='*70}")
    
    avg_params = {}
    for key in ['peak_prominence', 'peak_height', 'slope_thresh', 'min_hit_separation_seconds']:
        values = [p[key] for p in all_params if key in p]
        if values:
            avg_params[key] = np.mean(values)
            std = np.std(values)
            print(f"  {key}: {avg_params[key]:.4f} (±{std:.4f})")
    
    # Save recommended parameters
    save_parameters(avg_params, 'hit_detection_params_recommended.json')


def quick_test(shuttle_csv_path, params_json_path):
    """
    Quickly test parameters from a JSON file
    """
    import json
    
    print("="*70)
    print("QUICK HIT DETECTION TEST")
    print("="*70)
    
    # Load parameters
    with open(params_json_path, 'r') as f:
        params = json.load(f)
    
    print(f"\nLoaded parameters from: {params_json_path}")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Load shuttle data
    df = pd.read_csv(shuttle_csv_path)
    
    # Detect resolution
    max_x = df[df['Visibility'] == 1]['X'].max()
    max_y = df[df['Visibility'] == 1]['Y'].max()
    frame_resolution = (1920, 1080) if max_x > 1280 or max_y > 720 else (1280, 720)
    
    params['frame_resolution'] = frame_resolution
    
    # Run detection
    from hit_detect_v2 import HitDetector
    detector = HitDetector(fps=30, **params)
    hit_frames = detector.detect_from_df(df[['Visibility', 'X', 'Y']])
    
    print(f"\n✓ Detected {len(hit_frames)} hits")
    print(f"  Frames: {hit_frames[:20]}")
    if len(hit_frames) > 20:
        print(f"  ... and {len(hit_frames) - 20} more")
    
    # Basic statistics
    if len(hit_frames) > 1:
        intervals = np.diff(hit_frames)
        print(f"\nHit interval statistics:")
        print(f"  Mean: {np.mean(intervals):.1f} frames ({np.mean(intervals)/30:.2f} seconds)")
        print(f"  Min: {np.min(intervals)} frames")
        print(f"  Max: {np.max(intervals)} frames")


def main():
    parser = argparse.ArgumentParser(description="Interactive hit detection tuning")
    parser.add_argument('--csv', type=str, help='Path to shuttle CSV file')
    parser.add_argument('--fps', type=float, default=30, help='Video FPS')
    parser.add_argument('--batch', action='store_true', help='Tune on multiple videos')
    parser.add_argument('--test', type=str, help='Test parameters from JSON file')
    parser.add_argument('--num', type=int, default=3, help='Number of videos for batch tuning')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_tune_videos(None, args.num)
    elif args.test and args.csv:
        quick_test(args.csv, args.test)
    elif args.csv:
        params = tune_hit_detection(args.csv, args.fps)
        if params:
            save_parameters(params)
    else:
        print("Usage examples:")
        print("  # Tune on single video")
        print("  python tune_hit_detection.py --csv TrackNetV3/prediction/video1_ball.csv")
        print("")
        print("  # Tune on multiple videos to find robust parameters")
        print("  python tune_hit_detection.py --batch --num 5")
        print("")
        print("  # Test saved parameters")
        print("  python tune_hit_detection.py --csv video1_ball.csv --test hit_detection_params.json")


if __name__ == "__main__":
    main()