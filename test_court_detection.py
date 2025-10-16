"""
Test script to visualize and validate court detection
Run this on a few sample videos to ensure court corners are correct
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from court_detect import CourtDetect


def visualize_court_detection(video_path, output_dir='court_detection_tests/'):
    """
    Test court detection on a video and save visualization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    video_name = Path(video_path).stem
    print(f"\n{'='*70}")
    print(f"Testing court detection on: {video_name}")
    print(f"{'='*70}")
    
    # Initialize detector
    detector = CourtDetect()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video: {video_path}")
        return False
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video resolution: {frame_width}x{frame_height}")
    
    # Test multiple frames
    test_frames = [0, 30, 60, 90, 150, 300]
    successful_detections = []
    
    for frame_idx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"  Frame {frame_idx}: Could not read")
            continue
        
        # Try to detect court
        if detector.detect_court(frame):
            successful_detections.append(frame_idx)
            
            # Visualize all keypoints
            vis_frame = detector.draw_detected_court(frame)
            
            # Get and validate corners
            corners = detector.get_court_corners()

            if corners is not None:
                corners = np.array(corners)
                # reorder to [top-left, top-right, bottom-right, bottom-left]
                s = corners.sum(axis=1)
                diff = np.diff(corners, axis=1).ravel()
                top_left = corners[np.argmin(s)]
                bottom_right = corners[np.argmax(s)]
                top_right = corners[np.argmin(diff)]
                bottom_left = corners[np.argmax(diff)]
                corners = np.array([top_left, top_right, bottom_right, bottom_left])

            
            if corners is not None:
                # Draw corner annotations
                corner_labels = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                
                for i, (corner, label, color) in enumerate(zip(corners, corner_labels, colors)):
                    x, y = int(corner[0]), int(corner[1])
                    cv2.circle(vis_frame, (x, y), 15, color, 3)
                    cv2.putText(vis_frame, label, (x + 20, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Draw court polygon
                corner_points = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_frame, [corner_points], isClosed=True, 
                            color=(0, 255, 255), thickness=3)
                
                # Calculate court area
                area = cv2.contourArea(corner_points)
                frame_area = frame_width * frame_height
                area_ratio = area / frame_area
                
                # Add info text
                info_text = [
                    f"Frame: {frame_idx}",
                    f"Resolution: {frame_width}x{frame_height}",
                    f"Court Area: {area_ratio*100:.1f}% of frame",
                    f"Corners: OK" if 0.1 < area_ratio < 0.9 else "Corners: SUSPICIOUS"
                ]
                
                y_offset = 30
                for text in info_text:
                    color = (0, 255, 0) if "OK" in text else (255, 255, 255)
                    if "SUSPICIOUS" in text:
                        color = (0, 0, 255)
                    cv2.putText(vis_frame, text, (10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    y_offset += 30
                
                # Save visualization
                output_path = output_dir / f"{video_name}_frame_{frame_idx}.jpg"
                cv2.imwrite(str(output_path), vis_frame)
                
                print(f"  ✓ Frame {frame_idx}: Court detected, area={area_ratio*100:.1f}%")
                
                if area_ratio < 0.1 or area_ratio > 0.9:
                    print(f"    ⚠️  WARNING: Court area seems unusual!")
            else:
                print(f"  ✗ Frame {frame_idx}: Court detected but corners invalid")
        else:
            print(f"  ✗ Frame {frame_idx}: No court detected")
    
    cap.release()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"DETECTION SUMMARY")
    print(f"{'='*70}")
    print(f"Tested frames: {len(test_frames)}")
    print(f"Successful detections: {len(successful_detections)}")
    print(f"Success rate: {len(successful_detections)/len(test_frames)*100:.1f}%")
    
    if successful_detections:
        print(f"\n✓ Visualizations saved to: {output_dir}")
        print(f"  Please review the images to verify corner placement")
        return True
    else:
        print(f"\n❌ No successful detections. Check:")
        print(f"  1. Is the court clearly visible?")
        print(f"  2. Is the camera angle suitable?")
        print(f"  3. Are the corner indices correct?")
        return False


def batch_test_videos(video_dir, num_videos=5):
    """
    Test court detection on multiple videos
    """
    video_dir = Path(video_dir)
    
    if not video_dir.exists():
        print(f"❌ ERROR: Directory not found: {video_dir}")
        return
    
    # Get all video files
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        video_files.extend(list(video_dir.rglob(ext)))
    
    if not video_files:
        print(f"❌ ERROR: No video files found in: {video_dir}")
        return
    
    print(f"\nFound {len(video_files)} videos")
    print(f"Testing {min(num_videos, len(video_files))} videos...\n")
    
    # Test a sample
    results = []
    for video_path in video_files[:num_videos]:
        success = visualize_court_detection(video_path)
        results.append({
            'video': video_path.name,
            'success': success
        })
    
    # Overall summary
    print(f"\n{'='*70}")
    print(f"BATCH TEST SUMMARY")
    print(f"{'='*70}")
    
    successful = sum(1 for r in results if r['success'])
    print(f"Videos tested: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    
    if successful < len(results):
        print(f"\n⚠️  Failed videos:")
        for r in results:
            if not r['success']:
                print(f"  - {r['video']}")


def interactive_test(video_path):
    """
    Interactive testing with live preview
    """
    detector = CourtDetect()
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video: {video_path}")
        return
    
    print("\n{'='*70}")
    print("INTERACTIVE COURT DETECTION TEST")
    print("='*70}")
    print("Controls:")
    print("  SPACE: Next frame")
    print("  'r': Reset to frame 0")
    print("  'q': Quit")
    print("='*70}\n")
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video reached")
            break
        
        # Detect court
        detected = detector.detect_court(frame)
        
        if detected:
            vis_frame = detector.draw_detected_court(frame)
            corners = detector.get_court_corners()
            
            if corners is not None:
                # Draw corners
                corner_points = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_frame, [corner_points], isClosed=True, 
                            color=(0, 255, 255), thickness=3)
                
                status = "✓ Court Detected"
                color = (0, 255, 0)
            else:
                vis_frame = frame.copy()
                status = "✗ Invalid Corners"
                color = (0, 0, 255)
        else:
            vis_frame = frame.copy()
            status = "✗ No Court Detected"
            color = (0, 0, 255)
        
        # Add status text
        cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, status, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show frame
        cv2.imshow('Court Detection Test', vis_frame)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            frame_idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif key == ord(' '):
            frame_idx += 1
        
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Test court detection on videos")
    parser.add_argument('--video', type=str, help='Path to single video file')
    parser.add_argument('--dir', type=str, help='Path to directory with videos')
    parser.add_argument('--batch', type=int, default=5, help='Number of videos to test in batch mode')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode with live preview')
    
    args = parser.parse_args()
    
    if args.interactive and args.video:
        interactive_test(args.video)
    elif args.video:
        visualize_court_detection(args.video)
    elif args.dir:
        batch_test_videos(args.dir, args.batch)
    else:
        print("Please provide either --video or --dir argument")
        print("\nExamples:")
        print("  python test_court_detection.py --video data/Clear/video1.mp4")
        print("  python test_court_detection.py --dir data/Clear/ --batch 10")
        print("  python test_court_detection.py --video data/Clear/video1.mp4 --interactive")


if __name__ == "__main__":
    main()