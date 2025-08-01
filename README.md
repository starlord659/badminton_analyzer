# Pose Classification and Coaching Analyzer

This project provides tools for analyzing human poses in videos, classifying actions, and offering coaching feedback based on predefined rules. It leverages the PoseC3D model for pose recognition and a custom classifier for action classification.

## Project Structure

- `coach_analyzer.py`: The main script for analyzing videos and providing coaching feedback.
- `create_rules_db.py`: Script to create or update the rules database for coaching.
- `debug_motion_analyzer.py`: Script for debugging the motion analysis part of the project.
- `my_video.mp4`: Example video file for analysis.
- `requirements.txt`: Lists the Python dependencies required for the project.
- `test.py`: Script for testing different components of the project.
- `train_posec3d.py`: Script for training the PoseC3D classifier.
- `.idx/`: Configuration files for the development environment (Firebase Studio).
    - `airules.md`: AI assistance rules.
    - `dev.nix`: Nix configuration for the development environment.
- `models/`: Directory containing trained models and configuration files.
    - `class_map.json`: Mapping of class indices to action names.
    - `form_rules.json`: Rules for coaching feedback.
    - `posec3d_classifier.pth`: Trained PoseC3D classifier model.
    - `shot_classifier.pth`: Trained shot classifier model.
    - Various YOLO pose and object detection models (`yolo11n-pose.pt`, `yolo11n.pt`, etc.).
- `old_code/`: Directory containing older versions or experimental code.

## Setup

1.  **Clone the repository:**
