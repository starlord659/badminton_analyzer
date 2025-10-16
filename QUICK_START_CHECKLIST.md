# Quick Start Checklist

Use this checklist to ensure you run everything in the correct order.

---

## Prerequisites

- [ ] Python 3.8+
- [ ] CUDA-capable GPU (recommended)
- [ ] Required packages installed:
  ```bash
  pip install torch torchvision ultralytics opencv-python pandas numpy scipy scikit-learn matplotlib seaborn tqdm
  ```
- [ ] Raw videos organized by class in `data/`
- [ ] TrackNet shuttle predictions in `TrackNetV3/prediction/`
- [ ] Court detection model in `models/best_model.pth`
- [ ] YOLO pose model in `models/yolo11s-pose.pt`

---

## Phase 1: Validation & Testing (CRITICAL - Don't Skip!)

### ✅ Step 1.1: Test Court Detection

```bash
# Test on 3-5 sample videos
python test_court_detection.py --dir data/Clear/ --batch 3
```

**Expected Result:**
- Success rate > 80%
- Corner labels match actual court corners
- Court area is 10-90% of frame

**If failed:**
- Check court_detection_tests/ folder for visualizations
- Adjust corner indices in court_detect.py if needed
- Verify videos have clear court visibility

---

### ✅ Step 1.2: Tune Hit Detection (Optional but Recommended)

```bash
# Tune on a sample video
python tune_hit_detection.py --csv TrackNetV3/prediction/sample_video_ball.csv

# OR tune on multiple videos for robust parameters
python tune_hit_detection.py --batch --num 5
```

**Expected Result:**
- Interactive plot shows clear peaks at hit moments
- Parameters saved to `hit_detection_params.json`

**Action:**
- Copy recommended parameters to preprocessing script

---

## Phase 2: Data Preprocessing

### ✅ Step 2.1: Extract Pose Features

```bash
# Start with 1 worker for debugging
python preprocess_data_v5_fixed.py --workers 1

# Once validated, use multiple workers
python preprocess_data_v5_fixed.py --workers 4
```

**Expected Result:**
```
✓ Successful: 7200+ videos (>90%)
⊘ Skipped: <10%
✗ Failed: <5%
```

**Output Folders:**
- `preprocessed_data_v5_two_player/` - Pose data (.npy files)
- `preprocessed_metadata/` - Video metadata (.json files)

---

### ✅ Step 2.2: Create Training Windows

```bash
python preprocess_multimodal_v3_fixed.py
```

**Expected Result:**
```
Total windows generated: 8000-10000
Files skipped: <5%
```

**Output Folder:**
- `preprocessed_multimodal_v5_engineered/` - Training windows (16, 82)

**Check:**
- Each class folder contains .npy files
- File sizes are reasonable (~5-10 KB each)

---

### ✅ Step 2.3: Validate Preprocessing Quality

```bash
python validate_preprocessing.py
```

**Critical Checks:**
- [ ] ✅ NO ERRORS FOUND
- [ ] Average NaN ratio < 5%
- [ ] Invalid engineered features < 30%
- [ ] All classes have samples
- [ ] Value ranges look reasonable

**Output:**
- `validation_results/preprocessing_validation.png`
- `validation_results/quality_heatmap.png`
- `validation_results/validation_results.json`

**Action:**
- Review visualizations
- If errors found, fix and re-run preprocessing
- **DO NOT proceed to training if validation fails**

---

## Phase 3: Model Training

### ✅ Step 3.1: Train Transformer Model

```bash
python train_transformer_classifier_v7.py
```

**Monitoring:**
- Watch validation accuracy (should reach >75% by epoch 50)
- Check F1 scores (target: >0.70 macro)
- Monitor for overfitting (train loss << val loss)

**Expected Duration:**
- GPU: 4-6 hours
- CPU: 20-30 hours

**Output:**
```
models/
├── transformer_multimodal_classifier_v7.pth (15MB)
├── transformer_multimodal_classifier_v7_quantized.pth (4MB)
├── class_map_v7.json
└── normalization_params_v7.npz

training_results/
├── confusion_matrix_final.png
└── training_history.png
```

---

### ✅ Step 3.2: Analyze Training Results

```bash
# Check training curves
open training_results/training_history.png

# Check confusion matrix
open training_results/confusion_matrix_final.png
```

**What to Look For:**

✅ **Good Signs:**
- Smooth loss curves
- Val accuracy > 75%
- F1 macro > 0.70
- No extreme confusion between classes

❌ **Warning Signs:**
- Accuracy < 60% → Data quality or model capacity issue
- Large train/val gap → Overfitting
- One class predicted always → Severe imbalance

---

## Phase 4: Model Deployment

### ✅ Step 4.1: Update Inference Script

Edit `main_coach.py` to use new model paths:

```python
MODEL_PATH = "models/transformer_multimodal_classifier_v7.pth"
CLASS_MAP_PATH = "models/class_map_v7.json"
NORM_PARAMS_PATH = "models/normalization_params_v7.npz"
```

---

### ✅ Step 4.2: Test Inference

```bash
python main_coach.py --video data/Clear/test_video.mp4
```

**Expected Output:**
- Coached video in `output_videos/`
- Shot classifications displayed
- Form corrections shown

---

## Phase 5: Production Deployment

### ✅ For Mobile Deployment (Use Quantized Model)

```python
model = torch.load('models/transformer_multimodal_classifier_v7_quantized.pth')
# 4x smaller, suitable for mobile devices
```

### ✅ For Server Deployment (Use Full Model)

```python
model = torch.load('models/transformer_multimodal_classifier_v7.pth')
# Better accuracy for server-side processing
```

---

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| Court detection fails | Run `test_court_detection.py`, adjust corner indices |
| Memory error in training | Reduce BATCH_SIZE to 32 or 16 |
| Low accuracy (<60%) | Check validation results, consider data augmentation |
| High invalid features | Verify shuttle CSV quality, check pose detection |
| Validation errors | Re-run preprocessing with fixed parameters |
| Slow preprocessing | Increase `--workers` if CPU allows |

---

## Performance Benchmarks (Your Dataset)

| Metric | Expected Value |
|--------|----------------|
| **Preprocessing** |
| Success rate | >90% |
| Time (4 workers) | 2-3 hours |
| Disk space | ~1.5GB |
| **Training** |
| Best accuracy | 80-85% |
| F1 macro | 0.75-0.80 |
| Training time (GPU) | 4-6 hours |
| **Inference** |
| Speed (GPU) | ~5ms per hit |
| Speed (CPU) | ~20ms per hit |
| Model size (full) | 15MB |
| Model size (quantized) | 4MB |

---

## File Structure Reference

```
badminton_analyzer/
├── 📁 data/                       # Input: Raw videos by class
├── 📁 TrackNetV3/prediction/      # Input: Shuttle tracking CSVs
├── 📁 models/                     # Input/Output: All models
├── 📁 preprocessed_data_v5_two_player/     # Output: Pose features
├── 📁 preprocessed_metadata/               # Output: Video metadata
├── 📁 preprocessed_multimodal_v5_engineered/ # Output: Training windows
├── 📁 training_results/           # Output: Training visualizations
├── 📁 validation_results/         # Output: Validation reports
├── 📁 court_detection_tests/      # Output: Court detection tests
├── 📁 output_videos/              # Output: Coached videos
│
├── 🔧 test_court_detection.py     # Step 1.1
├── 🔧 tune_hit_detection.py       # Step 1.2
├── 🔧 preprocess_data_v5_fixed.py # Step 2.1
├── 🔧 preprocess_multimodal_v3_fixed.py # Step 2.2
├── 🔧 validate_preprocessing.py   # Step 2.3
├── 🔧 train_transformer_classifier_v7.py # Step 3.1
├── 🔧 main_coach.py               # Step 4.2
│
├── 📚 hit_detect_fixed.py         # Library
├── 📚 court_detect.py             # Library
│
├── 📖 IMPLEMENTATION_GUIDE.md     # Detailed guide
└── 📖 QUICK_START_CHECKLIST.md    # This file
```

---

## Ready to Start?

Follow the checklist above in order. Each step validates the previous one, so **don't skip steps!**

### Estimated Total Time:
- **Testing & Validation:** 1-2 hours
- **Preprocessing:** 2-3 hours
- **Training:** 4-6 hours (GPU)
- **Total:** 7-11 hours

Good luck! 🏸

---

## After Training - Next Steps

1. **Evaluate on test set** - Use videos not in training data
2. **Collect failure cases** - Videos where classification fails
3. **Iterate on features** - Add racket angle, player velocity, etc.
4. **Optimize for production** - API deployment, caching, batch processing
5. **Monitor in production** - Track accuracy drift, collect new data