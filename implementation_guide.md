# Complete Implementation Guide

## Overview of Changes

This guide covers the complete fixed implementation with all critical issues resolved:

### Key Improvements
1. ✅ **Resolution-aware preprocessing** - No more hardcoded frame dimensions
2. ✅ **Temporal consistency** - Player tracking with IoU matching
3. ✅ **Memory-efficient training** - Incremental normalization calculation
4. ✅ **Adaptive hit detection** - Resolution-aware thresholds
5. ✅ **Model quantization** - On-device deployment ready
6. ✅ **Data augmentation** - Automatic for underrepresented classes
7. ✅ **Comprehensive validation** - Pre-training quality checks
8. ✅ **Better metrics** - F1 scores, confusion matrices, training curves

---

## Step-by-Step Implementation

### Step 1: Test Court Detection (CRITICAL)

Before preprocessing your entire dataset, verify court detection works correctly:

```bash
# Test on a single video
python test_court_detection.py --video data/Clear/sample_video.mp4

# Test multiple videos
python test_court_detection.py --dir data/Clear/ --batch 5

# Interactive testing
python test_court_detection.py --video data/Clear/sample_video.mp4 --interactive
```

**What to check:**
- Corner labels should match court corners (clockwise: top-left, top-right, bottom-right, bottom-left)
- Court area should be 10-90% of frame
- Yellow polygon should outline the court accurately

**If corners are wrong:**
1. Open `court_detect.py`
2. Modify line 73: `COURT_CORNER_INDICES = [0, 4, 21, 17]`
3. Run visualization again to verify

---

### Step 2: Run Fixed Preprocessing

Replace your old preprocessing script with the fixed version:

```bash
# Single worker (easier for debugging)
python preprocess_data_v5_fixed.py --workers 1

# Multi-worker for speed (once validated)
python preprocess_data_v5_fixed.py --workers 4
```

**New features:**
- Saves metadata (resolution, FPS) for each video
- Temporal consistency with player tracking
- Resolution-aware coordinate normalization
- Better progress reporting

**Expected output:**
```
preprocessed_data_v5_two_player/
├── Clear/
│   ├── video1.npy
│   ├── video2.npy
│   └── ...
├── Drive/
└── ...

preprocessed_metadata/
├── Clear/
│   ├── video1.json
│   ├── video2.json
│   └── ...
└── ...
```

---

### Step 3: Create Training Windows

This step combines pose + shuttle + engineered features:

```bash
python preprocess_multimodal_v3_fixed.py
```

**New features:**
- Uses actual frame dimensions from metadata (no more hardcoding!)
- Automatic data augmentation for underrepresented classes
- Validates engineered feature quality
- Reduced sequence length (16 frames instead of 32)

**Expected output:**
```
preprocessed_multimodal_v5_engineered/
├── Clear/
│   ├── video1_hit_1.npy  (shape: 16, 82)
│   ├── video1_hit_2.npy
│   └── ...
└── ...
```

---

### Step 4: Validate Preprocessing (CRITICAL)

**Run this BEFORE training:**

```bash
python validate_preprocessing.py
```

This script checks:
- ✅ All files have correct shape (16, 82)
- ✅ No NaN or Inf values
- ✅ Value ranges are reasonable
- ✅ Metadata consistency
- ✅ Class distribution
- ✅ Feature quality by class

**Expected output:**
```
✅ NO ERRORS FOUND
⚠️  WARNINGS: 15
📊 DATASET STATISTICS
  Total samples: 8230
  Class distribution:
    Drive: 1432 samples
    Clear: 1252 samples
    ...
  Data quality:
    Average NaN ratio: 2.34%
    Average invalid engineered features: 18.5%

✅ RECOMMENDATION: Dataset looks good, proceed with training
```

**Visualizations saved:**
- `validation_results/preprocessing_validation.png` - Overall statistics
- `validation_results/quality_heatmap.png` - Quality by class

---

### Step 5: Train the Model

Now you can train with confidence:

```bash
python train_transformer_classifier_v7.py
```

**New features:**
- Incremental normalization (no memory overflow)
- Class-weighted loss for imbalanced data
- F1 scores tracking
- Confusion matrices saved every 10 epochs
- Training history plots
- **Automatic quantization for on-device deployment**

**Expected training time:**
- ~8,000 samples, 200 epochs
- GPU: 4-6 hours
- CPU: 20-30 hours

**Outputs:**
```
models/
├── transformer_multimodal_classifier_v7.pth          (Full model ~15MB)
├── transformer_multimodal_classifier_v7_quantized.pth (Quantized ~4MB)
├── class_map_v7.json
└── normalization_params_v7.npz

training_results/
├── confusion_matrix_epoch_10.png
├── confusion_matrix_epoch_20.png
├── ...
├── confusion_matrix_final.png
└── training_history.png
```

---

### Step 6: Evaluate Results

After training, analyze the results:

1. **Check training history:**
   - Open `training_results/training_history.png`
   - Look for: smooth loss curves, no overfitting, plateauing accuracy

2. **Check confusion matrix:**
   - Open `training_results/confusion_matrix_final.png`
   - Identify confused classes (e.g., Clear vs Drop_Shot)

3. **Review classification report:**
   - Check console output for per-class metrics
   - Look for classes with low F1 scores

**Good signs:**
- Val accuracy > 75%
- F1 macro > 0.70
- No single class dominating confusion

**Bad signs:**
- Accuracy < 60% (underfitting or data quality issues)
- Large gap between train/val loss (overfitting)
- One class predicted for everything (severe imbalance)

---

### Step 7: Test Inference

Update your `main_coach.py` to use the new models:

```python
# Update these paths at the top of main_coach.py
MODEL_PATH = "models/transformer_multimodal_classifier_v7.pth"
CLASS_MAP_PATH = "models/class_map_v7.json"
NORM_PARAMS_PATH = "models/normalization_params_v7.npz"
```

Test on a video:

```bash
python main_coach.py --video data/Clear/test_video.mp4
```

---

## Troubleshooting

### Issue: "No court found" during preprocessing

**Solutions:**
1. Run `test_court_detection.py` to visualize detection
2. Check if court corners are correct
3. Adjust confidence threshold in `court_detect.py` line 16:
   ```python
   def __init__(self, model_path='models/best_model.pth', confidence_threshold=0.5):  # Lower from 0.7
   ```

### Issue: "Shape mismatch" in validation

**Cause:** Pose data has wrong dimensions

**Solutions:**
1. Check YOLO model is detecting 17 keypoints
2. Verify `FEATURES_PER_PLAYER = 36` is correct
3. Re-run preprocessing from scratch

### Issue: High invalid engineered features (>50%)

**Cause:** Poor pose detection or shuttle tracking

**Solutions:**
1. Check shuttle CSV files exist and are accurate
2. Verify pose keypoints are in correct format (normalized or pixels)
3. Review a few videos manually with visualization

### Issue: Low training accuracy (<60%)

**Possible causes:**
1. **Data quality** - Run validation script, check quality metrics
2. **Class confusion** - Some shots are too similar (e.g., Clear vs Drop)
3. **Sequence length** - Try increasing to 24 frames
4. **Model capacity** - Increase D_MODEL or NUM_ENCODER_LAYERS

**Try:**
```python
# In train_transformer_classifier_v7.py
D_MODEL = 256  # Increase from 128
NUM_ENCODER_LAYERS = 4  # Increase from 3
SEQUENCE_LENGTH = 24  # In preprocessing script
```

### Issue: Out of memory during training

**Solutions:**
1. Reduce batch size:
   ```python
   BATCH_SIZE = 32  # Reduce from 64
   ```
2. Use gradient accumulation:
   ```python
   accumulation_steps = 2
   # In training loop:
   loss = loss / accumulation_steps
   loss.backward()
   if (batch_idx + 1) % accumulation_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```

---

## Performance Optimization Tips

### For Faster Preprocessing:
1. Use SSD instead of HDD for data storage
2. Increase workers: `--workers 8` (don't exceed CPU cores)
3. Pre-filter corrupted videos

### For Faster Training:
1. Use mixed precision training (PyTorch AMP)
2. Increase batch size if GPU memory allows
3. Use `pin_memory=True` in DataLoader
4. Reduce validation frequency

### For On-Device Deployment:
1. Use the quantized model (4x smaller)
2. Consider ONNX export for mobile frameworks
3. Implement batch inference for multiple hits
4. Cache pose detection results

---

## Model Deployment

### Option 1: Python Inference (Easiest)

```python
import torch
import numpy as np

# Load quantized model
model = torch.load('models/transformer_multimodal_classifier_v7_quantized.pth')
model.eval()

# Load normalization params
norm_data = np.load('models/normalization_params_v7.npz')
mean, std = norm_data['mean'], norm_data['std']

# Inference
def predict(window):
    window_tensor = torch.tensor(window, dtype=torch.float32)
    window_tensor = (window_tensor - mean) / std
    
    with torch.no_grad():
        output = model(window_tensor.unsqueeze(0))
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    return predicted.item(), confidence.item()
```

### Option 2: ONNX Export (For Mobile)

```python
import torch
import torch.onnx

model = torch.load('models/transformer_multimodal_classifier_v7.pth')
model.eval()

dummy_input = torch.randn(1, 16, 82)
torch.onnx.export(
    model, 
    dummy_input, 
    "models/badminton_classifier.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)
```

---

## Expected Performance Benchmarks

Based on your dataset:

### Preprocessing:
- **Time:** ~2-3 hours for 8,000 videos (4 workers)
- **Success rate:** >90% (with valid court detection)
- **Disk space:** ~1.5GB for preprocessed data

### Training:
- **Best accuracy:** 80-85% (target)
- **F1 Macro:** 0.75-0.80
- **Training time:** 4-6 hours (GPU)
- **Model size:** 15MB (full), 4MB (quantized)

### Inference:
- **Speed:** ~5ms per hit (GPU), ~20ms (CPU)
- **Memory:** <50MB (quantized model)

---

## Next Steps After Training

1. **Error Analysis:**
   - Review confused shot types
   - Collect more data for weak classes
   - Consider merging similar classes (e.g., Serve_Long + Serve_Short)

2. **Feature Engineering:**
   - Add racket angle features
   - Include player velocity
   - Track shot trajectory

3. **Model Improvements:**
   - Try LSTM or GRU architectures
   - Implement attention visualization
   - Add temporal convolutions

4. **Deployment:**
   - Build REST API for inference
   - Create mobile app integration
   - Implement real-time processing

---

## File Structure Summary

```
badminton_analyzer/
├── data/                          # Raw videos (organized by class)
├── preprocessed_data_v5_two_player/  # Pose + court coords
├── preprocessed_metadata/            # Video metadata (NEW)
├── preprocessed_multimodal_v5_engineered/  # Training windows
├── models/                        # Trained models
├── training_results/              # Training visualizations (NEW)
├── validation_results/            # Preprocessing validation (NEW)
├── court_detection_tests/         # Court detection tests (NEW)
├── preprocess_data_v5_fixed.py       # Step 2 (NEW)
├── preprocess_multimodal_v3_fixed.py  # Step 3 (NEW)
├── train_transformer_classifier_v7.py # Step 5 (NEW)
├── validate_preprocessing.py         # Step 4 (NEW)
├── test_court_detection.py          # Step 1 (NEW)
├── hit_detect_fixed.py              # Fixed hit detection (NEW)
└── main_coach.py                    # Inference script
```

---

## Questions & Support

If you encounter issues:

1. Run validation script first
2. Check error messages carefully
3. Review visualization outputs
4. Verify file paths and permissions

Good luck with your training! 🏸