# YOLOv8 Cat Detection Training Guide

## Overview

This guide explains the complete training process for the YOLOv8 cat detection model, including all parameters needed before training and what to expect after training.

---

## Table of Contents

1. [Before Training - Prerequisites](#before-training---prerequisites)
2. [Training Parameters Explained](#training-parameters-explained)
3. [Training Process](#training-process)
4. [After Training - Results](#after-training---results)
5. [Model Evaluation](#model-evaluation)
6. [Tips and Best Practices](#tips-and-best-practices)

---

## Before Training - Prerequisites

### 1. Dataset Structure

Your dataset must be organized in the following structure:

```
data/
├── train/
│   ├── images/     # Training images
│   └── labels/     # Training labels (YOLO format)
├── val/
│   ├── images/     # Validation images
│   └── labels/     # Validation labels
└── test/
    ├── images/     # Test images (optional)
    └── labels/     # Test labels (optional)
```

### 2. Label Format

Labels must be in YOLO format (`.txt` files):

- One file per image with the same name
- Each line: `<class_id> <x_center> <y_center> <width> <height>`
- All values normalized to [0, 1]
- Example: `0 0.5 0.5 0.3 0.4`

### 3. Required Files

- **data.yaml**: Dataset configuration file
- **train_yolo.py**: Training script
- **requirements.txt**: Python dependencies

### 4. System Requirements

#### Minimum Requirements

- **CPU**: Modern multi-core processor
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB+ free space for dataset and outputs
- **Python**: 3.8 or higher

#### Recommended for GPU Training

- **GPU**: NVIDIA GPU with 6GB+ VRAM (GTX 1660 Ti or better)
- **CUDA**: 11.0 or higher
- **cuDNN**: Compatible with CUDA version
- **RAM**: 16GB+

### 5. Software Setup

```bash
# Install required packages
pip install -r requirements.txt

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Training Parameters Explained

### Core Parameters

#### 1. **Model Size** (`--model-size`)

Determines the YOLOv8 architecture variant:

| Size | Model   | Parameters | Speed    | Accuracy  | Use Case                           |
| ---- | ------- | ---------- | -------- | --------- | ---------------------------------- |
| `n`  | YOLOv8n | 3.2M       | Fastest  | Good      | Mobile/Edge devices, Quick testing |
| `s`  | YOLOv8s | 11.2M      | Fast     | Better    | Real-time applications             |
| `m`  | YOLOv8m | 25.9M      | Moderate | Great     | Balanced performance               |
| `l`  | YOLOv8l | 43.7M      | Slower   | Excellent | High accuracy needed               |
| `x`  | YOLOv8x | 68.2M      | Slowest  | Best      | Maximum accuracy                   |

**Default**: `n` (nano)
**Recommendation**: Start with `n` for testing, use `s` or `m` for production

#### 2. **Epochs** (`--epochs`)

Number of complete passes through the training dataset.

- **Range**: 50-300 epochs
- **Default**: 100
- **Guidelines**:
  - Small dataset (< 1000 images): 100-150 epochs
  - Medium dataset (1000-5000 images): 50-100 epochs
  - Large dataset (> 5000 images): 50-80 epochs
- **Early stopping**: Model automatically stops if no improvement

#### 3. **Image Size** (`--img-size`)

Input image dimensions (square).

- **Default**: 640
- **Options**: 320, 416, 512, 640, 768, 1024, 1280
- **Trade-offs**:
  - Smaller (320-416): Faster training, lower accuracy, good for small objects
  - Medium (512-640): Balanced performance
  - Larger (768-1280): Slower training, higher accuracy, better for small objects

#### 4. **Batch Size** (`--batch-size`)

Number of images processed together in one iteration.

- **Default**: 16
- **Guidelines by GPU Memory**:
  - 4GB VRAM: batch=4-8
  - 6GB VRAM: batch=8-16
  - 8GB VRAM: batch=16-32
  - 12GB+ VRAM: batch=32-64
- **CPU Training**: Use batch=4-8
- **Impact**: Larger batches = more stable training but higher memory usage

#### 5. **Device** (`--device`)

Hardware to use for training.

- **Options**:
  - `cpu`: Use CPU only
  - `0`: Use first GPU
  - `1`: Use second GPU
  - `0,1`: Use multiple GPUs
- **Default**: Auto-detect (GPU if available, else CPU)

#### 6. **Pretrained** (`--pretrained`)

Whether to use pre-trained COCO weights.

- **Default**: True
- **Benefits**:
  - Faster convergence
  - Better accuracy with small datasets
  - Transfer learning from general object detection
- **When to disable**: Training from scratch on unique domain

### Optimization Parameters

#### 7. **Optimizer** (`--optimizer`)

Algorithm for updating model weights.

| Optimizer | Description                 | Best For                            |
| --------- | --------------------------- | ----------------------------------- |
| `SGD`     | Stochastic Gradient Descent | Classical training, robust          |
| `Adam`    | Adaptive Moment Estimation  | Fast convergence, adaptive learning |
| `AdamW`   | Adam with weight decay      | Better generalization               |
| `auto`    | Automatic selection         | General use (recommended)           |

**Default**: `auto`

#### 8. **Learning Rate** (`--lr0`)

Initial learning rate for training.

- **Default**: 0.01
- **Range**: 0.001 - 0.1
- **Guidelines**:
  - Small dataset: 0.001 - 0.005
  - Medium dataset: 0.005 - 0.01
  - Large dataset: 0.01 - 0.02
- **Auto-adjusts**: Learning rate scheduler reduces it during training

### Data Augmentation Parameters

Built into the training script:

| Parameter     | Default | Description                             |
| ------------- | ------- | --------------------------------------- |
| `hsv_h`       | 0.015   | Hue variation (color change)            |
| `hsv_s`       | 0.7     | Saturation variation                    |
| `hsv_v`       | 0.4     | Value/brightness variation              |
| `degrees`     | 0.0     | Rotation range (±degrees)               |
| `translate`   | 0.1     | Translation range (% of image)          |
| `scale`       | 0.5     | Scaling range                           |
| `shear`       | 0.0     | Shear transformation                    |
| `perspective` | 0.0     | Perspective transformation              |
| `flipud`      | 0.0     | Vertical flip probability               |
| `fliplr`      | 0.5     | Horizontal flip probability (50%)       |
| `mosaic`      | 1.0     | Mosaic augmentation (combines 4 images) |
| `mixup`       | 0.0     | MixUp augmentation                      |

### Output Parameters

#### 9. **Project & Name** (`--project`, `--name`)

- **Project**: Directory for all training runs (default: `runs/train`)
- **Name**: Specific run name (default: `cat_detection`)
- **Output**: Results saved to `{project}/{name}/`

#### 10. **Save Period** (`--save-period`)

- **Default**: 10
- **Function**: Save checkpoint every N epochs
- **Purpose**: Resume training if interrupted

---

## Training Process

### Step 1: Prepare Your Data

```bash
# Verify dataset structure
ls data/train/images | wc -l  # Count training images
ls data/train/labels | wc -l  # Count training labels
ls data/val/images | wc -l    # Count validation images
ls data/val/labels | wc -l    # Count validation labels
```

### Step 2: Basic Training Command

```bash
cd cat-detection/training

# Basic training with default parameters
python train_yolo.py --data-dir ../data

# Recommended training command
python training/train_yolo.py \
    --data-dir ./data \
    --model-size s \
    --epochs 100 \
    --img-size 640 \
    --batch-size 16 \
    --device 0
```

### Step 3: Advanced Training Examples

#### Example 1: Quick Test Run

- Training Original Dataset:

test with 100 images first

```bash
python ./training/train_yolo.py \
    --data-dir ./data/original \
    --model-size s \
    --epochs 100 \
    --img-size 100\
    --batch-size 16 \
    --device mps \
    --name cat_original
```

- Training Enhanced Dataset:

```bash

python ./training/train_yolo.py \
    --data-dir ./data/enhanced \
    --model-size s \
    --epochs 100 \
    --img-size 640 \
    --batch-size 16 \
    --device mps \
    --name cat_enhanced

```

### Step 4: Monitor Training

During training, you'll see:

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/100     2.84G      1.234      0.876      1.123        123        640
  2/100     2.84G      1.156      0.812      1.089        123        640
  ...
```

**Key Metrics to Watch**:

- **box_loss**: Bounding box localization error (should decrease)
- **cls_loss**: Classification error (should decrease)
- **dfl_loss**: Distribution focal loss (should decrease)
- **GPU_mem**: GPU memory usage
- **Instances**: Number of objects in batch

---

## After Training - Results

### Output Directory Structure

```
runs/train/cat_detection/
├── weights/
│   ├── best.pt          # Best model (highest validation mAP)
│   ├── last.pt          # Last epoch model
│   ├── epoch10.pt       # Checkpoint at epoch 10
│   ├── epoch20.pt       # Checkpoint at epoch 20
│   └── ...
├── results.csv          # Training metrics per epoch
├── results.png          # Training curves visualization
├── confusion_matrix.png # Confusion matrix
├── F1_curve.png        # F1 score curve
├── PR_curve.png        # Precision-Recall curve
├── P_curve.png         # Precision curve
├── R_curve.png         # Recall curve
├── labels.jpg          # Training label distribution
├── labels_correlogram.jpg
├── train_batch0.jpg    # Training batch samples
├── train_batch1.jpg
├── train_batch2.jpg
├── val_batch0_pred.jpg # Validation predictions
├── val_batch0_labels.jpg # Validation ground truth
└── args.yaml           # Training configuration used
```

### Understanding Results

#### 1. **Model Weights**

- **best.pt**: Use this for deployment (best validation performance)
- **last.pt**: Final model state (may not be the best)
- **Checkpoint files**: Resume interrupted training

#### 2. **Training Curves** (results.png)

Shows 8 plots:

1. **Box Loss (train/val)**: Bounding box accuracy
2. **Classification Loss (train/val)**: Object classification accuracy
3. **DFL Loss (train/val)**: Distribution focal loss
4. **Precision**: Correct detections / All detections
5. **Recall**: Correct detections / All ground truth objects
6. **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
7. **mAP@0.5:0.95**: Mean Average Precision across IoU 0.5-0.95

**Good Training Signs**:

- Losses decrease steadily
- mAP increases over time
- Train and validation curves are close (no overfitting)
- Curves plateau near the end (convergence)

**Warning Signs**:

- Losses increase or oscillate wildly → Reduce learning rate
- Large gap between train/val metrics → Overfitting, need more data
- Metrics don't improve → Check data quality, increase model size

#### 3. **Performance Metrics**

| Metric           | Description                          | Good Value |
| ---------------- | ------------------------------------ | ---------- |
| **Precision**    | How many detections are correct      | > 0.80     |
| **Recall**       | How many objects are detected        | > 0.75     |
| **mAP@0.5**      | Overall accuracy (loose threshold)   | > 0.85     |
| **mAP@0.5:0.95** | Overall accuracy (strict thresholds) | > 0.60     |
| **F1 Score**     | Balance of precision and recall      | > 0.75     |

#### 4. **Confusion Matrix**

Shows model performance:

- **True Positives**: Correct cat detections
- **False Positives**: Incorrect cat detections
- **False Negatives**: Missed cats
- **True Negatives**: Correct background (N/A for detection)

---

## Model Evaluation

### Understanding Dataset Splits: Validation vs Test Set

#### What's the Difference?

**Validation Set (val/)**:

- Used **during training** to monitor model performance
- YOLOv8 automatically evaluates on validation set after each epoch
- Helps with early stopping and hyperparameter tuning
- Used to select the "best.pt" model (highest mAP on validation set)
- **Purpose**: Guide training decisions and prevent overfitting

**Test Set (test/)**:

- Used **only after training is complete** for final evaluation
- Should NEVER be used during training or model selection
- Provides unbiased estimate of model performance on unseen data
- **Purpose**: Report final model performance for research/production

#### Current Setup

Your dataset has three splits:

- **Training**: 1,500 images (used for learning)
- **Validation**: 300 images (used during training for monitoring)
- **Test**: 300 images (used only for final evaluation)

#### Critical Rules

✅ **DO**:

- Use validation set during training (automatic in YOLOv8)
- Use test set ONLY ONCE at the very end for final evaluation
- Report test set metrics in your final results/paper

❌ **DON'T**:

- Don't use test set during training
- Don't tune hyperparameters based on test set performance
- Don't repeatedly evaluate on test set (causes data leakage)

### Evaluation During Training

YOLOv8 **automatically evaluates on validation set** during training:

```bash
python train_yolo.py \
    --data-dir ../data \
    --model-size s \
    --epochs 100
```

The training process:

1. Trains on training set
2. After each epoch, evaluates on **validation set**
3. Saves "best.pt" based on **validation mAP**
4. Generates validation metrics and plots

You'll see output like:

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/100      4.2G      1.234      0.567      1.234         64        640: 100%|██
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        300        450      0.812      0.756      0.834     0.612
```

### Final Evaluation on Test Set

**ONLY after training is complete**, evaluate on the test set:

```bash
# Option 1: Using train_yolo.py
python train_yolo.py \
    --data-dir ../data \
    --model-path runs/train/cat_detection/weights/best.pt \
    --evaluate
```

**IMPORTANT**: The current `evaluate_model()` function evaluates on **validation set**, not test set!

### Proper Test Set Evaluation

To evaluate on the test set, use YOLOv8 directly:

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/train/cat_detection/weights/best.pt')

# Evaluate on TEST set (not validation)
metrics = model.val(
    data='dataset.yaml',
    split='test',  # Explicitly use test set
    imgsz=640
)

print(f"Test Set Results:")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
```

Or via command line:

```bash
yolo detect val \
    model=runs/train/cat_detection/weights/best.pt \
    data=dataset.yaml \
    split=test \
    imgsz=640
```

### Evaluation Metrics Output

**Validation Set** (during training):

```
Validation Metrics (Epoch 100):
mAP@0.5: 0.8764
mAP@0.5:0.95: 0.6543
Precision: 0.8234
Recall: 0.7891
```

**Test Set** (final evaluation):

```
Test Set Metrics:
mAP@0.5: 0.8512
mAP@0.5:0.95: 0.6321
Precision: 0.8156
Recall: 0.7743
```

**Note**: Test set metrics are typically slightly lower than validation metrics.

### Using the Trained Model

#### Python Inference

```python
from ultralytics import YOLO

# Load model
model = YOLO('runs/train/cat_detection/weights/best.pt')

# Predict on single image
results = model('path/to/image.jpg')

# Predict on multiple images
results = model(['image1.jpg', 'image2.jpg'])

# Predict on folder
results = model('path/to/images/')

# Access results
for r in results:
    boxes = r.boxes  # Bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Coordinates
        conf = box.conf[0]  # Confidence
        cls = box.cls[0]  # Class
```

#### Command Line Inference

```bash
yolo detect predict \
    model=runs/train/cat_detection/weights/best.pt \
    source=path/to/images \
    conf=0.25 \
    save=True
```

### Using Multiple Models (Original vs Enhanced)

If you trained two models - one on the original dataset and one on the enhanced dataset - here's how to use both:

#### Python - Using Both Models Separately

```python
from ultralytics import YOLO

# Load both models
model_original = YOLO('runs/train/cat_original/weights/best.pt')
model_enhanced = YOLO('runs/train/cat_enhanced/weights/best.pt')

# Test on a single image
test_image = 'path/to/test_image.jpg'

# Predict with original model
results_original = model_original(test_image)
print("Original Model Results:")
for box in results_original[0].boxes:
    conf = box.conf[0]
    print(f"  Confidence: {conf:.2f}")

# Predict with enhanced model
results_enhanced = model_enhanced(test_image)
print("Enhanced Model Results:")
for box in results_enhanced[0].boxes:
    conf = box.conf[0]
    print(f"  Confidence: {conf:.2f}")
```

#### Python - Comparing Both Models

```python
from ultralytics import YOLO
import os

# Load both models
model_original = YOLO('runs/train/cat_original/weights/best.pt')
model_enhanced = YOLO('runs/train/cat_enhanced/weights/best.pt')

# Test on multiple images
test_images_folder = 'path/to/test_images/'
test_images = [os.path.join(test_images_folder, img)
               for img in os.listdir(test_images_folder)
               if img.endswith(('.jpg', '.jpeg', '.png'))]

# Compare results
for img_path in test_images:
    print(f"\nTesting: {os.path.basename(img_path)}")

    # Original model
    results_orig = model_original(img_path, conf=0.25)
    num_detections_orig = len(results_orig[0].boxes)
    avg_conf_orig = sum([box.conf[0] for box in results_orig[0].boxes]) / num_detections_orig if num_detections_orig > 0 else 0

    # Enhanced model
    results_enh = model_enhanced(img_path, conf=0.25)
    num_detections_enh = len(results_enh[0].boxes)
    avg_conf_enh = sum([box.conf[0] for box in results_enh[0].boxes]) / num_detections_enh if num_detections_enh > 0 else 0

    print(f"  Original: {num_detections_orig} cats, avg conf: {avg_conf_orig:.2f}")
    print(f"  Enhanced: {num_detections_enh} cats, avg conf: {avg_conf_enh:.2f}")
```

#### Python - Ensemble Prediction (Use Both Models Together)

```python
from ultralytics import YOLO
import numpy as np

def ensemble_predict(image_path, models, conf_threshold=0.25):
    """
    Use multiple models and combine their predictions
    """
    all_boxes = []

    for model in models:
        results = model(image_path, conf=conf_threshold)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            all_boxes.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf
            })

    return all_boxes

# Load both models
models = [
    YOLO('runs/train/cat_original/weights/best.pt'),
    YOLO('runs/train/cat_enhanced/weights/best.pt')
]

# Run ensemble prediction
image = 'path/to/image.jpg'
detections = ensemble_predict(image, models, conf_threshold=0.25)

print(f"Total detections from both models: {len(detections)}")
for i, det in enumerate(detections):
    print(f"  Detection {i+1}: Confidence {det['confidence']:.2f}")
```

#### Command Line - Using Specific Models

```bash
# Use original model
yolo detect predict \
    model=runs/train/cat_original/weights/best.pt \
    source=path/to/images \
    conf=0.25 \
    save=True \
    project=results \
    name=original_predictions

# Use enhanced model
yolo detect predict \
    model=runs/train/cat_enhanced/weights/best.pt \
    source=path/to/images \
    conf=0.25 \
    save=True \
    project=results \
    name=enhanced_predictions
```

#### Which Model to Use?

**Use Original Model When:**

- Faster inference is needed (if trained on smaller dataset)
- Testing on similar data distribution as original training set
- Resource-constrained deployment

**Use Enhanced Model When:**

- Maximum accuracy is needed
- Testing on diverse/challenging images
- Images have augmentations similar to enhanced training set

**Compare Both When:**

- Evaluating model improvements
- Determining which performs better on your specific use case
- Creating research/academic reports

#### Typical Model Path Structure

```
runs/train/
├── cat_original/
│   └── weights/
│       ├── best.pt
│       └── last.pt
└── cat_enhanced/
    └── weights/
        ├── best.pt
        └── last.pt
```

---

## Tips and Best Practices

### Before Training

1. **Data Quality**

   - ✅ Check label accuracy (random sample)
   - ✅ Verify image-label pairs match
   - ✅ Remove corrupted images
   - ✅ Balance dataset (similar number of images per class)

2. **Data Size Guidelines**

   - Minimum: 300 images per class
   - Recommended: 1000+ images per class
   - Ideal: 5000+ images per class

3. **Train/Val/Test Split**

   - **Current project**: 1500 train / 300 val / 300 test (~75%/12.5%/12.5%)
   - Standard: 80% train / 20% validation (no test set)
   - Recommended: 70% train / 15% validation / 15% test
   - Alternative: 80% train / 10% validation / 10% test

   **Important**: Test set is optional but recommended for research/academic projects

4. **Hardware Optimization**
   - Monitor GPU memory usage
   - Adjust batch size to fit GPU memory
   - Use mixed precision (FP16) for faster training
   - Close other GPU-intensive applications

### During Training

1. **Monitoring**

   - Check first few epochs for errors
   - Watch for memory issues
   - Monitor loss convergence
   - Use TensorBoard for real-time visualization

2. **When to Stop Early**

   - Validation loss stops improving (early stopping)
   - Overfitting detected (train loss << val loss)
   - Satisfactory metrics achieved

3. **Adjustments**
   - If loss explodes → Reduce learning rate
   - If slow convergence → Increase learning rate
   - If overfitting → Add augmentation, reduce model size
   - If underfitting → Increase model size, reduce augmentation

### After Training

1. **Model Selection**

   - Use `best.pt` for production
   - Test on unseen data
   - Check inference speed vs accuracy trade-off

2. **Model Optimization**

   - Export to ONNX for faster inference
   - Export to TensorRT for NVIDIA GPUs
   - Export to CoreML for iOS/macOS
   - Quantize for mobile deployment

3. **Fine-tuning**

   - If results are poor, analyze failure cases
   - Add more diverse training data
   - Adjust augmentation parameters
   - Try different model sizes

4. **Deployment Checklist**
   - ✅ Test on representative data
   - ✅ Measure inference speed
   - ✅ Set appropriate confidence threshold
   - ✅ Handle edge cases (blurry, occluded, etc.)
   - ✅ Monitor performance in production

---

## Troubleshooting

### Common Issues

#### Issue 1: Out of Memory

```
CUDA out of memory error
```

**Solutions**:

- Reduce batch size (`--batch-size 8` or lower)
- Reduce image size (`--img-size 416`)
- Use smaller model (`--model-size n`)
- Close other GPU applications

#### Issue 2: Poor Performance

```
mAP < 0.5 after training
```

**Solutions**:

- Check label quality
- Increase training data
- Train longer (more epochs)
- Use larger model size
- Increase image size
- Enable pretrained weights

#### Issue 3: Slow Training

```
Training takes too long
```

**Solutions**:

- Use GPU instead of CPU
- Increase batch size (if GPU memory allows)
- Reduce image size
- Use smaller model for testing
- Enable cache (`cache=True` in code)

#### Issue 4: Overfitting

```
Train loss << Validation loss
```

**Solutions**:

- Add more training data
- Increase augmentation strength
- Reduce model size
- Add regularization
- Train for fewer epochs

---

## Quick Reference

### Complete Workflow Summary

```bash
# 1. TRAINING (uses train + validation sets)
python training/train_yolo.py \
    --data-dir ./data/original \
    --model-size s \
    --epochs 100 \
    --batch-size 16

# Training automatically:
# - Trains on training set
# - Evaluates on VALIDATION set each epoch
# - Saves best model based on validation mAP

# 2. VALIDATION SET EVALUATION (during development)
# Check validation metrics from training logs or:
python training/evaluate_model.py \
    --model runs/train/cat_detection/weights/best.pt \
    --data dataset.yaml \
    --split val

# 3. TEST SET EVALUATION (ONLY ONCE at the end!)
python training/evaluate_model.py \
    --model runs/train/cat_detection/weights/best.pt \
    --data dataset.yaml \
    --split test
```

### When to Use Each Dataset Split

| Split          | When to Use                   | Purpose                                | How Often               |
| -------------- | ----------------------------- | -------------------------------------- | ----------------------- |
| **Training**   | During model training         | Learn patterns                         | Every epoch             |
| **Validation** | During training & development | Monitor performance, select best model | Every epoch + as needed |
| **Test**       | After all training is done    | Report final performance               | ONCE only               |

### Recommended Configurations

#### Configuration 1: Quick Testing

```bash
python train_yolo.py --data-dir ../data --model-size n --epochs 50 --batch-size 16
```

#### Configuration 2: Balanced Production

```bash
python train_yolo.py --data-dir ../data --model-size s --epochs 100 --batch-size 32 --img-size 640
```

#### Configuration 3: High Accuracy

```bash
python train_yolo.py --data-dir ../data --model-size m --epochs 150 --batch-size 16 --img-size 640
```

#### Configuration 4: CPU Training

```bash
python train_yolo.py --data-dir ../data --model-size n --epochs 50 --batch-size 4 --device cpu
```

---

## Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLO Dataset Format Guide](https://docs.ultralytics.com/datasets/)
- [Training Tips and Tricks](https://docs.ultralytics.com/guides/model-training-tips/)
- [Model Export Guide](https://docs.ultralytics.com/modes/export/)

---

## Contact & Support

For questions or issues with this training setup, please refer to:

- Project README: `README.md`
- Implementation Guide: `IMPLEMENTATION_GUIDE.md`
- Training Instructions: `HOW_TO_TRAIN.md`

Last updated: January 2026
