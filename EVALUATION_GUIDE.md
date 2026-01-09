# Model Evaluation Guide: Validation vs Test Set

## Quick Summary

Your cat detection project has **3 dataset splits**:

- **Training set**: 1,500 images (75%) - for training the model
- **Validation set**: 300 images (12.5%) - for monitoring during training
- **Test set**: 300 images (12.5%) - for final evaluation only

## Critical Issues Found & Fixed

### ❌ Previous Issues

1. **Confusion about validation vs test set usage**

   - The `evaluate_model()` function only evaluated on validation set
   - No clear guidance on when to use test set
   - Risk of data leakage by repeatedly using test set

2. **Missing test set evaluation capability**
   - Could not explicitly evaluate on test set
   - No warnings about proper test set usage

### ✅ What Was Fixed

1. **Updated `train_yolo.py`**

   - Added `--eval-split` parameter to choose val or test
   - Updated `evaluate_model()` to accept split parameter
   - Default remains `val` to prevent accidental test set usage

2. **Created `evaluate_model.py`**

   - Standalone script for clean evaluation
   - Clear warnings when using test set
   - Saves metrics to file for documentation
   - Better formatted output

3. **Updated TRAINING_GUIDE.md**
   - Clear explanation of validation vs test set
   - When to use each dataset split
   - Complete workflow with examples
   - Critical rules to follow

## How to Properly Evaluate Your Model

### During Training (Automatic)

```bash
python training/train_yolo.py \
    --data-dir ./data/original \
    --model-size s \
    --epochs 100
```

**What happens**:

- Trains on **training set** (1,500 images)
- After each epoch, evaluates on **validation set** (300 images)
- Saves `best.pt` based on **validation mAP**
- You can monitor these validation metrics

### During Development (As Needed)

```bash
# Check validation performance anytime
python training/evaluate_model.py \
    --model runs/train/cat_detection/weights/best.pt \
    --split val
```

**When to do this**:

- Compare different models
- Test different confidence thresholds
- Tune hyperparameters
- As often as needed

### Final Evaluation (ONCE ONLY)

```bash
# Evaluate on test set - do this only once!
python training/evaluate_model.py \
    --model runs/train/cat_detection/weights/best.pt \
    --split test
```

**When to do this**:

- After all training and tuning is complete
- When you're ready to report final results
- For your project report/paper
- **ONLY ONCE** - never tune based on test metrics

## Why This Matters

### The Problem: Data Leakage

If you repeatedly evaluate on the test set and make decisions based on it:

1. You inadvertently "train" on the test set
2. Your model becomes optimized for that specific test set
3. Reported metrics are **overly optimistic**
4. Model may not generalize to new real-world data

### The Solution: Proper Dataset Usage

```
Training Set → Learn patterns
     ↓
Validation Set → Monitor & select best model (can use repeatedly)
     ↓
Test Set → Final unbiased evaluation (use ONCE only)
```

## Correct Workflow for Your Project

### Phase 1: Model Development

```bash
# Try different configurations
python training/train_yolo.py --model-size n --epochs 50  # Quick test
python training/train_yolo.py --model-size s --epochs 100  # Better model
python training/train_yolo.py --model-size m --epochs 100  # Best model

# Each training run automatically uses validation set
# You can check validation metrics to compare models
```

### Phase 2: Model Selection

```bash
# Evaluate different models on validation set
python training/evaluate_model.py --model runs/train/exp1/weights/best.pt --split val
python training/evaluate_model.py --model runs/train/exp2/weights/best.pt --split val
python training/evaluate_model.py --model runs/train/exp3/weights/best.pt --split val

# Pick the best model based on VALIDATION metrics
```

### Phase 3: Final Testing (ONCE)

```bash
# Only after everything is done, evaluate on test set
python training/evaluate_model.py \
    --model runs/train/best_experiment/weights/best.pt \
    --split test \
    > test_results.txt

# These test metrics go in your final report
```

## Understanding the Metrics

### Validation Metrics (During Training)

- Used to select hyperparameters
- Used to choose best model
- Can be checked multiple times
- Expected to be slightly **higher** than test metrics

### Test Metrics (Final Evaluation)

- Used to report final performance
- Should only be computed once
- Provides unbiased estimate
- Usually slightly **lower** than validation metrics

### Why Test Metrics Are Lower

- No model selection bias
- Truly unseen data
- More representative of real-world performance

## Example Results

### Training Output (Validation Metrics)

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
100/100      4.2G      0.234      0.167      0.634         64        640: 100%

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        300        450      0.834      0.789      0.856     0.645
```

### Final Test Evaluation

```
TEST SET EVALUATION RESULTS
====================================================================
📊 Detection Metrics:
   mAP@0.5       : 0.8412   (0.85+ is good)
   mAP@0.5:0.95  : 0.6321   (0.60+ is good)
   Precision     : 0.8156   (0.80+ is good)
   Recall        : 0.7743   (0.75+ is good)
   F1 Score      : 0.7944   (0.75+ is good)
```

Notice test metrics (0.8412, 0.6321) are slightly lower than validation (0.856, 0.645).
This is **normal and expected**!

## Commands Reference

### Training (uses validation automatically)

```bash
python training/train_yolo.py --data-dir ./data/original --model-size s --epochs 100
```

### Evaluate on Validation (during development)

```bash
python training/evaluate_model.py --model best.pt --split val
```

### Evaluate on Test (final, once only)

```bash
python training/evaluate_model.py --model best.pt --split test
```

### Old way (still works)

```bash
python training/train_yolo.py --model-path best.pt --evaluate --eval-split test
```

## Common Mistakes to Avoid

❌ **Don't**:

- Don't use test set during model selection
- Don't tune hyperparameters based on test metrics
- Don't evaluate on test set multiple times
- Don't make training decisions based on test performance

✅ **Do**:

- Use validation set for all development decisions
- Reserve test set for final evaluation only
- Report test metrics in your final results
- Use validation metrics to compare models

## For Your Project Report

### Report These Metrics

**Validation Set Performance** (best model):

- mAP@0.5: X.XXXX
- mAP@0.5:0.95: X.XXXX
- Precision: X.XXXX
- Recall: X.XXXX

**Test Set Performance** (final evaluation):

- mAP@0.5: X.XXXX
- mAP@0.5:0.95: X.XXXX
- Precision: X.XXXX
- Recall: X.XXXX

### Include This Statement

"The model was trained on 1,500 images and validated on 300 images during training. The best model was selected based on validation mAP. Final performance was evaluated on a held-out test set of 300 images that was never used during training or model selection."

## Summary

✅ **Training**: Uses train + validation sets automatically
✅ **Development**: Check validation set as often as needed
✅ **Final Report**: Evaluate test set ONCE and report those metrics

This ensures your reported performance is honest and your model will generalize well to new cat images!
