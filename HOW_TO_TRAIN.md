# 🚀 How to Train YOLOv8 on Cat Detection Dataset

## ⚠️ IMPORTANT: Correct Command

You encountered an error because you used the **wrong path**. Here's the fix:

### ❌ WRONG (causes error):

```bash
--data-dir ./data/original/train
```

**Error:** Looking for `data/original/train/val/images` (doesn't exist)

### ✅ CORRECT:

```bash
--data-dir ./data/original
```

**Why:** The script expects `train/`, `val/`, `test/` folders inside the data directory.

---

## 📂 Your Data Structure

```
cat-detection/
└── data/
    └── original/
        ├── train/
        │   ├── images/  (1502 images)
        │   └── labels/  (1502 labels)
        ├── val/
        │   ├── images/  (300 images)
        │   └── labels/  (300 labels)
        └── test/
            ├── images/  (300 images)
            └── labels/  (300 labels)
```

---

## 🎯 Step-by-Step Training Instructions

### Step 1: Navigate to Project Directory

```bash
cd /Users/macbook/Documents/Documents\ -\ SokMeak/ITC-program/I5/Image\ Processing/Project25/cat-detection
```

### Step 2: Run Training on Original Dataset

```bash
./bin/python training/train_yolo.py \
    --data-dir ./data/original \
    --model-size n \
    --epochs 100 \
    --batch-size 16 \
    --name cat_detection_original \
    --evaluate
```

### Step 3: Wait for Training to Complete

**Expected time:** 30-60 minutes (CPU training on M1 Pro)

**What happens during training:**

1. Downloads YOLOv8n pretrained weights (~6.2 MB)
2. Trains for 100 epochs
3. Saves checkpoint every 10 epochs
4. Automatically evaluates on validation set after training

---

## 📊 Where Your Metrics Are Stored

After training completes, all results are saved to:

```
runs/train/cat_detection_original/
```

### 📁 Files Created:

| File/Folder                         | Description                                       |
| ----------------------------------- | ------------------------------------------------- |
| **weights/best.pt**                 | ✨ Best model (highest validation mAP)            |
| **weights/last.pt**                 | Latest model (final epoch)                        |
| **results.csv**                     | 📈 All metrics per epoch (CSV format)             |
| **results.png**                     | 📊 Training curves (loss, mAP, precision, recall) |
| **confusion_matrix.png**            | Confusion matrix                                  |
| **confusion_matrix_normalized.png** | Normalized confusion matrix                       |
| **PR_curve.png**                    | Precision-Recall curve                            |
| **F1_curve.png**                    | F1 score vs confidence threshold                  |
| **P_curve.png**                     | Precision vs confidence threshold                 |
| **R_curve.png**                     | Recall vs confidence threshold                    |
| **labels.jpg**                      | Ground truth label distribution                   |
| **labels_correlogram.jpg**          | Label correlation analysis                        |
| **train_batch\*.jpg**               | Sample training batches                           |
| **val_batch\*\_pred.jpg**           | Validation predictions                            |
| **args.yaml**                       | Training configuration (all hyperparameters)      |

---

## 📈 Understanding Your Metrics

### Key Performance Metrics

The training outputs these important metrics (all saved in `results.csv`):

| Metric           | Description                       | Goal                   |
| ---------------- | --------------------------------- | ---------------------- |
| **mAP@0.5**      | Mean Average Precision at IoU 0.5 | Higher is better (0-1) |
| **mAP@0.5:0.95** | mAP averaged across IoU 0.5-0.95  | Higher is better (0-1) |
| **Precision**    | % of correct detections           | Higher is better (0-1) |
| **Recall**       | % of ground truth objects found   | Higher is better (0-1) |
| **Box Loss**     | Bounding box regression loss      | Lower is better        |
| **Class Loss**   | Classification loss               | Lower is better        |
| **DFL Loss**     | Distribution Focal Loss           | Lower is better        |

### Terminal Output Example

During training, you'll see output like this:

```
Epoch   GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
1/100     0.0G      1.234      0.567      1.123         42      640

                 Class     Images  Instances      Box(P)      Rec     mAP50  mAP50-95
                   all        300        350       0.756    0.689     0.745     0.512
```

---

## 🎓 Training Parameters Explained

| Parameter      | Value                    | Description                                                                  |
| -------------- | ------------------------ | ---------------------------------------------------------------------------- |
| `--data-dir`   | `./data/original`        | Root directory containing train/val/test                                     |
| `--model-size` | `n`                      | Model size: `n` (nano), `s` (small), `m` (medium), `l` (large), `x` (xlarge) |
| `--epochs`     | `100`                    | Number of complete passes through the dataset                                |
| `--batch-size` | `16`                     | Number of images processed together                                          |
| `--name`       | `cat_detection_original` | Name for this training run (creates folder)                                  |
| `--evaluate`   | (flag)                   | Run evaluation on test set after training                                    |

### Other Available Parameters

```bash
# Adjust learning rate
--lr0 0.01              # Initial learning rate

# Change optimizer
--optimizer SGD         # Options: SGD, Adam, AdamW, auto

# Change image size
--img-size 640          # Input image size (default: 640)

# Save checkpoints more/less frequently
--save-period 5         # Save every N epochs (default: 10)

# Use GPU (if available)
--device 0              # Use GPU 0 (default: auto-detect)
```

---

## 🔄 Training on Enhanced Dataset

After image enhancement (Step 5 in workflow), train on enhanced data:

```bash
./bin/python training/train_yolo.py \
    --data-dir ./data/enhanced \
    --model-size n \
    --epochs 100 \
    --batch-size 16 \
    --name cat_detection_enhanced \
    --evaluate
```

**Results saved to:** `runs/train/cat_detection_enhanced/`

---

## 📊 How to View Your Results

### Option 1: View Training Curves

Open the generated plot:

```bash
open runs/train/cat_detection_original/results.png
```

### Option 2: Analyze CSV Data

```bash
# View with Numbers/Excel
open runs/train/cat_detection_original/results.csv

# Or view in terminal
cat runs/train/cat_detection_original/results.csv
```

### Option 3: Check Best Metrics

The final evaluation results are printed in the terminal after training completes:

```
Evaluation Metrics:
mAP@0.5: 0.7845
mAP@0.5:0.95: 0.5234
Precision: 0.8123
Recall: 0.7654
```

---

## 🐛 Troubleshooting

### Error: "Dataset images not found"

**Cause:** Wrong `--data-dir` path

**Solution:** Use `./data/original` NOT `./data/original/train`

### Error: "CUDA out of memory"

**Solution:** Reduce batch size:

```bash
--batch-size 8  # or smaller
```

### Training is very slow

**Cause:** Running on CPU (M1 Mac)

**Options:**

1. Reduce `--epochs 50` for faster testing
2. Use smaller `--batch-size 8`
3. Use GPU-enabled machine

### Want to resume training?

```bash
./bin/python training/train_yolo.py \
    --model-path runs/train/cat_detection_original/weights/last.pt \
    --data-dir ./data/original \
    --epochs 150 \
    --evaluate
```

---

## ✅ Quick Command Reference

### Train on Original Dataset

```bash
cd cat-detection
./bin/python training/train_yolo.py --data-dir ./data/original --model-size n --epochs 100 --batch-size 16 --name cat_detection_original --evaluate
```

### Train on Enhanced Dataset

```bash
cd cat-detection
./bin/python training/train_yolo.py --data-dir ./data/enhanced --model-size n --epochs 100 --batch-size 16 --name cat_detection_enhanced --evaluate
```

### View Results

```bash
# Open results folder
open runs/train/cat_detection_original/

# View training curves
open runs/train/cat_detection_original/results.png

# View metrics CSV
open runs/train/cat_detection_original/results.csv
```

---

## 🧪 Validation vs Testing

### During Training: Automatic Validation

**What happens automatically:**

- **Every epoch**, the model is validated on the **validation set** (`data/original/val/`)
- Metrics are computed: mAP@0.5, mAP@0.5:0.95, Precision, Recall
- Best model is saved based on validation mAP
- Results plotted in `results.png`

**No action needed** - this happens automatically! ✅

### After Training: Evaluate on Test Set

The `--evaluate` flag in the training command runs evaluation on the **validation set** by default. To test on the **test set**, you have two options:

#### Option 1: Evaluate Using Training Script

```bash
cd cat-detection

# Evaluate best model on validation set (default)
./bin/python training/train_yolo.py \
    --model-path runs/train/cat_detection_original/weights/best.pt \
    --data-dir ./data/original \
    --evaluate
```

#### Option 2: Use YOLO CLI Directly

```bash
cd cat-detection

# Evaluate on validation set
./bin/python -c "
from ultralytics import YOLO
model = YOLO('runs/train/cat_detection_original/weights/best.pt')
metrics = model.val(data='dataset.yaml', split='val')
print(f'mAP@0.5: {metrics.box.map50:.4f}')
print(f'mAP@0.5:0.95: {metrics.box.map:.4f}')
print(f'Precision: {metrics.box.mp:.4f}')
print(f'Recall: {metrics.box.mr:.4f}')
"

# Evaluate on test set
./bin/python -c "
from ultralytics import YOLO
model = YOLO('runs/train/cat_detection_original/weights/best.pt')
metrics = model.val(data='dataset.yaml', split='test')
print(f'mAP@0.5: {metrics.box.map50:.4f}')
print(f'mAP@0.5:0.95: {metrics.box.map:.4f}')
print(f'Precision: {metrics.box.mp:.4f}')
print(f'Recall: {metrics.box.mr:.4f}')
"
```

### Understanding the Data Splits

| Split          | Size        | Purpose                | When Used                        |
| -------------- | ----------- | ---------------------- | -------------------------------- |
| **Train**      | 1502 images | Model learns from this | Every training epoch             |
| **Validation** | 300 images  | Select best model      | After each epoch during training |
| **Test**       | 300 images  | Final evaluation       | After training completes         |

**Important:**

- ✅ **Validation set** = Used during training to pick best model
- ✅ **Test set** = Used only ONCE at the end for final performance report
- ❌ Never train on validation or test sets!

---

## 📊 How to Compare Original vs Enhanced

After training both models, create a comparison:

### Step 1: Train Both Models

```bash
# Train on original dataset
./bin/python training/train_yolo.py \
    --data-dir ./data/original \
    --model-size n \
    --epochs 100 \
    --batch-size 16 \
    --name cat_detection_original \
    --evaluate

# Train on enhanced dataset
./bin/python training/train_yolo.py \
    --data-dir ./data/enhanced \
    --model-size n \
    --epochs 100 \
    --batch-size 16 \
    --name cat_detection_enhanced \
    --evaluate
```

### Step 2: Evaluate Both on Test Set

```bash
# Evaluate original model on test set
./bin/python -c "
from ultralytics import YOLO
print('=== ORIGINAL MODEL ===')
model = YOLO('runs/train/cat_detection_original/weights/best.pt')
metrics = model.val(data='dataset.yaml', split='test')
print(f'Test mAP@0.5: {metrics.box.map50:.4f}')
print(f'Test mAP@0.5:0.95: {metrics.box.map:.4f}')
print(f'Test Precision: {metrics.box.mp:.4f}')
print(f'Test Recall: {metrics.box.mr:.4f}')
"

# Evaluate enhanced model on test set
./bin/python -c "
from ultralytics import YOLO
print('=== ENHANCED MODEL ===')
model = YOLO('runs/train/cat_detection_enhanced/weights/best.pt')
metrics = model.val(data='dataset.yaml', split='test')
print(f'Test mAP@0.5: {metrics.box.map50:.4f}')
print(f'Test mAP@0.5:0.95: {metrics.box.map:.4f}')
print(f'Test Precision: {metrics.box.mp:.4f}')
print(f'Test Recall: {metrics.box.mr:.4f}')
"
```

### Step 3: Compare Results CSV

```bash
# View original model results
cat runs/train/cat_detection_original/results.csv | tail -1

# View enhanced model results
cat runs/train/cat_detection_enhanced/results.csv | tail -1
```

### Step 4: Create Comparison Table

| Metric       | Original Dataset | Enhanced Dataset | Improvement |
| ------------ | ---------------- | ---------------- | ----------- |
| mAP@0.5      | 0.XXX            | 0.XXX            | +X.X%       |
| mAP@0.5:0.95 | 0.XXX            | 0.XXX            | +X.X%       |
| Precision    | 0.XXX            | 0.XXX            | +X.X%       |
| Recall       | 0.XXX            | 0.XXX            | +X.X%       |

---

## 📝 Next Steps After Training

1. ✅ Training completes → Results saved to `runs/train/cat_detection_original/`
2. 📊 Analyze metrics in `results.csv` and `results.png`
3. 🖼️ Check prediction samples in `val_batch*_pred.jpg`
4. 🎯 **Evaluate on test set** for final metrics
5. ✨ Enhance images (if not done) → `scripts/enhance_images.py`
6. 🔄 Train on enhanced dataset
7. 📈 Compare original vs enhanced test results
8. 📄 Document findings in report

---

## 🎉 Expected Results

For the Cat detection task, you should aim for:

- **mAP@0.5:** > 0.75 (75%)
- **mAP@0.5:0.95:** > 0.50 (50%)
- **Precision:** > 0.80 (80%)
- **Recall:** > 0.75 (75%)

Good luck with your training! 🚀🐱
