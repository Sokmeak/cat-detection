# 🔧 Complete Label Correction Guide

## Step-by-Step Process for Correcting 416 Flagged Images

---

## ✅ What You've Done So Far

1. ✅ Flagged 416 images with incorrect labels using web viewer
2. ✅ Separated flagged images to `data/needs_correction/`

---

## 📝 STEP 1: Start the Label Correction Tool

```bash
cd /home/sunleang/Documents/I5/TP/AI/Project/Web/scripts
streamlit run label_correction_viewer.py
```

This opens a web interface at: `http://localhost:8501`

---

## 🎯 STEP 2: How to Correct Each Label

### Understanding YOLO Label Format

Each line in a `.txt` file:
```
0 x_center y_center width height
```

- **`0`**: Class ID (always 0 for "Cat")
- **`x_center`**: Horizontal center (0.0 to 1.0)
- **`y_center`**: Vertical center (0.0 to 1.0)
- **`width`**: Box width (0.0 to 1.0)
- **`height`**: Box height (0.0 to 1.0)

**Example:**
```
0 0.5123 0.6234 0.3456 0.4123
```

### Correction Process for Each Image

1. **Look at the image** - See where the cat actually is
2. **Check the green box** - Is it covering the cat correctly?
3. **If box is wrong:**
   - Click **"✏️ Edit Labels"**
   - Modify the coordinates
   - Click **"👁️ Preview"** to see changes
   - If good, click **"💾 Save"**
   - If bad, adjust and preview again
4. **Click "➡️ Next"** to move to next image

### Common Corrections Needed

#### Problem 1: Box Too Small
**Current:**
```
0 0.5 0.5 0.1 0.1
```
**Fix:** Increase width/height
```
0 0.5 0.5 0.4 0.5
```

#### Problem 2: Box Off-Center
**Current:**
```
0 0.2 0.3 0.3 0.4
```
**Fix:** Adjust x_center and y_center
```
0 0.5 0.6 0.3 0.4
```

#### Problem 3: Box Completely Wrong
**Current:**
```
0 0.9 0.1 0.2 0.2
```
**Fix:** Recalculate all values based on cat position
```
0 0.45 0.55 0.35 0.45
```

#### Problem 4: No Cat Visible (False Positive)
**Fix:** Delete the entire line (leave file empty or delete it)

#### Problem 5: Multiple Boxes, One Wrong
**Current:**
```
0 0.3 0.4 0.2 0.3
0 0.9 0.1 0.1 0.1
```
**Fix:** Remove the wrong one
```
0 0.3 0.4 0.2 0.3
```

---

## 🖱️ Web Interface Features

### Main Screen
- **Left side**: Image with bounding box overlay
- **Right side**: Label editor with validation

### Buttons
- **⬅️ Previous**: Go to previous image
- **➡️ Next**: Go to next image  
- **✏️ Edit Labels**: Enter edit mode
- **👁️ Preview**: See how changes look
- **💾 Save**: Save corrections
- **❌ Cancel**: Discard changes
- **Jump to Image**: Go to specific image number

### Progress Tracking
- Shows: "Image X of 416"
- Shows: "Corrected: Y" (how many you've saved)
- Progress bar at bottom

---

## 💡 Tips for Fast Correction

### 1. Use Keyboard Navigation (if available)
- Tab through fields
- Enter to save

### 2. Batch Similar Issues
- If many images have the same problem type, you'll get faster

### 3. Work in Sessions
- Correct 50-100 images at a time
- Take breaks to avoid fatigue

### 4. Focus on Quick Wins
- Fix obviously wrong boxes first
- Save complex cases for later

### 5. Validation Helps You
- Red errors show what's wrong
- Green checkmarks confirm validity
- Use preview before saving

---

## 📊 STEP 3: Track Your Progress

### During Correction
Check the sidebar:
- **Current Image**: Shows position (e.g., "215 / 416")
- **Corrected**: How many you've saved

### Session Planning
- **416 images total**
- **~30 seconds per image** = 3.5 hours
- **Break into sessions**:
  - Session 1: Images 1-100 (~50 min)
  - Session 2: Images 101-200 (~50 min)
  - Session 3: Images 201-300 (~50 min)
  - Session 4: Images 301-416 (~1 hour)

---

## ✅ STEP 4: Validate Corrected Labels

After correcting all 416 images:

```bash
cd /home/sunleang/Documents/I5/TP/AI/Project/Web/scripts

python check_labels.py \
    --images ../data/needs_correction/images \
    --labels ../data/needs_correction/labels \
    --output ../results/corrected_validation
```

**Expected output:**
```
Valid images: 416 (100%)
Images with invalid bboxes: 0
```

---

## 📥 STEP 5: Copy Corrected Labels Back

```bash
python copy_corrected_labels.py
```

This will:
1. ✅ Backup original labels to `labels_backup_before_correction/`
2. ✅ Copy all 416 corrected labels to `data/original/train/labels/`

---

## 🔍 STEP 6: Final Dataset Validation

Validate the entire training set:

```bash
python check_labels.py \
    --images ../data/original/train/images \
    --labels ../data/original/train/labels \
    --output ../results/final_train_validation \
    --remove-invalid
```

**Expected:**
- Total images: 1500
- Valid images: 1500 (100%)
- Invalid: 0

---

## 🚀 STEP 7: Train the Model

```bash
cd /home/sunleang/Documents/I5/TP/AI/Project/Web

python training/train_yolo.py \
    --data-dir ./data/original \
    --model-size n \
    --epochs 100 \
    --batch-size 16 \
    --name cat_detection_corrected \
    --evaluate
```

**Training will take:** 30-60 minutes

**Results saved to:** `runs/train/cat_detection_corrected/`

---

## 🎯 Success Checklist

- [ ] Started label correction viewer
- [ ] Corrected all 416 flagged images
- [ ] Validated corrected labels (no errors)
- [ ] Copied corrected labels back
- [ ] Final validation passed (1500/1500 valid)
- [ ] Started training with clean dataset
- [ ] Training completed successfully

---

## 🆘 Troubleshooting

### Web viewer won't start
```bash
pip install streamlit opencv-python
```

### Can't see images
- Check: `data/needs_correction/images/` has 416 images
- Run: `ls ../data/needs_correction/images/ | wc -l`

### Validation shows errors after correction
- Review those specific images again
- Check coordinate ranges (must be 0.0-1.0)
- Use preview mode before saving

### Lost progress
- Don't worry! Saved labels are permanent
- Just restart from where you left off

---

## 📞 Quick Reference

**Start Correction:**
```bash
cd /home/sunleang/Documents/I5/TP/AI/Project/Web/scripts
streamlit run label_correction_viewer.py
```

**Check Progress:**
```bash
ls ../data/needs_correction/labels/ | wc -l  # Should be 416
```

**Validate:**
```bash
python check_labels.py --images ../data/needs_correction/images --labels ../data/needs_correction/labels
```

**Copy Back:**
```bash
python copy_corrected_labels.py
```

**Train:**
```bash
cd .. && python training/train_yolo.py --data-dir ./data/original --epochs 100
```

---

## 🎓 You're Ready!

**Start now with:**
```bash
streamlit run label_correction_viewer.py
```

Good luck! 🚀
