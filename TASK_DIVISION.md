# Task Division for Cat Detection Project

## ⚠️ Important: Data Folder Issue

The `/data` folder is gitignored. **Share data via Google Drive/Dropbox**, or comment out `data/` in `.gitignore` to commit label files.

---

## Team Tasks & Branches

### Contributor 1: Training Set Validation

**Branch:** `feature/validate-training-labels`

```bash
git checkout -b feature/validate-training-labels
```

**Task:** Validate labels in `data/original/train/labels/` and `data/enhanced/train/labels/`

- Check bounding box coordinates
- Find missing/incorrect labels
- Create report: `reports/training_label_validation.md`

---

### Contributor 2: Validation Set Validation

**Branch:** `feature/validate-val-labels`

```bash
git checkout -b feature/validate-val-labels
```

**Task:** Validate labels in `data/original/val/labels/` and `data/enhanced/val/labels/`

- Check bounding box coordinates
- Find missing/incorrect labels
- Create report: `reports/validation_label_validation.md`

---

### Contributor 3: Test Set Validation

**Branch:** `feature/validate-test-labels`

```bash
git checkout -b feature/validate-test-labels
```

**Task:** Validate labels in `data/original/test/labels/` and `data/enhanced/test/labels/`

- Check bounding box coordinates
- Find missing/incorrect labels
- Create report: `reports/test_label_validation.md`

---

### Contributor 4: Image Enhancement Review

**Branch:** `feature/enhancement-validation`

```bash
git checkout -b feature/enhancement-validation
```

**Task:** Review images needing enhancement + propose methods

- Identify low-quality images (blur, lighting, etc.)
- Document enhancement methods (CLAHE, denoising, sharpening, etc.)
- Create reports:
  - `reports/images_needing_enhancement.md`
  - `reports/proposed_enhancement_methods.md`

---

## Workflow

```bash
# 1. Create your branch
git checkout -b <your-branch-name>

# 2. Do your work & document findings

# 3. Commit changes
git add reports/*.md
git commit -m "feat: complete [task name]"
git push origin <your-branch-name>

# 4. Create Pull Request on GitHub
```

---

## Report Template

```markdown
# [Task Name] Report

**Date:** [Date]
**Name:** [Your Name]

## Summary

- Images/labels checked: [N]
- Issues found: [N]
- Corrections made: [N]

## Issues

1. [filename]: [problem] → [action taken]
2. ...

## Recommendations

[Suggestions]
```
