# Cat Detection using Open Images Dataset

A single-class object detection system for detecting cats using the Open Images Dataset. This project implements a complete machine learning pipeline including data collection, preprocessing, quality enhancement, model training, and performance evaluation.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Open Images Dataset](https://img.shields.io/badge/Dataset-Open%20Images-green.svg)](https://storage.googleapis.com/openimages/web/index.html)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Objectives](#key-objectives)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Information](#dataset-information)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Results](#results)
- [License](#license)

---

## 🎯 Project Overview

This project develops and evaluates a **single-class object detection system** focused on the **Cat** class from the Open Images Dataset. The complete pipeline covers:

- **Data Collection**: Automated downloading of 1,500-2,000 cat images
- **Data Preprocessing**: Quality verification and annotation validation
- **Data Enhancement**: Brightness correction and image sharpening
- **Model Training**: Single-class object detection model (YOLO/Faster R-CNN)
- **Performance Evaluation**: Comprehensive metrics and comparative analysis

---

## 🎯 Key Objectives

### 1. Data Collection

- Select the **Cat** class from the Open Images Dataset
- Download **1,500–2,000 images** containing cats
- Use predefined splits:
  - **Training**: 1,500 images
  - **Validation**: 300 images
  - **Testing**: 300 images
- Organize dataset for object detection models

### 2. Data Preprocessing and Quality Enhancement

- Verify bounding box annotations correspond to Cat class
- Remove unlabeled, duplicated, or incorrectly annotated images
- Analyze image quality (brightness and contrast)
- Apply image processing techniques:
  - Brightness correction
  - Image sharpening
  - Noise reduction

### 3. Model Development

- Select appropriate object detection model (YOLO or Faster R-CNN)
- Configure training environment and hyperparameters
- Train single-class object detection model
- Implement data augmentation strategies

### 4. Model Evaluation

- Evaluate using standard metrics:
  - **Precision**
  - **Recall**
  - **Mean Average Precision (mAP)**
- Analyze detection errors:
  - False positives
  - False negatives
- Generate confusion matrices and performance visualizations

### 5. Comparative Analysis

- Compare model performance:
  - Original dataset vs. enhanced dataset
  - Impact of preprocessing techniques
- Assess image enhancement impact on detection accuracy

### 6. Reporting

- Document methodology, experiments, and results
- Present findings and recommendations
- Provide reproducible research artifacts

---

## ✨ Features

- **Automated Data Collection**: Script to download images from Open Images Dataset
- **Data Quality Analysis**: Tools to assess and improve image quality
- **Image Enhancement**: Preprocessing pipeline for brightness and sharpness
- **Model Training**: Support for YOLO and Faster R-CNN architectures
- **Comprehensive Evaluation**: Multiple metrics and visualization tools
- **Reproducible Results**: Seed management and configuration tracking
- **Interactive Notebooks**: Jupyter notebooks for experimentation

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Git
- pip package manager
- Virtual environment (recommended)

### Clone the Repository

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cat-detection.git

# Navigate to the project directory
cd cat-detection
```

### Set Up Virtual Environment

#### On macOS/Linux:

```bash
# Create virtual environment
python3 -m venv cat-detection

# Activate virtual environment
source cat-detection/bin/activate
```

#### On Windows:

```bash
# Create virtual environment
python -m venv cat-detection

# Activate virtual environment
cat-detection\Scripts\activate
```

### Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### Install FiftyOne (for dataset management)

```bash
pip install fiftyone
```

### Additional Dependencies

For GPU acceleration (recommended for training):

```bash
# For PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For TensorFlow with GPU support
pip install tensorflow-gpu
```

---

## 📖 Usage

### 1. Download Dataset

```bash
# Run the download script
python download.py --class cat --train 1500 --val 300 --test 300

# Or use FiftyOne for interactive dataset exploration
fiftyone app launch
```

### 2. Data Preprocessing

Open the Jupyter notebook:

```bash
jupyter notebook Image_+_AI_Group_11_Cat_Detection.ipynb
```

Run the preprocessing cells to:

- Verify annotations
- Remove duplicates
- Analyze image quality
- Apply enhancements

### 3. Train Model

```bash
# Train YOLO model (example)
python train.py --model yolo --epochs 100 --batch-size 16

# Train with enhanced dataset
python train.py --model yolo --epochs 100 --batch-size 16 --enhanced
```

### 4. Evaluate Model

```bash
# Evaluate on test set
python evaluate.py --model-path weights/best.pt --dataset test

# Generate performance report
python evaluate.py --model-path weights/best.pt --dataset test --report
```

### 5. Run Inference

```bash
# Detect cats in new images
python detect.py --model-path weights/best.pt --source path/to/images

# Run on webcam
python detect.py --model-path weights/best.pt --source 0
```

---

## 📊 Dataset Information

### Open Images Dataset

- **Source**: [Open Images Dataset V7](https://storage.googleapis.com/openimages/web/index.html)
- **Class**: Cat (/m/01yrx)
- **Total Images**: 1,500-2,000
- **Splits**:
  - Training: 1,500 images
  - Validation: 300 images
  - Testing: 300 images

### Dataset Structure

```
data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Annotation Format

Annotations are provided in YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized to [0, 1].

---

## 🏗️ Model Architecture

### Supported Models

1. **YOLOv8** (Recommended)

   - Fast inference speed
   - High accuracy
   - Easy to train and deploy

2. **Faster R-CNN**
   - High accuracy
   - Better for small objects
   - Slower inference

### Model Configuration

- **Input Size**: 640x640 (YOLO) or 800x800 (Faster R-CNN)
- **Backbone**: CSPDarknet53 (YOLO) or ResNet50 (Faster R-CNN)
- **Optimizer**: Adam or SGD
- **Learning Rate**: 0.001 (with cosine decay)
- **Batch Size**: 16-32
- **Epochs**: 100-300

---

## 📈 Evaluation Metrics

The model is evaluated using:

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds 0.5 to 0.95
- **Confusion Matrix**: True positives, false positives, false negatives
- **Inference Time**: Average detection time per image

---

## 📁 Project Structure

```
cat-detection/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── download.py                         # Dataset download script
├── train.py                           # Model training script
├── evaluate.py                        # Model evaluation script
├── detect.py                          # Inference script
├── Image_+_AI_Group_11_Cat_Detection.ipynb  # Main notebook
├── data/                              # Dataset directory
│   ├── train/
│   ├── val/
│   └── test/
├── models/                            # Model configurations
│   ├── yolo.yaml
│   └── faster_rcnn.yaml
├── weights/                           # Trained model weights
│   ├── best.pt
│   └── last.pt
├── results/                           # Training results and visualizations
│   ├── plots/
│   ├── metrics/
│   └── reports/
├── preprocessing/                     # Preprocessing utilities
│   ├── quality_check.py
│   ├── enhancement.py
│   └── augmentation.py
└── utils/                            # Helper functions
    ├── dataset.py
    ├── metrics.py
    └── visualization.py
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### How to Contribute

1. **Fork the Repository**

   ```bash
   # Click the 'Fork' button on GitHub
   ```

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/YOUR_USERNAME/cat-detection.git
   cd cat-detection
   ```

3. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**

   - Write clean, documented code
   - Follow PEP 8 style guidelines
   - Add tests if applicable

5. **Commit Your Changes**

   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

6. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Describe your changes and submit

### Contribution Guidelines

- **Code Style**: Follow PEP 8 for Python code
- **Documentation**: Update README.md if adding new features
- **Testing**: Ensure all tests pass before submitting
- **Commits**: Use clear, descriptive commit messages
- **Issues**: Check existing issues before creating new ones

### Areas for Contribution

- 🐛 **Bug Fixes**: Report and fix bugs
- ✨ **New Features**: Add support for more models or datasets
- 📝 **Documentation**: Improve documentation and examples
- 🧪 **Testing**: Add unit tests and integration tests
- 🎨 **Visualization**: Enhance result visualization tools
- ⚡ **Performance**: Optimize training and inference speed

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Follow project guidelines

---

## 📊 Results

### Expected Performance

| Metric       | Original Dataset | Enhanced Dataset |
| ------------ | ---------------- | ---------------- |
| mAP@0.5      | TBD              | TBD              |
| mAP@0.5:0.95 | TBD              | TBD              |
| Precision    | TBD              | TBD              |
| Recall       | TBD              | TBD              |
| F1-Score     | TBD              | TBD              |

_Results will be updated after model training and evaluation._

### Sample Detections

Sample detection results and visualizations will be added to the `results/` directory after training.

---

## 🔮 Future Work

- [ ] Extend to multi-class detection (cats, dogs, etc.)
- [ ] Implement real-time detection on video streams
- [ ] Add mobile deployment (TensorFlow Lite/ONNX)
- [ ] Experiment with advanced architectures (YOLOv10, DETR)
- [ ] Add data augmentation techniques (CutMix, Mosaic)
- [ ] Create web interface for easy inference
- [ ] Implement active learning for continuous improvement

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Image Processing Group 11**

- Project Contributors: [List team members here]

---

## 🙏 Acknowledgments

- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) for providing the dataset
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 implementation
- [FiftyOne](https://github.com/voxel51/fiftyone) for dataset management tools
- All contributors and the open-source community

---

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Email**: [your.email@example.com]
- **GitHub Issues**: [Create an issue](https://github.com/YOUR_USERNAME/cat-detection/issues)
- **Project Link**: [https://github.com/YOUR_USERNAME/cat-detection](https://github.com/YOUR_USERNAME/cat-detection)

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

---

**Happy Detecting! 🐱**
