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

- **Automated Data Collection**: FiftyOne-based script to download images from Open Images Dataset
- **Data Quality Analysis**: Tools to assess and improve image quality
  - Label verification ([check_labels.py](scripts/check_labels.py))
  - Invalid image detection ([find_invalid_images.py](scripts/find_invalid_images.py))
  - Single image visualization ([visualize_single_image.py](scripts/visualize_single_image.py))
- **Image Enhancement**: Preprocessing pipeline for brightness correction and sharpness ([enhance_images.py](scripts/enhance_images.py))
- **YOLOv8 Training**: Complete training pipeline with configurable hyperparameters
- **Comprehensive Evaluation**: Multiple metrics and visualization tools (mAP, precision, recall, confusion matrix)
- **Virtual Environment**: Pre-configured Python environment with all dependencies
- **Interactive Notebooks**: Jupyter notebook for experimentation and analysis
- **Dual Dataset Support**: Train on original or enhanced datasets for comparison

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher (Python 3.14 used in this project)
- Git
- pip package manager
- Virtual environment (already configured in this project)

### Setup Instructions

#### 1. Clone the Repository (if needed)

```bash
git clone https://github.com/YOUR_USERNAME/cat-detection.git
cd cat-detection
```

#### 2. Activate Virtual Environment

This project already has a configured virtual environment. Simply activate it:

##### On macOS/Linux:

```bash
# From the cat-detection directory
source bin/activate
```

##### On Windows:

```bash
# From the cat-detection directory
Scripts\activate
```

#### 3. Verify Installation

The virtual environment already has all dependencies installed. Verify with:

```bash
# Check Python version
python --version  # Should show Python 3.14

# Check installed packages
pip list

# Verify YOLO is installed
yolo version
```

### Key Installed Tools

The project includes:

- **Ultralytics YOLOv8**: Object detection framework
- **FiftyOne**: Dataset management and visualization
- **Jupyter**: Interactive notebooks
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision library
- **Additional tools**: See [requirements.txt](requirements.txt)

### Fresh Installation (Optional)

If you need to set up a new environment:

```bash
# Create new virtual environment
python3 -m venv cat-detection-env

# Activate it
source cat-detection-env/bin/activate  # macOS/Linux
# or
cat-detection-env\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📖 Usage

### 1. Download Dataset

```bash
# Run the export script from FiftyOne
python scripts/export_from_fiftyone.py
```

This will download the Open Images dataset and organize it into the appropriate structure.

### 2. Data Preprocessing

#### Option A: Interactive Notebook (Recommended)

Open the Jupyter notebook:

```bash
jupyter notebook Image_+_AI_Group_11_Cat_Detection.ipynb
```

Run the preprocessing cells to:

- Verify annotations
- Remove duplicates
- Analyze image quality
- Apply enhancements

#### Option B: Command Line Scripts

```bash
# Check labels and annotations
python scripts/check_labels.py

# Find invalid images
python scripts/find_invalid_images.py

# Enhance images (brightness correction and sharpening)
python scripts/enhance_images.py

# Visualize specific image with annotations
python scripts/visualize_single_image.py
```

### 3. Train Model

```bash
# Navigate to cat-detection directory
cd cat-detection

# Train YOLO model on original dataset
python training/train_yolo.py --data-dir ./data/original --epochs 100 --batch-size 16

# Train on enhanced dataset
python training/train_yolo.py --data-dir ./data/enhanced --epochs 100 --batch-size 16
```

For detailed training instructions, see [HOW_TO_TRAIN.md](HOW_TO_TRAIN.md).

### 4. Evaluate Model

```bash
# Evaluate using YOLOv8 validation
yolo val model=weights/best.pt data=data.yaml

# Or use the trained model for validation
yolo val model=runs/train/cat_detection/weights/best.pt data=data.yaml
```

### 5. Run Inference

```bash
# Detect cats in new images
yolo predict model=weights/best.pt source=path/to/images

# Run on webcam
yolo predict model=weights/best.pt source=0

# Run on video
yolo predict model=weights/best.pt source=path/to/video.mp4
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
├── original/              # Original dataset
│   ├── train/
│   │   ├── images/       # 1502 training images
│   │   └── labels/       # 1502 training labels
│   ├── val/
│   │   ├── images/       # 300 validation images
│   │   └── labels/       # 300 validation labels
│   └── test/
│       ├── images/       # 300 test images
│       └── labels/       # 300 test labels
└── enhanced/             # Enhanced dataset (preprocessed)
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

### Primary Model: YOLOv8

This project uses **YOLOv8** (You Only Look Once version 8) from Ultralytics.

**Why YOLOv8?**

- **Fast inference speed**: Real-time detection capability
- **High accuracy**: State-of-the-art performance on object detection
- **Easy to train and deploy**: Simple API and CLI interface
- **Well-documented**: Extensive documentation and community support
- **Active development**: Regular updates and improvements

### Model Configuration

- **Input Size**: 640x640 pixels
- **Architecture**: YOLOv8n (nano) - lightweight and fast
- **Pretrained Weights**: COCO dataset pretrained ([yolov8n.pt](yolov8n.pt))
- **Optimizer**: Adam
- **Learning Rate**: 0.001 with cosine decay
- **Batch Size**: 16 (configurable)
- **Epochs**: 100 (configurable)
- **Image Augmentation**: Built-in YOLOv8 augmentations
  - Random scaling
  - Random cropping
  - Color jittering
  - Mosaic augmentation

### Training Features

- Single-class detection (Cat only)
- Transfer learning from COCO pretrained weights
- Automatic mixed precision (AMP) training
- TensorBoard logging
- Model checkpointing (best and last weights)
- Validation during training

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
├── README.md                            # Project documentation
├── HOW_TO_TRAIN.md                     # Detailed training guide
├── requirements.txt                     # Python dependencies
├── pyvenv.cfg                          # Virtual environment config
├── data.yaml                           # YOLOv8 dataset configuration
├── dataset.yaml                        # Dataset metadata
├── yolov8n.pt                         # YOLOv8 nano pretrained weights
├── .gitignore                         # Git ignore rules
│
├── 📁 bin/                            # Python virtual environment binaries
│   ├── python, python3               # Python interpreters
│   ├── yolo                          # Ultralytics YOLO CLI
│   ├── jupyter                       # Jupyter notebook
│   └── [other tools]                # Various installed tools
│
├── 📁 data/                           # Dataset directory
│   ├── original/                    # Original dataset (1502 train, 300 val, 300 test)
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/
│   │       └── labels/
│   └── enhanced/                    # Enhanced dataset (preprocessed images)
│       ├── train/
│       ├── val/
│       └── test/
│
├── 📁 scripts/                        # Utility scripts
│   ├── check_labels.py              # Label verification
│   ├── enhance_images.py            # Image enhancement (brightness/sharpening)
│   ├── export_from_fiftyone.py      # Dataset download from Open Images
│   ├── find_invalid_images.py       # Find corrupted/invalid images
│   └── visualize_single_image.py    # Visualize annotations
│
├── 📁 training/                       # Training scripts
│   └── train_yolo.py                # YOLOv8 training script
│
├── 📁 weights/                        # Model weights (empty - generated after training)
│   ├── best.pt                      # Best model checkpoint (after training)
│   └── last.pt                      # Last model checkpoint (after training)
│
├── 📁 results/                        # Analysis and validation results
│   ├── invalid_images/              # Corrupted image reports
│   ├── label_check/                 # Label validation results
│   └── visualize_*.jpg              # Visualization outputs
│
├── 📁 report/                         # Project report documentation
│   └── README.md                    # Report structure guide
│
├── 📁 runs/                           # Training runs (auto-generated by YOLOv8)
│   └── train/                       # Training run outputs
│       └── cat_detection_*/         # Individual run results
│           ├── weights/             # Model checkpoints
│           ├── results.png          # Training metrics plot
│           ├── confusion_matrix.png # Confusion matrix
│           └── [other metrics]      # Performance visualizations
│
├── 📁 lib/                           # Python libraries (virtual environment)
├── 📁 share/                         # Shared resources (virtual environment)
└── 📁 etc/                           # Configuration files (virtual environment)
```

### Key Files

- **[Image\_+_AI_Group_11_Cat_Detection.ipynb](../Image_+_AI_Group_11_Cat_Detection.ipynb)**: Main Jupyter notebook (located in parent directory)
- **[CONTRIBUTING.md](../CONTRIBUTING.md)**: Contribution guidelines (parent directory)
- **[IMPLEMENTATION_GUIDE.md](../IMPLEMENTATION_GUIDE.md)**: Implementation details (parent directory)
- **[PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md)**: Full structure documentation (parent directory)
- **data.yaml**: YOLOv8 dataset configuration file
- **requirements.txt**: All Python package dependencies

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

### Quick Contribution Guide

1. **Fork the Repository** on GitHub
2. **Clone Your Fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cat-detection.git
   cd cat-detection
   ```
3. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make Your Changes**: Write clean, documented code following PEP 8
5. **Commit Your Changes**:
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```
6. **Push to Your Fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request** on GitHub

### Areas for Contribution

- 🐛 **Bug Fixes**: Report and fix bugs
- ✨ **New Features**: Add support for more models or datasets
- 📝 **Documentation**: Improve documentation and examples
- 🧪 **Testing**: Add unit tests and integration tests
- 🎨 **Visualization**: Enhance result visualization tools
- ⚡ **Performance**: Optimize training and inference speed

For detailed guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).

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

- [ ] Complete model training on both original and enhanced datasets
- [ ] Generate comprehensive performance comparison report
- [ ] Extend to multi-class detection (cats, dogs, birds, etc.)
- [ ] Implement real-time detection on video streams
- [ ] Add mobile deployment support (TensorFlow Lite/ONNX)
- [ ] Experiment with larger YOLOv8 models (YOLOv8s, YOLOv8m, YOLOv8l)
- [ ] Add advanced data augmentation techniques (CutMix, Mosaic)
- [ ] Create web interface for easy inference
- [ ] Implement active learning for continuous improvement
- [ ] Add hyperparameter tuning automation
- [ ] Deploy model as REST API

---

## 📚 Additional Documentation

- **[HOW_TO_TRAIN.md](HOW_TO_TRAIN.md)**: Detailed training instructions and troubleshooting
- **[CONTRIBUTING.md](../CONTRIBUTING.md)**: Contribution guidelines
- **[PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md)**: Complete project structure documentation
- **[report/README.md](report/README.md)**: Project report structure

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
