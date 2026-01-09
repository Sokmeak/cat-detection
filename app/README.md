# 🐱 Cat Detection Streamlit App

A user-friendly web application for detecting cats in images using YOLOv8 deep learning models.

## Features

- 📤 **Upload Images**: Detect cats in uploaded images
- 📷 **Webcam Capture**: Real-time cat detection using your webcam
- 🎯 **Adjustable Confidence**: Control detection sensitivity with a slider
- 🔄 **Multiple Models**: Switch between different trained models (original/enhanced)
- 📊 **Detection Details**: View bounding boxes, confidence scores, and positions

## Installation

### 1. Install Dependencies

```bash
# Navigate to the app directory
cd app

# Install required packages
pip install -r requirements.txt
```

Or install from the main project requirements:

```bash
cd ..
pip install -r requirements.txt
```

### 2. Verify Model Availability

The app will automatically detect trained models from:

- `../runs/train/*/weights/best.pt` (your trained models)
- `../yolov8n.pt`, `../yolov8s.pt` (default pretrained models)

Make sure you have at least one trained model before running the app.

## Usage

### Running the App

From the `app` directory:

```bash
streamlit run cat_detection_app.py
```

Or from the project root:

```bash
streamlit run app/cat_detection_app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Using the App

1. **Select Model**: Choose a trained model from the sidebar dropdown
2. **Adjust Confidence**: Use the slider to set the minimum confidence threshold (default: 0.25)
3. **Choose Input Method**:
   - **Upload Image Tab**: Upload a JPG, JPEG, PNG, or BMP file
   - **Webcam Capture Tab**: Take a photo using your webcam
4. **View Results**: See the detection results with bounding boxes and confidence scores

## App Structure

```
app/
├── cat_detection_app.py    # Main Streamlit application
├── requirements.txt         # App-specific dependencies
├── README.md               # This file
└── .streamlit/             # Streamlit configuration (optional)
    └── config.toml         # Theme and settings
```

## Configuration

### Custom Theme (Optional)

Create `.streamlit/config.toml` for custom styling:

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
```

## Features Explained

### Upload Image Tab

- Drag and drop or browse to upload images
- Supports: JPG, JPEG, PNG, BMP formats
- Displays original image side-by-side with detection results
- Shows number of cats detected and confidence scores

### Webcam Capture Tab

- Real-time camera access (requires permission)
- Capture button to take photos
- Instant cat detection on captured images
- Perfect for testing with live subjects

### About Tab

- Application information and features
- Current model details
- List of all available models
- Links to documentation

## Troubleshooting

### No Models Found

```
❌ No trained models found! Please train a model first.
```

**Solution**: Train a model first using the training script:

```bash
cd ..
python training/train_yolo.py --data-dir ./data --model-size s --epochs 100
```

### Camera Not Working

```
Camera access denied or not available
```

**Solutions**:

- Allow camera permissions in your browser
- Check if another application is using the camera
- Try a different browser (Chrome/Firefox recommended)

### Model Loading Error

```
❌ Failed to load model from: <path>
```

**Solutions**:

- Verify the model file exists at the specified path
- Check if the model file is corrupted (re-train if needed)
- Ensure PyTorch and ultralytics are properly installed

### Slow Performance

**Solutions**:

- Use a smaller model (YOLOv8n instead of YOLOv8m/l)
- Reduce image size before uploading
- Close other resource-intensive applications
- Use GPU if available (check with `torch.cuda.is_available()`)

## Performance Tips

1. **Model Selection**:

   - **YOLOv8n**: Fastest, good for quick testing
   - **YOLOv8s**: Balanced speed and accuracy
   - **YOLOv8m/l**: Slower but more accurate

2. **Confidence Threshold**:

   - Lower (0.15-0.25): More detections, possible false positives
   - Medium (0.25-0.35): Balanced results (recommended)
   - Higher (0.35+): Fewer but more confident detections

3. **Image Size**:
   - Smaller images process faster
   - Larger images may provide better detection for small cats

## Development

### Running in Development Mode

```bash
# Enable auto-reload on file changes
streamlit run cat_detection_app.py --server.runOnSave true
```

### Debugging

```bash
# Run with verbose logging
streamlit run cat_detection_app.py --logger.level=debug
```

## Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

**Note**: Make sure to include trained model weights or download them during deployment.

### Deploy Locally on Network

```bash
# Allow access from other devices on your network
streamlit run cat_detection_app.py --server.address=0.0.0.0
```

Access from other devices: `http://<your-ip>:8501`

## Requirements

- Python 3.8+
- Streamlit 1.29+
- YOLOv8 (ultralytics)
- PyTorch 2.0+
- OpenCV
- PIL/Pillow
- NumPy

See [requirements.txt](requirements.txt) for complete list.

## Related Documentation

- **Training Guide**: `../TRAINING_GUIDE.md` - How to train models
- **Implementation Guide**: `../IMPLEMENTATION_GUIDE.md` - Project setup
- **Main README**: `../README.md` - Project overview

## License

This application is part of the Cat Detection project.

## Support

For issues or questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the training guide for model-related issues
3. Check Streamlit documentation for app-specific issues

---

**Happy Cat Detecting! 🐱**
