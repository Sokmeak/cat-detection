"""
Model Utilities
===============
Functions for loading models and running inference
"""

import os
import streamlit as st
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    st.error("ultralytics not installed. Run: pip install ultralytics")
    st.stop()


@st.cache_resource
def load_model(model_path):
    """
    Load YOLO model with caching
    
    Args:
        model_path: Path to the model file
        
    Returns:
        YOLO model instance or None if loading fails
    """
    try:
        if not os.path.exists(model_path):
            return None
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def run_detection(model, image, conf_threshold=0.25):
    """
    Run cat detection on image
    
    Args:
        model: YOLO model instance
        image: Input image (numpy array)
        conf_threshold: Confidence threshold for detections
        
    Returns:
        Tuple of (annotated_image, detections_list)
    """
    try:
        # Run inference
        results = model(image, conf=conf_threshold, verbose=False)
        
        # Get detection results
        result = results[0]
        boxes = result.boxes
        
        # Get annotated image
        annotated_image = result.plot()
        
        # Extract detection info
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf,
                'class': cls
            })
        
        return annotated_image, detections
    except Exception as e:
        st.error(f"Error during detection: {e}")
        return None, []
