"""
Detection View Components
=========================
Components for image upload and camera detection views
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image

from utils.model_utils import run_detection
from components.ui_components import (
    render_section_header,
    render_detection_result,
    render_detection_details
)


def render_upload_tab(model, conf_threshold):
    """
    Render the image upload tab
    
    Args:
        model: YOLO model instance
        conf_threshold: Confidence threshold for detection
    """
    render_section_header("Upload Image for Detection")
    
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, PNG, BMP)",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        process_image(uploaded_file, model, conf_threshold)


def render_camera_tab(model, conf_threshold):
    """
    Render the camera capture tab
    
    Args:
        model: YOLO model instance
        conf_threshold: Confidence threshold for detection
    """
    render_section_header("Camera Detection")
    st.info("📸 Capture an image using your webcam")
    
    camera_image = st.camera_input("Camera", label_visibility="collapsed")
    
    if camera_image is not None:
        process_image(camera_image, model, conf_threshold)


def process_image(image_source, model, conf_threshold):
    """
    Process and display image with detection results
    
    Args:
        image_source: Image file or camera input
        model: YOLO model instance
        conf_threshold: Confidence threshold for detection
    """
    # Load image
    image = Image.open(image_source)
    image_np = np.array(image)
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("**Original Image**")
        st.image(image, use_container_width=True)
    
    # Run detection
    with st.spinner("Analyzing image..."):
        annotated_image, detections = run_detection(model, image_np, conf_threshold)
    
    with col2:
        st.markdown("**Detection Result**")
        if annotated_image is not None:
            # Convert BGR to RGB for display
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_image_rgb, use_container_width=True)
    
    # Display results
    num_cats = len(detections)
    render_detection_result(num_cats)
    
    if num_cats > 0:
        render_detection_details(detections)
    else:
        st.info("Try adjusting the confidence threshold or use a different image")
