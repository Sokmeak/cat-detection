"""
UI Components
=============
Reusable UI components for the application
"""

import streamlit as st

# Material Icons CSS
MATERIAL_ICONS = '<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">'


def render_header():
    """Render the main page header"""
    st.markdown(MATERIAL_ICONS, unsafe_allow_html=True)
    st.markdown(
        '<h1 class="main-header"><span class="material-icons" style="vertical-align: middle; font-size: 1.2em;">search</span> Cat Detector</h1>', 
        unsafe_allow_html=True
    )


def render_section_header(title):
    """Render a section header"""
    st.markdown(
        f'<p class="section-header">{title}</p>', 
        unsafe_allow_html=True
    )


def render_detection_result(num_cats):
    """
    Render detection result message
    
    Args:
        num_cats: Number of cats detected
    """
    if num_cats > 0:
        st.markdown(
            f'<div class="detection-result cat-detected"><span class="material-icons" style="vertical-align: middle;">check_circle</span> Detected {num_cats} cat{"s" if num_cats > 1 else ""}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="detection-result no-cat"><span class="material-icons" style="vertical-align: middle;">info</span> No cats detected in this image</div>',
            unsafe_allow_html=True
        )


def render_detection_details(detections):
    """
    Render detailed detection information
    
    Args:
        detections: List of detection dictionaries
    """
    render_section_header("Detection Details")
    
    num_cats = len(detections)
    
    # Create grid layout based on number of detections
    if num_cats == 1:
        cols = [st.container()]
    elif num_cats == 2:
        cols = st.columns(2)
    else:
        cols = st.columns(3)
    
    for i, det in enumerate(detections):
        col_idx = i % len(cols)
        with cols[col_idx]:
            st.markdown(f'<span class="material-icons" style="font-size: 1.5em;">pets</span>', unsafe_allow_html=True)
            st.metric(
                f"Cat #{i+1}",
                f"{det['confidence']:.1%}",
                delta=None
            )
            bbox = det['bbox']
            st.caption(f"📍 Position: ({bbox[0]}, {bbox[1]})")
            st.caption(f"📏 Size: {bbox[2]-bbox[0]}×{bbox[3]-bbox[1]}px")


def render_sidebar_config():
    """Render sidebar configuration section"""
    st.sidebar.markdown(MATERIAL_ICONS, unsafe_allow_html=True)
    st.sidebar.markdown('### <span class="material-icons" style="vertical-align: middle;">settings</span> Configuration', unsafe_allow_html=True)
    st.sidebar.markdown("---")


def render_model_selector(available_models):
    """
    Render model selection dropdown with model information
    
    Args:
        available_models: Dictionary of available models
        
    Returns:
        Tuple of (model_name, model_path)
    """
    from utils.file_utils import get_model_info, load_test_metrics
    
    st.sidebar.markdown("**Model Selection**")
    model_name = st.sidebar.selectbox(
        "Choose Model",
        options=list(available_models.keys()),
        help="Select trained model for detection",
        label_visibility="collapsed"
    )
    
    model_path = available_models[model_name]
    
    # Display model info
    model_info = get_model_info(model_path)
    test_metrics = load_test_metrics(model_path)
    
    st.sidebar.markdown(f"**Active Model:** `{model_name}`")
    
    # Show test metrics if available
    if test_metrics:
        with st.sidebar.expander("🎯 Test Set Performance", expanded=True):
            st.caption("_Final evaluation on unseen data_")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("mAP@50", f"{test_metrics['map50']:.1%}")
                st.metric("Precision", f"{test_metrics['precision']:.1%}")
            with col2:
                st.metric("mAP@50-95", f"{test_metrics['map50_95']:.1%}")
                st.metric("Recall", f"{test_metrics['recall']:.1%}")
    
    # Show training/validation metrics
    if model_info:
        with st.sidebar.expander("📊 Training Details", expanded=False):
            if 'epochs_trained' in model_info:
                st.metric("Epochs", model_info['epochs_trained'])
            
            if 'final_map50' in model_info:
                st.caption("_Validation set metrics_")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Val mAP@50", f"{model_info['final_map50']:.2%}")
                with col2:
                    st.metric("Val mAP@50-95", f"{model_info['final_map50_95']:.2%}")
            
            if 'precision' in model_info and 'recall' in model_info:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Val Precision", f"{model_info['precision']:.2%}")
                with col2:
                    st.metric("Val Recall", f"{model_info['recall']:.2%}")
            
            if 'size_mb' in model_info:
                st.caption(f"Model size: {model_info['size_mb']} MB")
    
    st.sidebar.markdown("---")
    
    return model_name, model_path


def render_confidence_slider(default_value=0.25):
    """
    Render confidence threshold slider
    
    Args:
        default_value: Default confidence value
        
    Returns:
        Selected confidence threshold
    """
    st.sidebar.markdown("**Detection Threshold**")
    conf_threshold = st.sidebar.slider(
        "Confidence",
        min_value=0.0,
        max_value=1.0,
        value=default_value,
        step=0.05,
        help="Minimum confidence score for detections",
        label_visibility="collapsed"
    )
    st.sidebar.caption(f"Current: {conf_threshold:.0%}")
    
    return conf_threshold


def render_no_models_error():
    """Render error message when no models are found"""
    st.markdown(MATERIAL_ICONS, unsafe_allow_html=True)
    st.error("⚠️ No trained models found")
    st.info("""
    **Expected model locations:**
    - `runs/train/cat_original/weights/best.pt`
    - `runs/train/cat_enhanced/weights/best.pt`
    
    **Run from project root:**
    ```bash
    cd ..
    streamlit run app/cat_detection_app.py
    ```
    
    **Or train models first:**
    ```bash
    python training/train_yolo.py --data-dir ./data/original --name cat_original
    python training/train_yolo.py --data-dir ./data/enhanced --name cat_enhanced
    ```
    """)


# def render_info_tab(model_name, conf_threshold, available_models):
#     """
#     Render the information tab
    
#     Args:
#         model_name: Current model name
#         conf_threshold: Current confidence threshold
#         available_models: Dictionary of available models
#     """
#     render_section_header("System Information")
    
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.markdown("""
#         ### Cat Detection System
        
#         Deep learning-based cat detection using YOLOv8 architecture.
        
#         **Features:**
#         - Real-time object detection
#         - Multiple model support
#         - Adjustable detection threshold
#         - Image upload and webcam capture
        
#         **How to Use:**
#         1. Select a model from the sidebar
#         2. Set confidence threshold
#         3. Upload image or use camera
#         4. View detection results
#         """)
    
#     with col2:
#         st.markdown("**Current Configuration**")
#         st.markdown(f"**Model:** {model_name}")
#         st.markdown(f"**Threshold:** {conf_threshold:.0%}")
#         st.markdown("---")
#         st.markdown("**Technology Stack**")
#         st.markdown("- YOLOv8")
#         st.markdown("- Streamlit")
#         st.markdown("- OpenCV")
#         st.markdown("- PyTorch")
    
#     st.markdown("---")
    
#     render_section_header("Available Models")
    
#     for name, path in available_models.items():
#         with st.expander(f"📦 {name}"):
#             st.code(path, language="text")
    
#     st.markdown("---")
#     st.caption("For training documentation, see TRAINING_GUIDE.md")
