"""
Cat Detection Streamlit Application
====================================
Main application file
"""

import streamlit as st

# Import configuration
from config.styles import CSS_STYLES
from config.constants import PAGE_TITLE, PAGE_ICON, LAYOUT

# Import utilities
from utils.model_utils import load_model
from utils.file_utils import find_model_weights

# Import UI components
from components.ui_components import (
    render_header,
    render_sidebar_config,
    render_model_selector,
    render_confidence_slider,
    render_no_models_error
)
from components.detection_view import render_upload_tab, render_camera_tab


def setup_page():
    """Configure page settings and apply styles"""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state="expanded"
    )
    st.markdown(CSS_STYLES, unsafe_allow_html=True)


def main():
    """Main application entry point"""
    # Setup
    setup_page()
    render_header()
    
    # Sidebar configuration
    render_sidebar_config()
    
    # Find available models
    available_models = find_model_weights()
    
    if not available_models:
        render_no_models_error()
        st.stop()
    
    # Model selection
    model_name, model_path = render_model_selector(available_models)
    
    # Confidence threshold
    conf_threshold = render_confidence_slider()
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(model_path)
    
    if model is None:
        st.error(f"Failed to load model: {model_path}")
        st.stop()
    
    st.sidebar.markdown("---")
    st.sidebar.success("✓ Model Ready")
    
    # Main content tabs
    tab1, tab2 = st.tabs(["Upload Image", "Camera"])
    
    with tab1:
        render_upload_tab(model, conf_threshold)
    
    with tab2:
        render_camera_tab(model, conf_threshold)
    
    # with tab3:
    #     render_info_tab(model_name, conf_threshold, available_models)


if __name__ == "__main__":
    main()
