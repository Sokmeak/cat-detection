import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import json
import os
import shutil

# Configure page
st.set_page_config(page_title="Cat Detection Label Viewer", layout="wide")

# Session state initialization
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'incorrect_images' not in st.session_state:
    st.session_state.incorrect_images = set()
if 'flagged_file' not in st.session_state:
    st.session_state.flagged_file = Path("../flagged_images.txt")
if 'dataset_type' not in st.session_state:
    st.session_state.dataset_type = 'train'

# Load existing flagged images
if st.session_state.flagged_file.exists():
    with open(st.session_state.flagged_file, 'r') as f:
        st.session_state.incorrect_images = set(f.read().strip().split('\n'))

def read_yolo_label(label_path):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append([float(x) for x in parts])
    return boxes

def draw_boxes(img, boxes):
    h, w = img.shape[:2]
    for i, (cls, x, y, bw, bh) in enumerate(boxes):
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            img, f"Cat {i+1}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )
    return img

def parse_label_text(text):
    """Parse label text into boxes"""
    boxes = []
    for line in text.strip().split('\n'):
        if line.strip():
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    boxes.append([float(x) for x in parts])
                except ValueError:
                    pass
    return boxes

def save_label_file(label_path, text):
    """Save label content to file"""
    with open(label_path, 'w') as f:
        f.write(text.strip())
    if not text.strip():
        # If empty, delete the file
        if label_path.exists():
            label_path.unlink()

def move_to_issued(img_path, dataset_type):
    """Move image to issued-images folder"""
    # Create issued-images directory structure
    issued_dir = Path("../issued-images") / dataset_type
    issued_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the image
    dest_path = issued_dir / img_path.name
    shutil.copy2(img_path, dest_path)
    return dest_path

# Load image paths
dataset_type = st.sidebar.selectbox(
    "📁 Dataset Type",
    options=['train', 'test', 'val'],
    index=['train', 'test', 'val'].index(st.session_state.dataset_type),
    key='dataset_selector'
)

# Update session state and reset index if dataset changed
if dataset_type != st.session_state.dataset_type:
    st.session_state.dataset_type = dataset_type
    st.session_state.current_index = 0

img_dir = Path(f"../data/original/{dataset_type}/images")
label_dir = Path(f"../data/original/{dataset_type}/labels")

# Use os.listdir to handle special characters in filenames
try:
    all_files = os.listdir(img_dir)
    image_paths = [img_dir / f for f in all_files if f.lower().endswith('.jpg')]
    # Sort alphabetically by cleaned filename (remove special chars for sorting)
    image_paths = sorted(image_paths, key=lambda p: p.name.replace('\n', '').replace('\r', ''))
except Exception as e:
    st.error(f"Error loading images: {e}")
    image_paths = []

if not image_paths:
    st.error("No images found! Check the path.")
    st.stop()

# Header
st.title("🐱 Cat Detection Label Viewer")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    total_images = len(image_paths)
    
    # Jump to image
    jump_to = st.number_input(
        "Jump to image #", 
        min_value=1, 
        max_value=total_images, 
        value=st.session_state.current_index + 1
    )
    if st.button("Go"):
        st.session_state.current_index = jump_to - 1
        st.rerun()
    
    st.markdown("---")
    
    # Statistics
    st.metric("Total Images", total_images)
    st.metric("Flagged as Incorrect", len(st.session_state.incorrect_images))
    st.metric("Progress", f"{st.session_state.current_index + 1}/{total_images}")
    
    st.markdown("---")
    
    # Export options
    st.header("Export Flagged Images")
    if st.session_state.incorrect_images:
        incorrect_list = '\n'.join(sorted(st.session_state.incorrect_images))
        st.download_button(
            "📥 Download List",
            incorrect_list,
            file_name="incorrect_images.txt",
            mime="text/plain"
        )
        
        st.text_area(
            "Copy Names", 
            incorrect_list, 
            height=200,
            help="Copy these names to clipboard"
        )
        
        if st.button("🗑️ Clear All Flags"):
            st.session_state.incorrect_images.clear()
            if st.session_state.flagged_file.exists():
                st.session_state.flagged_file.unlink()
            st.rerun()
    else:
        st.info("No images flagged yet")

# Main content
current_path = image_paths[st.session_state.current_index]
# Keep the original filename as-is
img_name = current_path.name
# Create label path - remove any special characters for label matching
label_stem = img_name.replace('\n', '').replace('\r', '').replace('.jpg', '').replace('.JPG', '')
label_path = label_dir / f"{label_stem}.txt"
label_name = label_path.name

# Initialize edit mode state
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'preview_mode' not in st.session_state:
    st.session_state.preview_mode = False
if 'edited_label_text' not in st.session_state:
    st.session_state.edited_label_text = ""
if 'issued_images' not in st.session_state:
    st.session_state.issued_images = set()

# Read label file content
label_content = ""
if label_path.exists():
    with open(label_path, 'r') as f:
        label_content = f.read()

# Display current image info in compact format
col_info1, col_info2, col_info3 = st.columns([1, 2, 1])
with col_info2:
    col_name, col_status = st.columns([3, 1])
    with col_name:
        st.markdown(f"**Image:** `{img_name}`")
        st.caption(f"Label: {label_name}")
    with col_status:
        is_flagged = img_name in st.session_state.incorrect_images
        if is_flagged:
            st.error("⚠️ Flagged")
        else:
            st.success("✅ OK")

# Load and display image
img = cv2.imread(str(current_path))
if img is None:
    st.error(f"Could not load image: {img_name}")
else:
    # Determine which label to use for display
    if st.session_state.preview_mode and st.session_state.edited_label_text:
        boxes = parse_label_text(st.session_state.edited_label_text)
        preview_banner = "🔍 PREVIEW MODE - Showing edited labels"
    else:
        boxes = read_yolo_label(label_path)
        preview_banner = None
    
    img_with_boxes = draw_boxes(img.copy(), boxes)
    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    
    # Main content area with image and editor side by side
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        # Show preview banner if in preview mode
        if preview_banner:
            st.warning(preview_banner)
        st.info(f"📊 {img.shape[1]}x{img.shape[0]} | Boxes: {len(boxes)}")
        st.image(img_rgb, use_container_width=True)
    
    with col_right:
        st.markdown("### 📝 Label Editor")
        
        if not st.session_state.edit_mode:
            st.code(label_content if label_content else "No labels", language=None, height=300)
            if st.button("✏️ Edit Labels", use_container_width=True):
                st.session_state.edit_mode = True
                st.session_state.edited_label_text = label_content
                st.rerun()
        else:
            # Initialize edited text on first edit
            if not st.session_state.edited_label_text and not st.session_state.preview_mode:
                st.session_state.edited_label_text = label_content
            
            edited_text = st.text_area(
                "YOLO format: class x y width height",
                value=st.session_state.edited_label_text if st.session_state.edited_label_text else label_content,
                height=250,
                label_visibility="collapsed"
            )
            st.session_state.edited_label_text = edited_text
            
            # Edit action buttons in 2 rows
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                if st.button("👁️ Preview", use_container_width=True):
                    st.session_state.preview_mode = True
                    st.rerun()
            with col_e2:
                if st.button("💾 Save", use_container_width=True):
                    save_label_file(label_path, st.session_state.edited_label_text)
                    st.session_state.edit_mode = False
                    st.session_state.preview_mode = False
                    st.session_state.edited_label_text = ""
                    st.success("✅ Saved!")
                    st.rerun()
            
            col_e3, col_e4 = st.columns(2)
            with col_e3:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.edit_mode = False
                    st.session_state.preview_mode = False
                    st.session_state.edited_label_text = ""
                    st.rerun()
            with col_e4:
                if st.button("🗑️ Clear All", use_container_width=True):
                    st.session_state.edited_label_text = ""
                    st.rerun()

# Navigation and action buttons
col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])

with col1:
    if st.button("⬅️ Previous", use_container_width=True):
        if st.session_state.current_index > 0:
            st.session_state.current_index -= 1
            st.rerun()

with col2:
    if st.button("➡️ Next", use_container_width=True):
        if st.session_state.current_index < len(image_paths) - 1:
            st.session_state.current_index += 1
            st.rerun()

with col3:
    if is_flagged:
        if st.button("✅ Mark as Correct", use_container_width=True):
            st.session_state.incorrect_images.discard(img_name)
            # Save to file
            with open(st.session_state.flagged_file, 'w') as f:
                f.write('\n'.join(sorted(st.session_state.incorrect_images)))
            st.rerun()
    else:
        if st.button("❌ Flag as Incorrect", use_container_width=True):
            st.session_state.incorrect_images.add(img_name)
            # Save to file
            with open(st.session_state.flagged_file, 'w') as f:
                f.write('\n'.join(sorted(st.session_state.incorrect_images)))
            st.rerun()

with col4:
    if st.button("⏩ Next + Flag", use_container_width=True):
        st.session_state.incorrect_images.add(img_name)
        with open(st.session_state.flagged_file, 'w') as f:
            f.write('\n'.join(sorted(st.session_state.incorrect_images)))
        if st.session_state.current_index < len(image_paths) - 1:
            st.session_state.current_index += 1
        st.rerun()

with col5:
    if st.button("📋 Copy Name", use_container_width=True):
        st.code(img_name, language=None)
        st.success("Image name shown above!")

with col6:
    is_issued = img_name in st.session_state.issued_images
    if is_issued:
        st.success("✅ Issued")
    else:
        if st.button("📌 Make Issued", use_container_width=True):
            try:
                dest = move_to_issued(current_path, st.session_state.dataset_type)
                st.session_state.issued_images.add(img_name)
                st.success(f"Moved to issued-images!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
