import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import json

# Configure page
st.set_page_config(page_title="Cat Detection Label Viewer", layout="wide")

# Session state initialization
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'incorrect_images' not in st.session_state:
    st.session_state.incorrect_images = set()
if 'flagged_file' not in st.session_state:
    st.session_state.flagged_file = Path("../flagged_images.txt")

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

# Load image paths
img_dir = Path("../data/original/train/images")
label_dir = Path("../data/original/train/labels")
image_paths = sorted(img_dir.glob("*.jpg"))

if not image_paths:
    st.error("No images found! Check the path.")
    st.stop()

# Header
st.title("🐱 Cat Detection Label Viewer")
st.markdown("---")

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
label_path = label_dir / f"{current_path.stem}.txt"
img_name = current_path.name

# Display current image info
col1, col2, col3 = st.columns([2, 3, 2])
with col2:
    st.subheader(f"Image: {img_name}")
    is_flagged = img_name in st.session_state.incorrect_images
    if is_flagged:
        st.error("⚠️ Flagged as Incorrect")
    else:
        st.success("✅ OK")

# Load and display image
img = cv2.imread(str(current_path))
if img is None:
    st.error(f"Could not load image: {img_name}")
else:
    boxes = read_yolo_label(label_path)
    img_with_boxes = draw_boxes(img.copy(), boxes)
    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    
    # Show image info
    st.info(f"📊 Resolution: {img.shape[1]}x{img.shape[0]} | Boxes: {len(boxes)}")
    
    # Display image
    st.image(img_rgb, use_container_width=True)

# Action buttons
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

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



# Footer
st.markdown("---")
st.caption(f"Viewing image {st.session_state.current_index + 1} of {len(image_paths)}")
