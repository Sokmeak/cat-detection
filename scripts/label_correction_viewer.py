"""
Label Correction Viewer - For fixing flagged images
"""
from pathlib import Path
import cv2
import streamlit as st
import os

# Configure page
st.set_page_config(page_title="Label Correction Tool", layout="wide")

# Session state initialization
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'corrected_count' not in st.session_state:
    st.session_state.corrected_count = 0
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'preview_mode' not in st.session_state:
    st.session_state.preview_mode = False
if 'edited_label_text' not in st.session_state:
    st.session_state.edited_label_text = ""

def read_yolo_label(label_path):
    """Read YOLO format label file."""
    boxes = []
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 5:
                    try:
                        box = [float(x) for x in parts]
                        boxes.append(box)
                    except ValueError:
                        pass
    return boxes

def draw_boxes(img, boxes):
    """Draw bounding boxes on image."""
    h, w = img.shape[:2]
    img_copy = img.copy()
    
    for i, (cls, x, y, bw, bh) in enumerate(boxes):
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, f"Cat {i+1}", (x1, max(y1 - 5, 0)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return img_copy

def parse_label_text(text):
    """Parse label text into boxes."""
    boxes = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) == 5:
                try:
                    box = [float(x) for x in parts]
                    boxes.append(box)
                except ValueError:
                    pass
    return boxes

def save_label_file(label_path, text):
    """Save label text to file."""
    try:
        with open(label_path, 'w') as f:
            f.write(text.strip())
            if text.strip():
                f.write('\n')
        return True
    except Exception as e:
        st.error(f"Error saving label: {e}")
        return False

def validate_bbox(bbox):
    """Validate bounding box coordinates."""
    class_id, x_center, y_center, width, height = bbox
    
    errors = []
    
    if class_id != 0:
        errors.append(f"Class ID should be 0, got {class_id}")
    
    if not (0 <= x_center <= 1):
        errors.append(f"x_center {x_center:.4f} out of range [0,1]")
    
    if not (0 <= y_center <= 1):
        errors.append(f"y_center {y_center:.4f} out of range [0,1]")
    
    if not (0 < width <= 1):
        errors.append(f"width {width:.4f} out of range (0,1]")
    
    if not (0 < height <= 1):
        errors.append(f"height {height:.4f} out of range (0,1]")
    
    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2
    
    if x_min < 0:
        errors.append(f"Box extends left: x_min={x_min:.4f}")
    if x_max > 1:
        errors.append(f"Box extends right: x_max={x_max:.4f}")
    if y_min < 0:
        errors.append(f"Box extends top: y_min={y_min:.4f}")
    if y_max > 1:
        errors.append(f"Box extends bottom: y_max={y_max:.4f}")
    
    return len(errors) == 0, errors

# Load image paths
img_dir = Path("../data/needs_correction/images")
label_dir = Path("../data/needs_correction/labels")

try:
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_paths = [img_dir / f for f in sorted(image_files)]
except Exception as e:
    st.error(f"Error loading images: {e}")
    image_paths = []

if not image_paths:
    st.error("No images found in data/needs_correction/images/")
    st.stop()

# Header
st.title("🔧 Label Correction Tool")
st.markdown(f"**Correcting flagged images - {len(image_paths)} images to review**")

# Sidebar
with st.sidebar:
    st.header("📊 Progress")
    progress = st.session_state.current_index + 1
    st.metric("Current Image", f"{progress} / {len(image_paths)}")
    st.metric("Corrected", st.session_state.corrected_count)
    
    st.markdown("---")
    st.header("📍 Navigation")
    
    # Jump to image
    jump_to = st.number_input(
        "Jump to image:",
        min_value=1,
        max_value=len(image_paths),
        value=st.session_state.current_index + 1,
        step=1
    )
    
    if st.button("Go to Image"):
        st.session_state.current_index = jump_to - 1
        st.session_state.edit_mode = False
        st.session_state.preview_mode = False
        st.rerun()
    
    st.markdown("---")
    st.header("ℹ️ Instructions")
    st.markdown("""
    **How to Correct Labels:**
    
    1. **Review** the image and box
    2. Click **Edit Labels**
    3. **Modify** coordinates:
       ```
       0 x_center y_center width height
       ```
       - All values: 0.0 to 1.0
       - Class ID always: 0
    4. Click **Preview** to check
    5. Click **Save** to apply
    6. Use **Next** to continue
    
    **Common Fixes:**
    - Box too small → increase width/height
    - Box off-center → adjust x/y
    - Wrong position → recalculate all
    - No cat visible → delete the line
    """)

# Main content
current_path = image_paths[st.session_state.current_index]
img_name = current_path.name
label_stem = img_name.replace('.jpg', '').replace('.JPG', '').replace('.png', '')
label_path = label_dir / f"{label_stem}.txt"

# Display current image info
col_info1, col_info2, col_info3 = st.columns([1, 2, 1])
with col_info2:
    st.markdown(f"""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;'>
        <h3 style='margin: 0; color: #1f77b4;'>Image {st.session_state.current_index + 1} of {len(image_paths)}</h3>
        <p style='margin: 5px 0; font-size: 14px; color: #666;'>{img_name}</p>
        <p style='margin: 5px 0; font-size: 12px; color: #999;'>Label: {label_path.name}</p>
    </div>
    """, unsafe_allow_html=True)

# Read label file content
label_content = ""
if label_path.exists():
    with open(label_path, 'r') as f:
        label_content = f.read()

# Load and display image
img = cv2.imread(str(current_path))
if img is None:
    st.error(f"Cannot load image: {img_name}")
else:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Display image with boxes
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("🖼️ Image with Annotations")
        
        if st.session_state.preview_mode and st.session_state.edited_label_text:
            preview_boxes = parse_label_text(st.session_state.edited_label_text)
            preview_img = draw_boxes(img_rgb, preview_boxes)
            st.image(preview_img, use_container_width=True)
            st.info("👁️ Preview Mode - This is how it will look after saving")
        else:
            boxes = read_yolo_label(label_path)
            if boxes:
                annotated_img = draw_boxes(img_rgb, boxes)
                st.image(annotated_img, use_container_width=True)
            else:
                st.image(img_rgb, use_container_width=True)
                st.warning("⚠️ No annotations found")
        
        # Image info
        st.caption(f"Image size: {w} x {h} pixels")
    
    with col2:
        st.subheader("📝 Label Editor")
        
        if not st.session_state.edit_mode:
            # Display current labels
            boxes = read_yolo_label(label_path)
            
            if boxes:
                st.success(f"✅ Found {len(boxes)} box(es)")
                
                for i, bbox in enumerate(boxes):
                    is_valid, errors = validate_bbox(bbox)
                    
                    if is_valid:
                        st.markdown(f"""
                        **Box {i+1}:** ✅ Valid  
                        `{' '.join([f'{x:.4f}' for x in bbox])}`
                        """)
                    else:
                        st.error(f"**Box {i+1}:** ❌ Invalid")
                        st.code(' '.join([f'{x:.4f}' for x in bbox]))
                        for error in errors:
                            st.markdown(f"- {error}")
            else:
                st.warning("⚠️ No boxes in label file")
            
            st.markdown("---")
            st.markdown("**Current Label Content:**")
            if label_content:
                st.code(label_content, language='text')
            else:
                st.info("Empty label file")
            
            if st.button("✏️ Edit Labels", use_container_width=True):
                st.session_state.edit_mode = True
                st.session_state.edited_label_text = label_content
                st.session_state.preview_mode = False
                st.rerun()
        
        else:
            # Edit mode
            st.session_state.edited_label_text = st.text_area(
                "Edit YOLO Labels:",
                value=st.session_state.edited_label_text,
                height=200,
                help="Format: class_id x_center y_center width height (one per line)"
            )
            
            # Validate edited content
            if st.session_state.edited_label_text.strip():
                edited_boxes = parse_label_text(st.session_state.edited_label_text)
                
                if edited_boxes:
                    all_valid = True
                    for i, bbox in enumerate(edited_boxes):
                        is_valid, errors = validate_bbox(bbox)
                        if not is_valid:
                            all_valid = False
                            st.error(f"Box {i+1} has errors:")
                            for error in errors:
                                st.markdown(f"- {error}")
                    
                    if all_valid:
                        st.success(f"✅ All {len(edited_boxes)} box(es) are valid!")
                else:
                    st.warning("No valid boxes parsed")
            
            # Action buttons
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("👁️ Preview", use_container_width=True):
                    st.session_state.preview_mode = True
                    st.rerun()
            
            with col_b:
                if st.button("💾 Save", use_container_width=True):
                    if save_label_file(label_path, st.session_state.edited_label_text):
                        st.success("✅ Saved!")
                        st.session_state.corrected_count += 1
                        st.session_state.edit_mode = False
                        st.session_state.preview_mode = False
                        st.rerun()
            
            with col_c:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.edit_mode = False
                    st.session_state.preview_mode = False
                    st.rerun()

# Navigation and action buttons
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

with col1:
    if st.button("⬅️ Previous", disabled=st.session_state.current_index == 0, use_container_width=True):
        st.session_state.current_index -= 1
        st.session_state.edit_mode = False
        st.session_state.preview_mode = False
        st.rerun()

with col2:
    if st.button("➡️ Next", disabled=st.session_state.current_index >= len(image_paths) - 1, use_container_width=True):
        st.session_state.current_index += 1
        st.session_state.edit_mode = False
        st.session_state.preview_mode = False
        st.rerun()

with col3:
    if st.button("⏭️ Skip to End", use_container_width=True):
        st.session_state.current_index = len(image_paths) - 1
        st.session_state.edit_mode = False
        st.session_state.preview_mode = False
        st.rerun()

with col4:
    if st.button("🔄 Reset Progress", use_container_width=True):
        st.session_state.corrected_count = 0
        st.rerun()

with col5:
    if st.button("🏠 First Image", use_container_width=True):
        st.session_state.current_index = 0
        st.session_state.edit_mode = False
        st.session_state.preview_mode = False
        st.rerun()

# Progress bar at bottom
st.progress((st.session_state.current_index + 1) / len(image_paths))
