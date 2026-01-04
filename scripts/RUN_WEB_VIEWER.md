# Web Label Viewer - Quick Start

## Installation

Make sure streamlit is installed:

```bash
pip install streamlit
```

## Run the Web Interface

```bash
cd scripts
streamlit run web_label_viewer.py
```

## Features

✅ **Navigate easily** - Previous/Next buttons or jump to any image  
✅ **Flag incorrect labels** - Mark images with wrong bounding boxes  
✅ **Auto-save** - Flagged images saved to `flagged_images.txt`  
✅ **Export list** - Download or copy incorrect image names  
✅ **No crashes** - Memory efficient, web-based  
✅ **Visual feedback** - See boxes and image info clearly

## How to Use

1. **View images** - Images with bounding boxes shown in browser
2. **Mark incorrect** - Click "Flag as Incorrect" for bad labels
3. **Quick flag + next** - Use "Next + Flag" button for fast reviewing
4. **Export** - Download the list from sidebar or copy names
5. **Navigate** - Use Previous/Next or jump to specific image number

## Output

Incorrect images are saved to: `cat-detection/flagged_images.txt`

Each line contains one image filename that needs label correction.
