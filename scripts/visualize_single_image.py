import cv2
import numpy as np
from pathlib import Path

# Path to image and label
img_path = "../data/original/train/images/0a0bff7edde8b466.jpg"
label_path = "../data/original/train/labels/0a0bff7edde8b466.txt"
output_path = "../results/visualize_0a0bff7edde8b466.jpg"

# Read image
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")
h, w = img.shape[:2]

# Read labels
def read_yolo_label(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                box = [float(x) for x in parts]
                boxes.append(box)
    return boxes

boxes = read_yolo_label(label_path)

# Draw boxes
for i, (cls, x, y, bw, bh) in enumerate(boxes):
    x1 = int((x - bw/2) * w)
    y1 = int((y - bh/2) * h)
    x2 = int((x + bw/2) * w)
    y2 = int((y + bh/2) * h)
    color = (0, 255, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, f"Box {i+1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

cv2.imwrite(output_path, img)
print(f"Saved visualization to {output_path}")
