import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to image and label
img_path = "../data/original/test/images/0a4bfa962852da33.jpg"
label_path = "../data/original/test/labels/0a4bfa962852da33.txt"

# Read image
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")
h, w = img.shape[:2]

# Read YOLO labels
def read_yolo_label(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append([float(x) for x in parts])
    return boxes

boxes = read_yolo_label(label_path)

# Draw boxes
for i, (cls, x, y, bw, bh) in enumerate(boxes):
    x1 = int((x - bw / 2) * w)
    y1 = int((y - bh / 2) * h)
    x2 = int((x + bw / 2) * w)
    y2 = int((y + bh / 2) * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"Box {i + 1}",
        (x1, max(y1 - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

# Convert BGR → RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Show image
plt.figure(figsize=(8, 8))
plt.imshow(img_rgb)
plt.axis("off")
plt.title("YOLO Bounding Box Visualization")
plt.show()
