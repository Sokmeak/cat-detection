import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import gc

# Use non-interactive backend to prevent memory issues
plt.switch_backend('TkAgg')

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

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"Box {i+1}",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    return img


# Paths
img_dir = Path("../data/original/train/images")
label_dir = Path("../data/original/train/labels")

image_paths = sorted(img_dir.glob("*.jpg"))

print(f"Found {len(image_paths)} images")

BATCH_SIZE = 3  # Reduced from 5 to use less memory

for i in range(0, len(image_paths), BATCH_SIZE):
    batch = image_paths[i:i + BATCH_SIZE]
    
    print(f"\nShowing batch {i//BATCH_SIZE + 1}/{(len(image_paths)-1)//BATCH_SIZE + 1}")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # Reduced figure size
    axes = axes.flatten()

    for ax, img_path in zip(axes, batch):
        label_path = label_dir / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read {img_path.name}")
            continue
            
        boxes = read_yolo_label(label_path)

        img = draw_boxes(img, boxes)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax.imshow(img_rgb)
        ax.set_title(img_path.name, fontsize=8)
        ax.axis("off")
        
        # Immediately free image memory
        del img, img_rgb

    # Hide unused subplots
    for ax in axes[len(batch):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show(block=False)  # Non-blocking show
    plt.pause(0.1)  # Brief pause for rendering

    user_input = input("Press ENTER to load next batch (or 'q' to quit)...")
    
    # Aggressive memory cleanup
    plt.close(fig)
    plt.close('all')  # Close any lingering figures
    del fig, axes, batch
    gc.collect()  # Force garbage collection
    
    if user_input.lower() == 'q':
        break
