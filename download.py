import fiftyone as fo
import fiftyone.zoo as foz


print(fo.__version__)

# Train Dataset
# dataset = foz.load_zoo_dataset(
#     "open-images-v6",
#     split="train",
#     label_types=["detections"],
#     classes=["Cat"],
#     max_samples=1500
# )

# Validate Dataset
# val_dataset = foz.load_zoo_dataset(
#     "open-images-v6",
#     split="validation",
#     label_types=["detections"],
#     classes=["Cat"],
#     max_samples=300
# )
# Test Datatset

# test_dataset = foz.load_zoo_dataset(
#     "open-images-v6",
#     split="test",
#     label_types=["detections"],
#     classes=["Cat"],
#     max_samples=300
# 