# Copyright (C) 2025 Anthony Casagrande
# AGPL-3.0 license

from ultralytics import YOLO, checks, hub

checks()
hub.login()

# Batch sizes by imgsz
batch_sizes = {
    1920: {
        'yolo11x.pt': 1,
        'yolo11l.pt': 1,
        'yolo11m.pt': 2,
        'yolo11s.pt': 3, # 4,
        'yolo11n.pt': 6,
    },
    1280: {
        'yolo11s.pt': 9,   # 10 = 11.3 GB, 11 = 11.4 - 12 GB
        'yolo11m.pt': 5,   # 5 = 10.7 - 11.1 GB
        'yolo11l.pt': 4,   # 4 = 11.3 GB, 3 = 8.7 GB
    },
    1088: {
        'yolo11l.pt': 5,
        'yolo11s.pt': 15,
    },
    960: {
        'yolo11l.pt': 8,
    },
    640: {
        "yolo11x.pt": 11,
    }
}

imgsz = 1280

# # For fresh learning
model_file = 'yolo11s.pt'
model = YOLO(model_file)

batch_size = -1
if imgsz in batch_sizes and model_file in batch_sizes[imgsz]:
    batch_size = batch_sizes[imgsz][model_file]

results = model.train(
    data="dataset.v3.yaml",
    imgsz=imgsz,
    epochs=500,
    patience=100,
    batch=batch_size,
)
