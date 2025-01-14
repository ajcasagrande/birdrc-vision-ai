import cv2
import numpy as np
from ultralytics import YOLO
from constants import *

STABILIZATION_OFFSET_REF_SIZE = _4k

# todo: how can we compute this dynamically (maybe using a specific start of race frame)?
STABILIZATION_OFFSETS = {
    "Exalt": (100, -(158*2)),
    "ORCA Blue": (502, -(158*2) + 34),
    "ORCA Orange": (-937, (-158*2) - 2),
}


class TrailStabilizer3:
    def __init__(self, primary_stabilization_class: str, in_size: tuple[int, int], out_size: tuple[int, int], margin: float = 10):
        self.primary_stabilization_class = primary_stabilization_class
        self.margin = margin  # Margin around the edges to exclude objects
        self.in_size = in_size
        self.out_size = out_size
        self.previous_translation = None
        self.previous_translations = None  # Track previous translation
        # Adjust stabilization centers based on offsets adjusted for difference in x, y scaling
        self.stabilization_centers = {
            cls: np.array([
                int(round(out_size[0] / 2 + (offset[0] * out_size[0]/STABILIZATION_OFFSET_REF_SIZE[0]))),
                int(round(out_size[1] / 2 + (offset[1] * out_size[1]/STABILIZATION_OFFSET_REF_SIZE[1])))
            ])
            for cls, offset in STABILIZATION_OFFSETS.items()
        }

    def trail_diff(self, frame, detections):
        new_translations = self._process_frame(frame, detections)

        if len(new_translations) == 0:
            return np.array([0, 0], dtype=np.float32)

        # Calculate average translation
        total_translation = np.array([0.0, 0.0])
        count = 0
        for cls, translation in new_translations.items():
            total_translation += translation
            count += 1
        new_translation = total_translation / count

        if self.previous_translation is None:
            # nothing to compare it to yet
            diff = np.array([0, 0], dtype=np.float32)
        else:
            diff = new_translation - self.previous_translation

        self.previous_translation = new_translation
        self.previous_translations = new_translations
        return diff


    def _process_frame(self, frame, detections):
        scale_x = 1  # self.out_size[0] / self.in_size[0]
        scale_y = 1  # self.out_size[1] / self.in_size[1]
        new_translations = {}
        """Process a single frame to stabilize the target object."""
        for xyxy, cls in zip(detections.xyxy, detections.data['class_name']):
            if cls not in self.stabilization_centers.keys():
                continue

            x1, y1, x2, y2 = xyxy
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)

            # Skip objects near the frame edges
            if (x1 * scale_x < self.margin or x2 * scale_x > self.out_size[0] - self.margin or
                    y1 * scale_y < self.margin or y2 * scale_y > self.out_size[1] - self.margin):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                continue

            if cls != self.primary_stabilization_class and \
                    self.primary_stabilization_class in detections.data['class_name']:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 5)
                continue

            # Calculate object's center
            object_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            # Calculate translation to stabilize the object
            new_translations[cls] = self.stabilization_centers[cls] - object_center

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        return new_translations
