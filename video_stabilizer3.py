import cv2
import numpy as np
import supervision as sv


class VideoStabilizer3:
    def __init__(self, target_classes: list[str], in_size: tuple[int, int], out_size: tuple[int, int],
                 primary_class: str = "Exalt", stabilization_offset: tuple[int, int] = (0, 0),
                 sticky_bg: bool = True, margin: int = 1, max_shift: float = 15.0):
        self.target_classes = target_classes  # List of target classes to stabilize
        self.in_size = in_size
        self.out_size = out_size
        self.frame_width = out_size[0]
        self.frame_height = out_size[1]
        self.margin = margin  # Margin around the edges to exclude objects
        self.sticky_background = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        self.previous_translation = np.array([0.0, 0.0])  # Smooth previous translation
        self.primary_class = primary_class
        self.stabilization_offset = stabilization_offset
        self.max_shift = max_shift  # Maximum shift allowed per frame
        self.sticky_bg = sticky_bg
        self.global_stabilization_offset = np.array([self.frame_width // 2 - self.stabilization_offset[0],
                                                     self.frame_height // 2 - self.stabilization_offset[1]])
        self.primary_center = None
        self.previous_centers = {}

    def stabilize_video_frame(self, frame, detections: sv.Detections):
        stabilized_frame, shift = self._process_frame(frame, detections)
        if not self.sticky_bg:
            return stabilized_frame, shift
        else:
            return self.combine_with_sticky_background(stabilized_frame), shift

    def _process_frame(self, frame, detections: sv.Detections):
        """Process a single frame to stabilize using the primary or fallback class."""
        scale_x = self.out_size[0] / self.in_size[0]
        scale_y = self.out_size[1] / self.in_size[1]

        visible_classes = {}  # Map class names to center points
        frame_center = np.array([self.frame_width / 2, self.frame_height / 2])

        for i, xyxy in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, xyxy)
            cls = detections.data['class_name'][i]

            if cls not in self.target_classes:
                continue

            # Skip objects near the frame edges
            if (x1 * scale_x < self.margin or x2 * scale_x > self.frame_width - self.margin or
                    y1 * scale_y < self.margin or y2 * scale_y > self.frame_height - self.margin):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                continue

            # Calculate the center of the detected object
            object_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            visible_classes[cls] = object_center

            # Update previous centers
            self.previous_centers[cls] = object_center

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Check for the primary class
        if self.primary_class in visible_classes:
            # Use the primary class
            self.primary_center = self.global_stabilization_offset
            primary_center = visible_classes[self.primary_class]
        else:
            # Fallback logic: Choose the class closest to the frame center
            if visible_classes:
                fallback_class, fallback_center = min(
                    visible_classes.items(),
                    key=lambda item: np.linalg.norm(item[1] - frame_center)
                )
                if self.primary_center is not None:
                    primary_center = fallback_center
                    self.primary_center = self.global_stabilization_offset + \
                        (fallback_center - self.previous_centers.get(fallback_class, fallback_center))
                else:
                    # If no prior stabilization center exists, initialize it
                    primary_center = fallback_center
                    self.primary_center = self.global_stabilization_offset
            else:
                # No visible classes
                primary_center = None

        # Compute translation
        if primary_center is not None and self.primary_center is not None:
            translation = self.primary_center - primary_center
        else:
            translation = np.array([0.0, 0.0])

        # Smooth translation
        self.previous_translation = 0.8 * self.previous_translation + 0.2 * translation

        # Clamp the shift to avoid large jumps
        shift = self.previous_translation  #np.clip(self.previous_translation, -self.max_shift, self.max_shift)

        # Apply translation
        stabilized_frame = self._translate_frame(frame, shift)

        # Debug: Log stabilization details
        print(f"Primary Class: {self.primary_class}, Translation: {translation}, Final Shift: {shift}")

        return stabilized_frame, shift

    def _translate_frame(self, frame, translation):
        """Apply translation to the frame."""
        transformation_matrix = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
        return cv2.warpAffine(frame, transformation_matrix, (self.frame_width, self.frame_height))

    def combine_with_sticky_background(self, frame):
        """Combine the current frame with the sticky background."""
        combined_frame = self.sticky_background.copy()
        non_zero_mask = np.any(frame != 0, axis=2)
        combined_frame[non_zero_mask] = frame[non_zero_mask]
        return combined_frame
