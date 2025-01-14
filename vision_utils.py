from typing import Optional

import cv2
from supervision.annotators.base import ImageType
from supervision.annotators.utils import resolve_color
from supervision.utils.conversion import ensure_cv2_image_for_annotation
from ultralytics.utils.plotting import Annotator, colors

import supervision as sv
import numpy as np


class CustomTraceAnnotator(sv.TraceAnnotator):
    @ensure_cv2_image_for_annotation
    def annotate(
            self,
            scene: ImageType,
            detections: sv.Detections,
            custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        assert isinstance(scene, np.ndarray)
        if detections.tracker_id is None:
            raise ValueError(
                "The `tracker_id` field is missing in the provided detections."
                " See more: https://supervision.roboflow.com/latest/how_to/track_objects"
            )

        self.trace.put(detections)
        for detection_idx in range(len(detections)):
            tracker_id = int(detections.tracker_id[detection_idx])
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            xmin, ymin, xmax, ymax = detections.xyxy[detection_idx].astype(int)
            xy = self.trace.get(tracker_id=tracker_id)
            outside_xy = xy[
                (xy[:, 0] <= xmin) | (xy[:, 0] >= xmax) |  # x not in [xmin, xmax]
                (xy[:, 1] <= ymin) | (xy[:, 1] >= ymax)    # y not in [ymin, ymax]
            ]
            if len(outside_xy) > 1:
                scene = cv2.polylines(
                    scene,
                    [outside_xy.astype(np.int32)],
                    False,
                    color=color.as_bgr(),
                    thickness=self.thickness,
                    lineType=cv2.LINE_AA
                )
        return scene


def plot_scaled_results(results, img=None, line_width=1, font_size=None, font="Arial.ttf", labels=True,
                        color_mode='class', scale_factor=(1, 1), conf=True):
    names = results.names
    is_obb = results.obb is not None
    pred_boxes = results.obb if is_obb else results.boxes
    annotator = Annotator(
        results.orig_img if img is None else img,
        line_width,
        font_size,
        font,
        False,
        example=results.names,
    )

    # Plot Detect results
    if pred_boxes is not None:
        for i, d in enumerate(reversed(pred_boxes)):
            c, d_conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
            name = ("" if id is None else f"id:{id} ") + names[c]
            label = (f"{name} {d_conf:.2f}" if conf else name) if labels else None
            box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze().tolist()
            box[0] *= scale_factor[0]
            box[1] *= scale_factor[1]
            box[2] *= scale_factor[0]
            box[3] *= scale_factor[1]
            annotator.box_label(
                box,
                label,
                color=colors(
                    c
                    if color_mode == "class"
                    else id
                    if id is not None
                    else i
                    if color_mode == "instance"
                    else None,
                    True,
                ),
                rotated=is_obb,
            )
    return annotator.result()


def scale_detections(detections, scale_factor):
    for box in detections.xyxy:
        box[0] *= scale_factor[0]
        box[1] *= scale_factor[1]
        box[2] *= scale_factor[0]
        box[3] *= scale_factor[1]

def shift_detections(detections, shift):
    for box in detections.xyxy:
        box[0] += shift[0]
        box[1] += shift[1]
        box[2] += shift[0]
        box[3] += shift[1]
