# Copyright (C) 2025 Anthony Casagrande
# AGPL-3.0 license

import argparse

from ffmpeg_decode import FfmpegVisionProcessor


def parse_resolution_string(res_str: str) -> tuple[int, int]:
    """
    Parse a resolution string like '3840x2160' or '1920,1080' into (width, height).
    You can make this more robust to handle multiple formats.
    """
    # Try splitting by 'x' first, then fall back to ',' if needed.
    if 'x' in res_str:
        w_str, h_str = res_str.lower().split('x')
    elif ',' in res_str:
        w_str, h_str = res_str.lower().split(',')
    else:
        raise ValueError(f"Invalid resolution format: '{res_str}'. Use e.g. 1920x1080")
    return int(w_str), int(h_str)


def parse_bool(value: str) -> bool:
    """
    Convert common string representations of booleans to Python bool.
    """
    value = value.strip().lower()
    if value in ['true', '1', 'yes']:
        return True
    elif value in ['false', '0', 'no']:
        return False
    else:
        raise ValueError(f"Invalid boolean value: '{value}'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI for configuring and running the FfmpegVisionProcessor."
    )

    # Required positional argument
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file."
    )

    # Optional arguments
    parser.add_argument(
        "--start_time", type=float, default=0,
        help="Start time in seconds (default: 0)."
    )
    parser.add_argument(
        "--end_time", type=float, default=0,
        help="End time in seconds (default: 0 for no limit)."
    )
    parser.add_argument(
        "--vid_stride", type=int, default=1,
        help="Process every Nth frame (default: 1)."
    )
    parser.add_argument(
        "--out_size", type=str, default="3840x2160",
        help="Output resolution as WIDTHxHEIGHT (default: 3840x2160)."
    )
    parser.add_argument(
        "--yolo_size", type=str, default="1920x1080",
        help="YOLO input resolution as WIDTHxHEIGHT (default: 1920x1080)."
    )
    parser.add_argument(
        "--preview_size", type=str, default="1920x1080",
        help="Preview window resolution as WIDTHxHEIGHT (default: 1920x1080)."
    )
    parser.add_argument(
        "--use_cuda", type=parse_bool, nargs='?', const=True, default=None,
        help="Use GPU acceleration (True/False). If provided without value, defaults to Autodetect."
    )

    parser.add_argument(
        "--iou", type=float, default=0.2,
        help="IOU threshold for object detection (default: 0.2)."
    )
    parser.add_argument(
        "--conf", type=float, default=0.01,
        help="Confidence threshold for object detection (default: 0.01)."
    )

    # Boolean flags: store_true means if the flag is present, the value is True
    parser.add_argument(
        "--show_plot", action="store_true",
        help="If set, display the detection result plots."
    )
    parser.add_argument(
        "--show_preview", action="store_true",
        help="If set, show a preview window with detections."
    )
    parser.add_argument(
        "--save_video", action="store_true",
        help="If set, save output video with detections."
    )
    parser.add_argument(
        "--save_lost_trackers", action="store_true",
        help="If set, save data of lost trackers."
    )
    parser.add_argument(
        "--show_trace", action="store_true",
        help="If set, draw the path/trace of objects."
    )
    parser.add_argument(
        "--trace_length", type=int, default=50,
        help="Maximum length of each object's trace (default: 50)."
    )
    parser.add_argument(
        "--trace_thickness", type=int, default=4,
        help="Line thickness for traces (default: 4)."
    )
    parser.add_argument(
        "--show_box", action="store_true",
        help="If set, draw bounding boxes around detections."
    )
    parser.add_argument(
        "--show_label", action="store_true",
        help="If set, draw text labels above bounding boxes."
    )
    parser.add_argument(
        "--show_triangle", action="store_true",
        help="If set, draw a triangle marker on each detected object."
    )
    parser.add_argument(
        "--triangle_size", type=str, default="50x50",
        help="Triangle size as WIDTHxHEIGHT (default: 50x50)."
    )
    parser.add_argument(
        "--triangle_thickness", type=int, default=4,
        help="Line thickness for triangles (default: 4)."
    )
    parser.add_argument(
        "--save_auto_tracker", action="store_true",
        help="If set, save auto tracker data."
    )
    parser.add_argument(
        "--auto_tracker_preview", action="store_true",
        help="If set, show a preview window of auto tracker steps."
    )
    parser.add_argument(
        "--model_file", type=str, default="weights/train29s.pt",
        help="Path to the model weights file (default: weights/train29s.pt)."
    )
    parser.add_argument(
        "--stabilization_classes", nargs='*',
        default=["Exalt", "ORCA Orange", "ORCA Blue", "Fridge"],
        help="List of classes to stabilize (default: ['Exalt','ORCA Orange','ORCA Blue','Fridge'])."
    )
    parser.add_argument(
        "--primary_stabilization_class", type=str, default="Exalt",
        help="Main class used for stabilization (default: 'Exalt')."
    )
    parser.add_argument(
        "--agnostic_nms", action="store_true",
        help="Use class-agnostic NMS (default: True)."
    )
    parser.add_argument(
        "--stabilize", action="store_true",
        help="If set, attempt to stabilize the video."
    )
    parser.add_argument(
        "--trail_stabilize", action="store_true",
        help="If set, apply additional stabilization on object trails."
    )
    parser.add_argument(
        "--gyro_adjust", action="store_true",
        help="If set, apply gyroscopic adjustments to the video (experimental)."
    )
    parser.add_argument(
        "--tracker_output_dir", type=str, default="tracked_cars",
        help="Directory to save tracker videos (default: tracked_cars)."
    )
    parser.add_argument(
        "--min_tracker_seconds", type=float, default=1.0,
        help="Minimum tracker lifespan in seconds before saving (default: 1.0)."
    )
    parser.add_argument(
        "--save_tracker_videos", action="store_true",
        help="If set, save individual tracker videos."
    )
    parser.add_argument(
        "--tracker_video_size", type=str, default="608x1080",
        help="Cropped tracker video size as WIDTHxHEIGHT (default: 608x1080)."
    )
    parser.add_argument(
        "--target_bbox_size", type=str, default="150x50",
        help="Approx. bounding box size of tracker target as WIDTHxHEIGHT (default: 150x50)."
    )
    parser.add_argument(
        "--tracker_lose_track", type=int, default=30,
        help="Number of frames allowed to lose track before giving up (default: 30)."
    )
    parser.add_argument(
        "--tracker_video_auto_scale", action="store_true",
        help="If set, automatically zoom tracker videos to keep a consistent bounding box size."
    )
    parser.add_argument(
        "--tracker_scale_min", type=float, default=0.5,
        help="Minimum scale factor for auto-scaling (default: 0.5)."
    )
    parser.add_argument(
        "--tracker_scale_max", type=float, default=1.5,
        help="Maximum scale factor for auto-scaling (default: 1.5)."
    )
    parser.add_argument(
        "--tracker_scale_rate", type=float, default=1.10,
        help="Rate at which changes in bbox size affect the zoom (default: 1.10)."
    )
    parser.add_argument(
        "--tracker_scale_min_delta", type=float, default=0.2,
        help="Minimum delta in scale before applying changes (default: 0.2)."
    )
    parser.add_argument(
        "--tracker_scale_smooth_delta", type=float, default=0.01,
        help="Maximum per-frame scale change (default: 0.01)."
    )
    parser.add_argument(
        "--auto_tracker_timeout", type=float, default=5,
        help="Time in seconds to stay on the same tracker ID before stopping (default: 5)."
    )
    parser.add_argument(
        "--auto_tracker_concurrent_missing", type=int, default=3,
        help="Number of consecutive missing frames before switching track ID (default: 3)."
    )
    parser.add_argument(
        "--save_json", action="store_true",
        help="If set, also save results to a JSON file."
    )

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Convert resolution-like strings to tuples
    args.out_size = parse_resolution_string(args.out_size)
    args.yolo_size = parse_resolution_string(args.yolo_size)
    args.preview_size = parse_resolution_string(args.preview_size)
    args.triangle_size = parse_resolution_string(args.triangle_size)
    args.tracker_video_size = parse_resolution_string(args.tracker_video_size)
    args.target_bbox_size = parse_resolution_string(args.target_bbox_size)

    # Instantiate the processor using all parsed arguments
    # We can do **vars(args) to unpack or list them individually.
    p = FfmpegVisionProcessor(**vars(args))
    p.process_video()


if __name__ == "__main__":
    main()
