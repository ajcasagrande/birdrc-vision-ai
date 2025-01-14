import unittest

from ffmpeg_decode import FfmpegVisionProcessor
from constants import *


_model_file = (
    # PATH TO MODEL FILE
    "weights/train29s.pt"
)

_video_path = (
    # PATH TO VIDEO FILE
    "videos/rc_demo.mp4"
)

_start_time =  0  # number of seconds to seek into the video

def main():
    p = FfmpegVisionProcessor(
        _video_path,
        model_file=_model_file,
        show_preview=True,
        preview_size=_720p,
        save_video=True,
        show_plot=True,
        # show_box=True,
        # show_label=True,
        show_triangle=True,
        triangle_size=(10, 10),
        triangle_thickness=2,
        show_trace=True,
        start_time=_start_time,  # 80
        yolo_size=_720p,
        out_size=_720p,
        vid_stride=1,
        # stabilize=True,
        trail_stabilize=True,
        trace_length=500,
        trace_thickness=5,
    )
    p.process_video()

if __name__ == '__main__':
    main()