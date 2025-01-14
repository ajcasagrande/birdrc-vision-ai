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


class TestFfmpeg(unittest.TestCase):

    def testTrailStabilization(self):
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

    def testTrackerVideos(self):
        p = FfmpegVisionProcessor(
            _video_path,
            model_file=_model_file,
            show_preview=True,
            preview_size=_720p,
            yolo_size=_720p,
            out_size=_720p,
            # save_video=True,
            show_plot=True,
            show_box=True,
            show_label=True,
            # show_triangle=True,
            show_trace=True,
            trace_length=500,
            # save_lost_trackers=True,
            start_time=_start_time,  # 80
            vid_stride=1,
            # stabilize=True,
            # trail_stabilize=True,
            # save_tracker_videos=True,
            save_auto_tracker=True,
            auto_tracker_preview=True,
            tracker_output_dir="tracked_cars",
            min_tracker_seconds=0,
            tracker_scale_max=1.25,
            tracker_scale_min=0.25,
            tracker_video_size=(1216, 2160),
        )
        p.process_video()

    def testAutoTrackerVideo(self):
        p = FfmpegVisionProcessor(
            _video_path,
            model_file=_model_file,
            # show_preview=True,
            preview_size=_720p,
            # save_video=True,
            show_plot=True,
            show_box=True,
            show_label=True,
            # show_triangle=True,
            show_trace=True,
            # save_lost_trackers=True,
            start_time=_start_time,  # 80
            yolo_size=_720p,
            out_size=_720p,
            vid_stride=1,

            # stabilize=True,
            # trail_stabilize=True,
            # save_tracker_videos=True,
            save_auto_tracker=True,
            auto_tracker_preview=True,
            tracker_video_auto_scale=True,
            tracker_video_size=(608, 1080),
        )
        p.process_video()


    def testBasicTracking(self):
        p = FfmpegVisionProcessor(
            _video_path,
            model_file=_model_file,
            show_preview=True,
            preview_size=_720p,
            # save_video=True,
            show_plot=True,
            show_box=True,
            show_label=True,
            # show_trace=True,
            start_time=_start_time,  # 80
            vid_stride=1,
        )
        p.process_video()

    def testBasicTracking720p(self):
        p = FfmpegVisionProcessor(
            _video_path,
            model_file=_model_file,
            # model_file="weights/train33s-1280.pt",
            show_preview=True,
            preview_size=_720p,
            yolo_size=_720p,
            out_size=_720p,
            # save_video=True,
            show_plot=True,
            show_box=True,
            # show_label=True,
            show_trace=True,
            start_time=_start_time,  # 80
            vid_stride=1,
            save_lost_trackers=True,
            use_cuda=False,
        )
        p.process_video()

    def testStabilization(self):
        p = FfmpegVisionProcessor(
            _video_path,
            model_file=_model_file,
            show_preview=True,
            preview_size=_720p,
            # save_video=True,
            show_plot=True,
            show_box=True,
            show_label=True,
            # show_triangle=True,
            show_trace=True,
            # save_lost_trackers=True,
            start_time=_start_time,  # 80
            yolo_size=_720p,
            out_size=_720p,
            vid_stride=1,

            stabilize=True,
        )
        p.process_video()
