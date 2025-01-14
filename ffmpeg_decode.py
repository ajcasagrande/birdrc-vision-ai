# Copyright (C) 2025 Anthony Casagrande
# AGPL-3.0 license

import os
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import ffmpeg
import numpy as np
import supervision as sv
import torch.cuda
import ultralytics.trackers
from supervision import ColorLookup, JSONSink
from ultralytics import YOLO

import yolo_trackers
from classes import VideoInfo
from constants import *
from timer import Timer
from tracker_writer import TrackerWriter
from trail_stabilizer3 import TrailStabilizer3
from video_stabilizer3 import VideoStabilizer3
from vision_utils import CustomTraceAnnotator, scale_detections, shift_detections

# monkey-patch the botsort implementation
ultralytics.trackers.BOTSORT = yolo_trackers.BOTSORT
ultralytics.trackers.track.TRACKER_MAP['botsort'] = yolo_trackers.BOTSORT

os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "65535"


@dataclass
class FfmpegVisionProcessor:
    # Required
    video_path: str

    # Optional
    start_time: float = 0
    end_time: float = 0
    vid_stride: int = 1
    out_size: tuple[int, int] = _4k
    yolo_size: tuple[int, int] = _1080p
    preview_size: tuple[int, int] = _1080p
    use_cuda: bool = None

    iou: float =  0.2 # 0.7
    conf: float = 0.01
    show_plot: bool = False
    show_preview: bool = False
    save_video: bool = False
    save_lost_trackers: bool = False
    show_trace: bool = False
    trace_length: int = 50
    trace_thickness: int = 4
    show_box: bool = False
    show_label: bool = False
    show_triangle: bool = False
    triangle_size: tuple[int, int] = (50, 50)
    triangle_thickness: int = 4
    save_auto_tracker: bool = False
    auto_tracker_preview: bool = False
    model_file: str = "weights/train29s.pt"
    stabilization_classes: list[str] = field(default_factory=lambda: ['Exalt', 'ORCA Orange', 'ORCA Blue', 'Fridge'])
    primary_stabilization_class: str = "Exalt"
    agnostic_nms: bool = True
    stabilize: bool = False
    trail_stabilize: bool = False
    gyro_adjust: bool = False
    tracker_output_dir: str = "tracked_cars"  # Directory to save tracker videos
    min_tracker_seconds: float = 1.0  # deletes any tracker videos seen for less than this amount of seconds
    save_tracker_videos: bool = False
    tracker_video_size: tuple[int, int] = (608, 1080)  # New setting for cropped video size
    target_bbox_size: tuple[int, int] = (150, 50)
    tracker_lose_track: int = 30
    tracker_video_auto_scale: bool = False  # whether to zoom the tracker videos based on bbox sizing
    tracker_scale_min: float = 0.5  # the minimum zoom amount
    tracker_scale_max: float = 1.5  # the maximum zoom amount
    tracker_scale_rate: float = 1.10   # how much the change in bbox size affects the zoom
    tracker_scale_min_delta: float = 0.2  # the minimum amount of zoom change before we do anything (prevent jitter)
    tracker_scale_smooth_delta: float = 0.01   # the maximum amount of zoom we are willing to change per frame
    auto_tracker_timeout: float = 5  # the number of seconds to stay on the same track_id before stopping
    auto_tracker_concurrent_missing: int = 3  # the number of missing frames before switching to a new tracker id
    save_json: bool = True

    # Internal
    output_json: str = None
    output_file: str = None
    info: VideoInfo = None
    track_counts: dict[int] = field(default_factory=lambda: defaultdict(int))
    tracker_writers: dict[int, any] = None
    auto_tracker: TrackerWriter = None
    gyro: 'GoProGyro' = None
    _timer = Timer()
    scale_factor: tuple[float, float] = None
    _decode_process = None
    _encode_process = None
    _model: YOLO = None
    _model_obb: YOLO = None
    stabilizer = None

    def __post_init__(self):
        if self.output_json is None:
            self.output_json = self.video_path + ".json"
        if self.output_file is None:
            self.output_file = self.video_path + "_out.mp4"
        if self.use_cuda is None or self.use_cuda:
            self.use_cuda = torch.cuda.is_available()
        self._model = YOLO(self.model_file)

        if self.gyro_adjust:
            from gopro_utils import GoProGyro
            self.gyro = GoProGyro(self.video_path, self.yolo_size)
        if self.stabilize:
            self.trail_stabilize = False  # only support 1 or the other
            self.stabilizer = VideoStabilizer3(self.stabilization_classes, self.yolo_size, self.out_size,
                                               primary_class=self.primary_stabilization_class)
        elif self.trail_stabilize:
            self.trail_stabilizer = TrailStabilizer3(self.primary_stabilization_class, self.yolo_size, self.out_size)

        if self.save_json:
            self.json_sink = JSONSink(self.output_json)

        self.tracker_writers = {}

    def _probe_video_metadata(self):
        """Function to get video metadata using ffprobe"""
        probe = ffmpeg.probe(self.video_path)
        video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        duration = float(video_info['duration'] if 'duration' in video_info else -1)
        fps = eval(video_info['avg_frame_rate'])  # Convert avg_frame_rate to float
        codec = video_info['codec_name']
        self.info = VideoInfo(width, height, duration, fps, codec)

    def _init_decode_process(self):
        """ffmpeg command to decode video frames with NVIDIA hevc decoder from start_time to end_time"""
        input_kwargs = {
            'vcodec': f'{self.info.codec}_cuvid' if self.use_cuda else self.info.codec
        }
        if self.start_time > 0:
            input_kwargs['ss'] = self.start_time
        if self.end_time > 0:
            input_kwargs['to'] = self.end_time
        if self.use_cuda:
            input_kwargs['hwaccel'] = 'cuda'

        self._decode_process = (
            ffmpeg
            .input(self.video_path, **input_kwargs)
            .output('pipe:',
                    format='rawvideo',
                    pix_fmt='bgr24',
                    vf=f'scale={self.out_size[0]}:{self.out_size[1]}')
            .run_async(pipe_stdout=True)
        )

    def _init_encode_process(self):
        if not self.save_video:
            return
        self._encode_process = (
            ffmpeg
            .input('pipe:',
                   format='rawvideo',
                   pix_fmt='bgr24',
                   s=f'{self.out_size[0]}x{self.out_size[1]}',
                   framerate=self.info.fps / self.vid_stride)
            .output(self.output_file,
                    vcodec='hevc_nvenc',
                    pix_fmt='yuv420p',
                    preset='p3',
                    qp=21)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def process_video(self):
        """Function to process video with ffmpeg-python and YOLOv11 using NVIDIA GPU for HEVC decoding"""

        if self.gyro_adjust:
            self.gyro.process()

        if self.save_json:
            self.json_sink.open()

        prev_results = {}
        cur_results = {}
        results_data = []
        try:
            # Get video metadata
            self._probe_video_metadata()
            if self.out_size is None or self.out_size > (self.info.width, self.info.height):
                self.out_size = (self.info.width, self.info.height)
            self.scale_factor = (self.out_size[0] / self.yolo_size[0], self.out_size[1] / self.yolo_size[1])

            self._init_decode_process()
            self._init_encode_process()

            if self.save_auto_tracker:
                self.auto_tracker = TrackerWriter(self, 'auto', 0, use_cuda=True, preview=self.auto_tracker_preview)

            # frame size = width * height * 3 channels
            frame_size = self.out_size[0] * self.out_size[1] * 3
            frame_index = -1  # start at -1 so first increment goes to 0

            color_palette = (
                sv.ColorPalette.DEFAULT
                # sv.ColorPalette.from_hex(['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#00ffff', '#ff00ff'])
            )

            obb_annotator = sv.OrientedBoxAnnotator(
                color=color_palette,
                thickness=2,
            )
            box_annotator = sv.BoxAnnotator(
                color=color_palette,
                thickness=2,
                color_lookup=ColorLookup.TRACK,
            )
            triangle_annotator = sv.TriangleAnnotator(
                color=color_palette,
                base=self.triangle_size[0],
                height=self.triangle_size[1],
                color_lookup=ColorLookup.TRACK,
                outline_thickness=self.triangle_thickness,
            )
            label_annotator = sv.LabelAnnotator(
                color=color_palette,
                text_padding=1,
                color_lookup=ColorLookup.TRACK,
                text_scale=3,
                text_thickness=8,
            )
            ellipse_annotator = sv.EllipseAnnotator(
                color=color_palette,
                color_lookup=ColorLookup.TRACK,
            )
            if self.show_trace:
                # trace_annotator = sv.TraceAnnotator(
                trace_annotator = CustomTraceAnnotator(
                    color=color_palette,
                    thickness=self.trace_thickness,
                    color_lookup=ColorLookup.TRACK,
                    trace_length=self.trace_length,
                )
            background_overlay_annotator = sv.BackgroundOverlayAnnotator()
            smoother = sv.DetectionsSmoother()
            prev_detections = None

            while True:
                # Read a frame
                in_bytes = self._decode_process.stdout.read(frame_size)
                if not in_bytes:
                    break

                frame_index += 1

                # Skip frames based on frame_interval
                if frame_index % self.vid_stride != 0:
                    continue

                # Convert raw bytes to a numpy array
                frame = np.frombuffer(in_bytes, np.uint8).reshape([self.out_size[1], self.out_size[0], 3])
                if self.yolo_size < self.out_size:
                    scaled_frame = cv2.resize(
                        frame, self.yolo_size, interpolation=cv2.INTER_CUBIC
                    )
                else:
                    scaled_frame = frame

                # res = self._model_obb(frame, conf=0.9, iou=0.1)[0]
                # det = sv.Detections.from_ultralytics(res)
                # frame = obb_annotator.annotate(frame.copy(), det)
                # print(det)

                # Run YOLO inference on the scaled frame
                results = self._model.track(scaled_frame,
                                            persist=True,
                                            conf=self.conf,
                                            iou=self.iou,
                                            agnostic_nms=self.agnostic_nms,
                                            tracker='./trackers/botsort_cfg.yaml')

                detections = sv.Detections.from_ultralytics(results[0])
                if not self.stabilize:
                    scale_detections(detections, self.scale_factor)

                # detections = smoother.update_with_detections(detections)

                if self.save_json:
                    self.json_sink.append(detections, {'frame_index': frame_index})

                if self.save_auto_tracker:
                    # car_trackers = {id: i for i, (cls, id, xyxy) in enumerate(zip(detections.data['class_name'], detections.tracker_id, detections.xyxy)) if cls in RC_CAR_CLASSES}

                    bboxes = [bbox.astype(int) for bbox in detections.xyxy]
                    crop_bbox = self.auto_tracker.calculate_crop_region(bboxes)
                    self.auto_tracker.write_frame(frame, crop_bbox)

                    # if detections.tracker_id is None or len(car_trackers) == 0:
                    #     self.auto_tracker.write_frame(frame, None)
                    # else:
                    #     if self.auto_tracker.current_id is None \
                    #                 or (self.auto_tracker.current_id not in car_trackers
                    #             and self.auto_tracker.current_missing >= self.auto_tracker_concurrent_missing) \
                    #             or self.auto_tracker.current_tracker_frames / self.info.fps > self.auto_tracker_timeout:
                    #         self.auto_tracker.next_track_id(car_trackers)
                    #
                    #     bbox = None
                    #     if self.auto_tracker.current_id in car_trackers:
                    #         bbox = detections.xyxy[car_trackers[self.auto_tracker.current_id]].astype(int)
                    #     self.auto_tracker.write_frame(frame, bbox)

                if self.save_tracker_videos:
                    found_trackers = {id: False for id in self.tracker_writers.keys()}
                    # Process each tracker

                    if detections.tracker_id is not None:
                        # frame = box_annotator.annotate(
                        #     scene=frame.copy(),
                        #     detections=detections
                        # )
                        for bbox, tracker_id, cls in zip(detections.xyxy, detections.tracker_id, detections.data['class_name']):
                            if cls not in RC_CAR_CLASSES:
                                continue
                            bbox = bbox.astype(int)
                            if tracker_id not in self.tracker_writers:
                                writer = TrackerWriter(self, tracker_id, frame_index)
                            else:
                                writer = self.tracker_writers[tracker_id]

                            writer.write_frame(frame, bbox)
                            found_trackers[tracker_id] = True

                    # write all missing trackers
                    for id in [k for k, v in found_trackers.items() if not v]:
                        self.tracker_writers[id].write_frame(frame, None)

                if self.save_lost_trackers and detections.tracker_id is not None:
                    if prev_detections is not None and prev_detections.tracker_id is not None:
                        for track_id in prev_detections.tracker_id:
                            if track_id not in detections.tracker_id:
                                filename = f"lost_tracker_frames/frame_{frame_index:06d}_lost_{track_id:03d}.jpg"

                                # Write the frame to a jpg file
                                cv2.imwrite(filename, frame)
                                print(f"Saved lost tracker frame for ID {track_id} to {filename}")
                                break


                if self.stabilize:
                    scaled_frame, shift = self.stabilizer.stabilize_video_frame(scaled_frame, detections)
                    frame = scaled_frame
                    shift_detections(detections, shift)

                if self.show_plot and (self.show_preview or self.save_video) and detections.tracker_id is not None:
                    # plot results on the full size frame
                    # annotated_image = vision_utils.plot_scaled_results(
                    #     results[0],
                    #     img=frame,
                    #     line_width=1,
                    #     color_mode='instance',
                    #     scale_factor=self._scale_factor)

                    orig_frame = frame
                    frame = frame.copy()
                    # background_overlay_annotator.annotate(
                    #     scene=frame, detections=detections
                    # )
                    if self.show_box:
                        frame = box_annotator.annotate(
                            scene=frame,
                            detections=detections
                        )
                    if self.show_label:
                        frame = label_annotator.annotate(
                            scene=frame,
                            detections=detections,
                            labels=[
                                f"{tracker_id}" # f"{class_name} {tracker_id}"
                                for class_name, confidence, tracker_id
                                in zip(detections['class_name'], detections.confidence, detections.tracker_id)
                            ],
                        )
                    if self.show_triangle:
                        frame = triangle_annotator.annotate(
                            scene=frame, detections=detections
                        )
                    if self.show_trace:
                        if self.trail_stabilize:
                            diff_x, diff_y = self.trail_stabilizer.trail_diff(frame, detections)
                            if len(trace_annotator.trace.xy) > 0:
                                trace_annotator.trace.xy[:, 0] -= diff_x
                                trace_annotator.trace.xy[:, 1] -= diff_y

                        if self.gyro_adjust:
                            diff_x, diff_y = self.gyro.get_shift_at_frame(frame_index)
                            print(diff_x, diff_y)
                            if len(trace_annotator.trace.xy) > 0:
                                # print(trace_annotator.trace.xy)
                                trace_annotator.trace.xy[:, 0] += diff_x
                                # trace_annotator.trace.xy[:, 1] -= diff_y
                        frame = trace_annotator.annotate(scene=frame, detections=detections)

                    alpha = 0.75  # transparency factor for
                    frame = cv2.addWeighted(frame, alpha, orig_frame, 1-alpha, 0)

                if self.save_video:
                    # Write frame to FFmpeg's stdin
                    self._encode_process.stdin.write(frame.tobytes())

                if self.show_preview:
                    to_show = cv2.resize(frame, self.preview_size, interpolation=cv2.INTER_CUBIC)
                    cv2.imshow("YOLO11 Tracking", to_show)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise KeyboardInterrupt

                prev_detections = detections
                if self._timer.start_time != 0:
                    self._timer.toc(average=False)
                    timer_text = f'fps: {1 / self._timer.duration:.1f}, latency: {int(self._timer.duration * 1000)} ms Avg: (fps: {1 / self._timer.average_time:.1f}, latency: {int(self._timer.average_time * 1000)} ms)'
                    print(timer_text)
                self._timer.tic()
        except KeyboardInterrupt:
            print('Exiting...')
        finally:
            self._finalize_video_writers()
            if self.save_json:
                self.json_sink.write_and_close()

        if self.show_preview:
            cv2.destroyAllWindows()

        self._decode_process.stdout.close()
        self._decode_process.wait()

        if self.save_video:
            self._encode_process.stdin.close()
            self._encode_process.wait()

    def _finalize_video_writers(self):
        """Release all video writers."""
        values = list(self.tracker_writers.values())
        for writer in values:
            writer.release()
            # writer.stdin.close()
            # writer.wait()
        print("All tracker video writers closed.")

