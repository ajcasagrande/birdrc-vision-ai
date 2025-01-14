import os
from dataclasses import dataclass

import cv2
import ffmpeg
import numpy as np

from ffmpeg_encoder import FfmpegVideoEncoder


@dataclass
class TrackerWriter:
    ff: 'FfmpegVisionProcessor'
    track_id: int | str
    frame_index: int

    output_file: str = None
    preview: bool = False
    use_cuda: bool = False
    is_auto = False
    current_id = None
    prev = None
    missing = 0
    frames_written = 0
    current_missing = 0
    prev_scale_factor = None
    encoder = None
    current_tracker_frames = 0

    def __post_init__(self):
        """Initialize video encoder for a specific tracker."""
        os.makedirs(self.ff.tracker_output_dir, exist_ok=True)
        self.is_auto = self.track_id == 'auto'
        if self.output_file is None:
            if not self.is_auto:
                self.output_file = f"{self.ff.tracker_output_dir}/car_{self.track_id:04d}.mp4"
            else:
                self.output_file = f"{self.ff.tracker_output_dir}/auto_tracker.mp4"
        if not self.is_auto:
            self.ff.tracker_writers[self.track_id] = self
        self.encoder = FfmpegVideoEncoder(self.output_file, self.ff.tracker_video_size, self.ff.info.fps / self.ff.vid_stride, use_cuda=self.use_cuda)
        self.encoder.start_encoding()

    def calculate_crop_region(self, bboxes):
        crop_width, crop_height = self.ff.tracker_video_size  # Final cropped frame size
        frame_width, frame_height = self.ff.out_size  # frame size
        if not bboxes:
            center_x = frame_width // 2
            center_y = frame_height // 2
            crop_x = max(0, center_x - crop_width // 2)
            crop_y = max(0, center_y - crop_height // 2)
            return (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)

        # Step 1: Build graph of overlapping RC cars
        nodes = []
        for b in bboxes:
            x_center = (b[0] + b[2]) // 2
            y_center = (b[1] + b[3]) // 2
            width = b[2] - b[0]
            height = b[3] - b[1]
            nodes.append((x_center, y_center, width, height))

        adjacency_list = []
        for i, (x1, y1, w1, h1) in enumerate(nodes):
            overlaps = []
            for j, (x2, y2, w2, h2) in enumerate(nodes):
                if i != j and abs(x1 - x2) < crop_width and abs(y1 - y2) < crop_height:
                    overlaps.append(j)
            adjacency_list.append(overlaps)

        # Step 2: Find the subset of cars that fit in the crop region
        best_group = []
        max_visible_cars = 0

        for i in range(len(nodes)):
            visited = set()
            stack = [i]
            group = []

            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    group.append(node)
                    stack.extend(adjacency_list[node])

            # Check if this group fits in the crop
            xs = [nodes[j][0] for j in group]
            ys = [nodes[j][1] for j in group]

            if max(xs) - min(xs) <= crop_width and max(ys) - min(ys) <= crop_height:
                if len(group) > max_visible_cars:
                    max_visible_cars = len(group)
                    best_group = group

        # Step 3: Calculate crop center based on best group
        if best_group:
            xs = [nodes[j][0] for j in best_group]
            ys = [nodes[j][1] for j in best_group]
            center_x = sum(xs) // len(xs)
            center_y = sum(ys) // len(ys)
        else:
            center_x = frame_width // 2
            center_y = frame_height // 2

        crop_x = max(0, center_x - crop_width // 2)
        crop_y = max(0, center_y - crop_height // 2)
        crop_x = min(crop_x, frame_width - crop_width)
        crop_y = min(crop_y, frame_height - crop_height)

        return (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)


    def _crop_frame(self, frame, bbox):
        """
        Crop the frame to a dynamically scaled size around the bounding box to match
        the object to a target size of 150x50, then resize to the defined tracker size.
        """
        _x1, _y1, _x2, _y2 = bbox
        target_bbox_width, target_bbox_height = self.ff.target_bbox_size  # Desired size of the bounding box
        output_width, output_height = self.ff.tracker_video_size  # Final cropped frame size
        frame_height, frame_width = frame.shape[:2]

        # Calculate the actual bounding box dimensions
        bbox_width = _x2 - _x1
        bbox_height = _y2 - _y1

        # Calculate the scaling factor based on the target bounding box size
        if self.ff.tracker_video_auto_scale:
            crop_scale_factor = target_bbox_height / bbox_height * self.ff.tracker_scale_rate
            if self.prev_scale_factor is not None:
                delta = crop_scale_factor - self.prev_scale_factor
                if abs(delta) <= self.ff.tracker_scale_min_delta:
                    # print(f'Dropping small transition for {self.prev_scale_factor=}, {crop_scale_factor=}, {delta=}')
                    crop_scale_factor = self.prev_scale_factor
                elif abs(delta) >= self.ff.tracker_scale_smooth_delta:
                    # print(f'Smoothing out large transition for {self.prev_scale_factor=}, {crop_scale_factor=}, {delta=}')
                    if delta < 0:
                        crop_scale_factor = self.prev_scale_factor - self.ff.tracker_scale_smooth_delta
                    else:
                        crop_scale_factor = self.prev_scale_factor + self.ff.tracker_scale_smooth_delta
            # clamp it
            crop_scale_factor = min(crop_scale_factor, self.ff.tracker_scale_max)
            crop_scale_factor = max(crop_scale_factor, self.ff.tracker_scale_min)
        else:
            crop_scale_factor = 1.0

        self.prev_scale_factor = crop_scale_factor

        # Adjust the crop size
        crop_width = int(round(output_width / crop_scale_factor))
        crop_height = int(round(output_height / crop_scale_factor))

        # Center of the bounding box
        cx = int(round(_x1 + bbox_width / 2))
        cy = int(round(_y1 + bbox_height / 2))

        # Calculate crop boundaries
        x1 = max(cx - crop_width // 2, 0)
        y1 = max(cy - crop_height // 2, 0)
        x2 = min(x1 + crop_width, frame_width)
        y2 = min(y1 + crop_height, frame_height)

        # Adjust boundaries if the crop exceeds the frame size
        if x2 - x1 != crop_width:
            x1 = max(x2 - crop_width, 0)
        if y2 - y1 != crop_height:
            y1 = max(y2 - crop_height, 0)

        # Crop the frame
        cropped_frame = frame[y1:y2, x1:x2]

        if crop_scale_factor != 1.0:
            # Resize the cropped frame to the tracker video size
            cropped_frame = cv2.resize(cropped_frame, (output_width, output_height),
                                       # Use INTER_CUBIC for upscale, and INTER_AREA for downscale
                                       interpolation=cv2.INTER_CUBIC if crop_scale_factor > 1.0 else cv2.INTER_AREA)
        return cropped_frame

    def write_frame(self, frame, bbox):
        if bbox is not None:
            self.prev = bbox
            if self.missing > 0:
                print(f'missing tracker {self.track_id} resumed after {self.missing} frames')
                self.missing = 0
            self.current_missing = 0
        else:
            bbox = self.prev
            self.missing += 1
            self.current_missing += 1
            print(f'writing frame for missing tracker {self.track_id}')

        if bbox is None:
            print(f'--- Error - unable to crop invalid bbox of None ---')
            return
        cropped_frame = self._crop_frame(frame, bbox)
        self.encoder.write_frame(cropped_frame)
        self.frames_written += 1
        self.current_tracker_frames += 1
        if self.missing > self.ff.tracker_lose_track:
            self.release()
        if self.preview:
            cv2.imshow(f'RC Car {self.track_id}', cropped_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise  KeyboardInterrupt

    def release(self):
        print(f'Releasing tracker {self.track_id}')
        self.encoder.close()
        if self.track_id in self.ff.tracker_writers:
            del self.ff.tracker_writers[self.track_id]
        if self.frames_written - self.missing < (self.ff.min_tracker_seconds * self.ff.info.fps):
            print(f'Deleting video for tracker {self.track_id} which had only {self.frames_written - self.missing} frame(s)')
            os.remove(self.output_file)
        elif self.frames_written - self.missing > 10:
            try:
                self.add_audio()
            except:
                pass

    def add_audio(self):
        # extract the audio file
        audio_file = self.output_file.replace('.mp4', '.aac')
        out = (
            ffmpeg
            .input(self.ff.video_path, ss=f'{self.frame_index * self.ff.info.fps + self.ff.start_time}', t=f'{self.frames_written * self.ff.info.fps}')
            .output(audio_file, acodec='aac')
        )
        print(out.compile())
        out.run(overwrite_output=True)

        # join streams
        inv = ffmpeg.input(self.output_file)
        ina = ffmpeg.input(audio_file)
        out = ffmpeg.output(inv, ina, self.output_file.replace('.mp4','_audio.mp4'),
                            vcodec='copy', acodec='copy')
        out.run(overwrite_output=True)

        os.remove(self.output_file)
        os.remove(audio_file)

    def next_track_id(self, car_trackers):
        old_id = self.current_id
        ids = [id for id in car_trackers.keys() if old_id is None or id > old_id]
        if len(ids) > 0:
            self.current_id = ids[0]
        else:
            self.current_id = [id for id in car_trackers.keys() if id != old_id][0]
        print(f' --- Auto tracker switching from {old_id} to {self.current_id} after {self.current_tracker_frames} frames ---')
        self.current_tracker_frames = 0
        self.current_missing = 0
