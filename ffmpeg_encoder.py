# Copyright (C) 2025 Anthony Casagrande
# AGPL-3.0 license

from dataclasses import dataclass

import ffmpeg
import torch.cuda


@dataclass
class FfmpegVideoEncoder:
    output_file: str
    out_size: tuple[int, int]
    fps: float | str

    preset: str = 'p3'
    qp: int = 21
    use_cuda: bool = None
    overwrite: bool = True

    _encode_process = None
    frames_written = 0

    def __post_init__(self):
        if self.use_cuda is None or self.use_cuda:
            self.use_cuda = torch.cuda.is_available()
        # configure GoPro's odd framerates as strings to avoid issues with floats
        if 119 < self.fps < 120:
            self.fps = "119.88"
        elif 89 < self.fps < 90:
            self.fps = "89.91"
        elif 59 < self.fps < 60:
            self.fps = "59.94"
        elif 29 < self.fps < 30:
            self.fps = "29.97"

    def start_encoding(self):
        # self._init_encode_process_nvenc()
        # time.sleep(0.25)
        # if self._encode_process.poll() is not None:
        self._init_encode_process()


    def write_frame(self, frame):
        try:
            self._encode_process.stdin.write(frame.tobytes())
            self.frames_written += 1
        except Exception as e:
            print(f"ffmpeg write error: {e}")

    def close(self):
        self._encode_process.stdin.close()
        self._encode_process.wait()

    def _init_encode_process_nvenc(self):
        out = (
            ffmpeg
            .input('pipe:',
                   format='rawvideo',
                   pix_fmt='bgr24',
                   s=f'{self.out_size[0]}x{self.out_size[1]}',
                   framerate=self.fps)
            .output(self.output_file,
                    vcodec='hevc_nvenc',
                    pix_fmt='yuv420p',
                    preset=self.preset,
                    qp=self.qp)
        )
        if self.overwrite:
            out = out.overwrite_output()
        print(out.compile())
        self._encode_process = out.run_async(pipe_stdin=True)

    def _init_encode_process(self):
        out = (
            ffmpeg
            .input('pipe:',
                   format='rawvideo',
                   pix_fmt='bgr24',
                   s=f'{self.out_size[0]}x{self.out_size[1]}',
                   framerate=self.fps)
            .output(self.output_file,
                    preset='ultrafast',
                    vcodec='libx265',
                    pix_fmt='yuv420p',
                    crf=16)
        )
        if self.overwrite:
            out = out.overwrite_output()
        print(out.compile())
        self._encode_process = out.run_async(pipe_stdin=True)

