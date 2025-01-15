# RC Race Vision
RC Car detection using Computer Vision and YOLOv11

### Notes
This was designed around being used with an Nvidia GPU, however it potentially will run (albeit much slower) on the CPU if you use `--use-cuda False`.

Also, the tracking logic currently is very taxing, and reduces the processing FPS drastically. Use `--no-track` to get an idea of the raw performance of the model.

## Installation

```shell
# Install requirements
pip install -r requirements.txt

# Remove headless opencv and re-install the regular opencv (weird quirk)
pip uninstall -y opencv-python-headless opencv-python
pip install opencv-python~=4.10.0.84
```

Make sure you have `git-lfs` installed, and then run the following to download the sample model and data:

```shell
git lfs pull
```


## Example Usage

#### Tracking _With_ Trail Stabilization
```shell
python vision_cli.py videos/rc_demo.mp4 \
    --show-preview \
    --preview-size 1280x720 \
    --save-video \
    --show-plot \
    --show-triangle \
    --triangle-size 10x10 \
    --triangle-thickness 2 \
    --show-trace \
    --start-time 5 \
    --yolo-size 1280x720 \
    --out-size 1280x720 \
    --vid-stride 1 \
    --trail-stabilize \
    --trace-length 500 \
    --trace-thickness 3
```

https://github.com/user-attachments/assets/3e4b13fb-ba4a-490b-b6f0-dbb220c004b7


#### Tracking _Without_ Trail Stabilization
```shell
python vision_cli.py videos/rc_demo.mp4 \
    --show-preview \
    --preview-size 1280x720 \
    --save-video \
    --show-plot \
    --show-triangle \
    --triangle-size 10x10 \
    --triangle-thickness 2 \
    --show-trace \
    --start-time 5 \
    --yolo-size 1280x720 \
    --out-size 1280x720 \
    --vid-stride 1 \
    --trace-length 500 \
    --trace-thickness 3
```

https://github.com/user-attachments/assets/0ac61615-6f46-409c-8e9a-684203c78155


#### Basic Bounding Box + ID
```shell
python vision_cli.py videos/rc_demo.mp4 \
    --show-preview \
    --preview-size 1280x720 \
    --save-video \
    --show-plot \
    --show-box \
    --show-label \
    --start-time 5
```

https://github.com/user-attachments/assets/27e4df76-921c-43b2-812b-90c1e34be439


#### No Tracking (Detection Only)
```shell
python vision_cli.py videos/rc_demo.mp4 \
    --show-preview \
    --preview-size 1280x720 \
    --save-video \
    --show-plot \
    --show-box \
    --show-label \
    --start-time 5 \
    --no-track \
    --conf 0.4
```

https://github.com/user-attachments/assets/c434f2e2-ea9d-42cf-8cfb-1cb85f60471f

