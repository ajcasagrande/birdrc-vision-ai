# BirdRC Vision AI
RC Car detection using Computer Vision and YOLOv11

```shell
pip install -r requirements.txt
```

Make sure you have `git-lfs` installed, and then run the following to download the sample model and data:

```shell
git lfs pull
```

### Example Usage
#### Trail stabilization
```shell
python vision_cli.py videos/rc_demo.mp4 \
    --show_preview \
    --preview_size 1280x720 \
    --save_video \
    --show_plot \
    --show_triangle \
    --triangle_size 10x10 \
    --triangle_thickness 2 \
    --show_trace \
    --start_time 5 \
    --yolo_size 1280x720 \
    --out_size 1280x720 \
    --vid_stride 1 \
    --trail_stabilize \
    --trace_length 500 \
    --trace_thickness 3
```

