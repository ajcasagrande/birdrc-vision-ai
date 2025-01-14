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

#### Tracking _With_ Trail Stabilization
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

https://github.com/user-attachments/assets/3e4b13fb-ba4a-490b-b6f0-dbb220c004b7


#### Tracking _Without_ Trail Stabilization
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
    --trace_length 500 \
    --trace_thickness 3
```

https://github.com/user-attachments/assets/0ac61615-6f46-409c-8e9a-684203c78155


#### Basic Bounding Box + ID
```shell
python vision_cli.py videos/rc_demo.mp4 \
    --show_preview \
    --preview_size 1280x720 \
    --save_video \
    --show_plot \
    --show_box \
    --show_label \
    --start_time 5
```

https://github.com/user-attachments/assets/27e4df76-921c-43b2-812b-90c1e34be439




