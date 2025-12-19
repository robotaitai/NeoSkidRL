# Visual Demo Instructions

This project includes a visual demo that shows the RGB camera and a lidar plot while running a simple policy.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Option 1: Live camera + lidar windows (requires a GUI)

```bash
pip install -e .[viz]
python -m neoskidrl.scripts.visual_demo --mode live --policy heuristic --steps 800 --config config/static_goal.yml
```

You can also run with a trained policy:

```bash
pip install -e .[train,viz]
python -m neoskidrl.scripts.visual_demo --mode live --policy sac --model runs/skidnav_sac.zip
```

If you are in a headless environment, use the frames mode below and set an offscreen backend if needed.

## Option 2: Save rendered frames (no GUI)

RGB frames (headless example uses EGL):

```bash
MUJOCO_GL=egl python -m neoskidrl.scripts.visual_demo --mode frames --render rgb --steps 300 --outdir runs/demo_frames
```

Depth frames (grayscale PGM):

```bash
MUJOCO_GL=egl python -m neoskidrl.scripts.visual_demo --mode frames --render depth --steps 300 --outdir runs/demo_frames_depth
```

The output directory will contain `frame_0000.ppm` (RGB) or `frame_0000.pgm` (depth) and subsequent frames.
If EGL is not available on your system, try `MUJOCO_GL=osmesa`.
