# Air Writer Studio

Air Writer Studio is a local webcam drawing app that lets you write in the air with hand gestures and can also publish its output as a virtual camera for Zoom, Google Meet, Microsoft Teams, and other camera-based apps.

## What it does

- Left hand, only index finger open: draw in the air.
- Left hand, only middle finger open: open the color palette and choose with the middle fingertip.
- Left hand, only ring finger open: open the tool palette and choose with the ring fingertip.
- Left hand, only little finger open: save a screenshot.
- Left hand, open palm: erase.
- Right hand, thumb and index active: pinch closer to make the brush and eraser smaller, or spread them apart to make them larger.

## One-click launchers

- `Launch Air Writer.bat`: opens the normal fullscreen local app.
- `Launch Air Writer Virtual Camera.bat`: opens the app and also publishes a virtual camera feed for Zoom and similar apps.

## Virtual camera use

To use Air Writer inside Zoom, Meet, Teams, or other camera apps:

1. Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

2. Install a Windows virtual camera backend.

Recommended:
- OBS Studio with OBS Virtual Camera

Also supported by `pyvirtualcam`:
- Unity Capture

3. Start the app with:

```powershell
python app.py --virtual-camera --windowed
```

or just double-click:

- `Launch Air Writer Virtual Camera.bat`

4. In Zoom, Meet, or Teams, choose the virtual camera device created by your backend, such as `OBS Virtual Camera`.

## Command-line options

```powershell
python app.py --help
```

Useful options:

- `--virtual-camera`: enable virtual camera output
- `--virtual-backend obs`: force the OBS backend
- `--virtual-device "OBS Virtual Camera"`: select a specific device
- `--windowed`: use a normal preview window instead of fullscreen
- `--no-preview`: run only the virtual camera feed without the local preview window
- `--camera-index 0`: choose which physical webcam to use

## Files

- `app.py`: main Air Writer application
- `models/hand_landmarker.task`: bundled MediaPipe hand model
- `Launch Air Writer.bat`: preview launcher
- `Launch Air Writer Virtual Camera.bat`: Zoom/Meet launcher
- `captures/`: screenshots saved from the app

## Notes

- The project uses OpenCV, MediaPipe Tasks, and `pyvirtualcam`.
- If virtual camera mode fails with a backend error, install OBS Studio or Unity Capture first and relaunch.
- Good lighting improves gesture detection, but the drawing palette is intentionally dark and high-contrast so it stays visible against bright backgrounds.
