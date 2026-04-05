# Air Writer Studio

Air Writer Studio now has two ways to use it:

- A browser version that can be hosted publicly on Vercel so anyone can use it from a modern browser.
- A local desktop-style Python version that can also publish to a virtual camera for Zoom, Google Meet, Microsoft Teams, and similar apps.

## Browser version

The browser app uses your webcam directly on the page and supports the same core gesture flow:

- Right hand, only index finger open: draw in the air.
- Right hand, only middle finger open: open the color palette and choose with the middle fingertip.
- Right hand, only ring finger open: open the tool palette and choose with the ring fingertip.
- Right hand, only little finger open: save a screenshot.
- Right hand, open palm: erase.
- Left hand, thumb and index active: pinch closer to make the brush and eraser smaller, or spread them apart to make them larger.

Important:

- The web version works in the browser with webcam permission.
- A website cannot become a system-wide Zoom camera device by itself. For actual Zoom camera output, use the desktop version with OBS Virtual Camera.

## What it does

- Right hand, only index finger open: draw in the air.
- Right hand, only middle finger open: open the color palette and choose with the middle fingertip.
- Right hand, only ring finger open: open the tool palette and choose with the ring fingertip.
- Right hand, only little finger open: save a screenshot.
- Right hand, open palm: erase.
- Left hand, thumb and index active: pinch closer to make the brush and eraser smaller, or spread them apart to make them larger.

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
python desktop_app.py --virtual-camera --windowed
```

or just double-click:

- `Launch Air Writer Virtual Camera.bat`

4. In Zoom, Meet, or Teams, choose the virtual camera device created by your backend, such as `OBS Virtual Camera`.

## Command-line options

```powershell
python desktop_app.py --help
```

Useful options:

- `--virtual-camera`: enable virtual camera output
- `--virtual-backend obs`: force the OBS backend
- `--virtual-device "OBS Virtual Camera"`: select a specific device
- `--windowed`: use a normal preview window instead of fullscreen
- `--no-preview`: run only the virtual camera feed without the local preview window
- `--camera-index 0`: choose which physical webcam to use

## Files

- `index.html`: Vercel/browser app entry page
- `styles.css`: browser app styles
- `web-app.js`: browser app hand tracking logic
- `desktop_app.py`: main desktop Air Writer application
- `models/hand_landmarker.task`: bundled MediaPipe hand model
- `Launch Air Writer.bat`: preview launcher
- `Launch Air Writer Virtual Camera.bat`: Zoom/Meet launcher
- `captures/`: screenshots saved from the app

## Notes

- The project uses OpenCV, MediaPipe Tasks, and `pyvirtualcam`.
- If virtual camera mode fails with a backend error, install OBS Studio or Unity Capture first and relaunch.
- Good lighting improves gesture detection, but the drawing palette is intentionally dark and high-contrast so it stays visible against bright backgrounds.
