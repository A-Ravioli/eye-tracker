# Eye Tracking Application

A Python-based eye tracking application that uses your webcam to track eye movements and visualize gaze position on screen. The application uses MediaPipe for face mesh detection and OpenCV for camera handling.

## Features

- Real-time eye tracking using webcam
- Interactive calibration process
- Multiple visualization styles
- Gaze position overlay on screen
- Face and eye region visualization
- Calibration data persistence
- User-friendly desktop interface

## Requirements

- Python 3.7+
- Webcam
- macOS, Windows, or Linux

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/eye-tracker.git
cd eye-tracker
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python examples/gaze_overlay.py
```

2. The main setup window will appear showing:
   - Camera status
   - Calibration status
   - Instructions
   - Control buttons

3. Calibration Process:
   - Click "Start Calibration"
   - Look at each red dot as it appears on screen
   - Keep your head relatively still
   - The process takes about 30 seconds
   - A preview window shows your face tracking
   - Press ESC to cancel calibration

4. Eye Tracking:
   - After successful calibration, click "Start Tracking"
   - A transparent overlay will show where you're looking
   - A preview window shows face tracking
   - Press 'Q' to stop tracking and return to setup

## Tips for Best Results

- Ensure good lighting on your face
- Sit at a comfortable distance from the screen (about arm's length)
- Try to minimize head movement during tracking
- Calibrate in the same position you plan to use the tracker
- Recalibrate if you change position significantly

## Visualization Styles

The application supports different visualization styles for the gaze overlay:

- **Natural**: Large, soft circles showing approximate gaze area
- **Precise**: Smaller, more precise indicators
- **Minimal**: Simple dot without effects

## Troubleshooting

### Camera Issues
- Ensure your webcam is connected and working
- Check camera permissions in your system settings
- On macOS, the app may need camera access permission

### Tracking Issues
- Ensure good lighting conditions
- Adjust your distance from the camera
- Try recalibrating if tracking seems off
- Keep your head relatively still during use

### Performance Issues
- Close other applications using the camera
- Ensure your system meets the minimum requirements
- Try reducing the preview window size if needed

## Technical Details

The application uses several key technologies:

- **MediaPipe**: For face mesh detection and landmark tracking
- **OpenCV**: For camera handling and image processing
- **Tkinter**: For the user interface
- **NumPy**: For numerical computations and transformations

The eye tracking process involves:
1. Face detection using MediaPipe Face Mesh
2. Eye region extraction and iris position calculation
3. Gaze direction estimation
4. Screen coordinate mapping using calibration data
5. Smoothing and visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
