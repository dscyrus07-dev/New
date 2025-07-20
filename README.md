# AirCo Secure Surveillance System

A comprehensive face recognition-based surveillance system for employee tracking and security monitoring.

## Features

- **Face Recognition**: Real-time face detection and recognition using dlib and face_recognition libraries
- **Employee Management**: Register and manage employees with multiple face encodings
- **Attendance Tracking**: Automatic attendance logging with timestamps and confidence scores
- **Unknown Face Detection**: Alerts and logs for unrecognized faces
- **Multi-threaded Processing**: Optimized for performance with GPU acceleration support
- **Database Integration**: SQLite database for storing employee data and attendance records
- **Email Alerts**: Configurable email notifications for security events

## Prerequisites

- Python 3.8+
- CMake (required for dlib installation)
- C++ Build Tools (for Windows)
- CUDA Toolkit (optional, for GPU acceleration)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd face-recognition
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Copy `.env.example` to `.env` and update the settings:
   ```
   EMAIL_ENABLED=False
   SMTP_SERVER=smtp.example.com
   SMTP_PORT=587
   EMAIL_ADDRESS=your@email.com
   EMAIL_PASSWORD=yourpassword
   ALERT_EMAILS=admin@example.com,security@example.com
   ```

2. For GPU acceleration, install the appropriate version of `cupy` based on your CUDA version:
   ```bash
   pip install cupy-cuda11x  # Replace x with your CUDA version
   ```

## Usage

### Registering Employees

1. Run the registration script:
   ```bash
   python airco.py --register
   ```

2. Follow the prompts to enter employee details and capture face images.

### Running the Surveillance System

1. For webcam:
   ```bash
   python airco.py
   ```

2. For RTSP camera stream:
   ```bash
   python airco.py --rtsp rtsp://username:password@camera-ip:port/stream
   ```

### Command Line Options

- `--register`: Start employee registration mode
- `--rtsp URL`: Use RTSP stream instead of webcam
- `--threshold FLOAT`: Set recognition threshold (default: 0.6)
- `--model [cnn|hog]`: Face detection model (default: cnn for GPU, hog for CPU)
- `--no-gpu`: Disable GPU acceleration

## Directory Structure

```
face-recognition/
├── airco.py              # Main application
├── requirements.txt      # Python dependencies
├── surveillance.db       # SQLite database
├── employee_photos/      # Employee face images
├── attendance_photos/    # Captured attendance photos
├── unknown_faces/        # Detected unknown faces
└── logs/                 # System logs
```

## Performance Tips

- Use `cnn` model with GPU for better accuracy (requires CUDA)
- Adjust `recognition_threshold` to balance between false positives and false negatives
- For large deployments, consider using a more robust database like PostgreSQL

## Troubleshooting

1. **Installation issues with dlib**:
   - Ensure you have CMake and C++ build tools installed
   - Try installing dlib with pip: `pip install dlib --verbose`

2. **Face detection not working**:
   - Check camera permissions
   - Ensure proper lighting conditions
   - Try different face detection models (`--model hog` or `--model cnn`)

3. **Low recognition accuracy**:
   - Capture multiple face images from different angles
   - Ensure good lighting during registration
  - Adjust the recognition threshold

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [face_recognition](https://github.com/ageitgey/face_recognition)
- Uses [dlib](http://dlib.net/) for facial landmark detection
- Inspired by various open-source face recognition projects
