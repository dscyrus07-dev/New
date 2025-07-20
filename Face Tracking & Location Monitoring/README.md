# Face Tracking & Location Monitoring System

A comprehensive real-time face tracking and behavior analysis system that combines advanced computer vision techniques with zone monitoring and drowsiness detection capabilities.

## 🌟 Features

### Module 1: Face Tracking & Location Monitoring
- **Real-time Face Tracking**: Uses DeepSort algorithm for consistent track IDs
- **Zone Monitoring**: Define virtual zones and track movement between areas
- **Location Analytics**: Movement patterns and dwell time analysis
- **Integration**: Seamlessly works with existing face recognition systems
- **High Performance**: Optimized for 10+ FPS on CPU/GPU

### Module 2: Facial Behavior Analysis
- **Eye State Detection**: Real-time eye closure monitoring using Eye Aspect Ratio (EAR)
- **Head Pose Estimation**: 3D head pose tracking for attention analysis
- **Yawn Detection**: Mouth state analysis for fatigue detection
- **Drowsiness Scoring**: Comprehensive drowsiness analysis with alert levels
- **MediaPipe Integration**: Robust facial landmark detection

### System Integration
- **Unified Interface**: Single application combining both modules
- **Database Logging**: Comprehensive data storage and retrieval
- **Real-time Visualization**: Rich overlays and dashboards
- **Configuration Management**: Flexible YAML-based configuration
- **Export Capabilities**: CSV export and analysis tools

## 📋 Requirements

### System Requirements
- Python 3.8+
- OpenCV 4.5+
- 4GB+ RAM recommended
- CUDA-compatible GPU (optional, for better performance)

### Dependencies
```bash
# Core computer vision
opencv-python>=4.5.0
opencv-contrib-python>=4.5.0
numpy>=1.21.0
face-recognition>=1.3.0
dlib>=19.22.0

# Deep learning and tracking
torch>=1.9.0
torchvision>=0.10.0
deep-sort-realtime>=1.2.1
filterpy>=1.4.5

# Behavior analysis
mediapipe>=0.8.10

# Data processing and visualization
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
shapely>=1.8.0

# Configuration and utilities
pyyaml>=6.0
python-dotenv>=0.19.0
click>=8.0.0
loguru>=0.6.0
```

## 🚀 Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd "Face Tracking & Location Monitoring"
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install DeepSort (Optional for enhanced tracking)**
   ```bash
   pip install deep-sort-realtime
   ```

5. **Setup Directories**
   ```bash
   mkdir -p data/logs data/screenshots data/tracking_data
   ```

## ⚙️ Configuration

### Basic Configuration
Copy and modify the configuration file:
```bash
cp config/config.yaml config/my_config.yaml
```

### Zone Configuration
Define monitoring zones in the config file:
```yaml
zones:
  entry:
    name: "Entry Zone"
    color: [0, 255, 0]  # Green
    points: [[100, 100], [300, 100], [300, 200], [100, 200]]
  
  desk_area:
    name: "Desk Area" 
    color: [0, 0, 255]  # Red
    points: [[400, 150], [800, 150], [800, 400], [400, 400]]
```

### Behavior Analysis Settings
```yaml
behavior:
  eye_aspect_ratio_threshold: 0.25
  eye_closure_frames: 16
  yawn_threshold: 20.0
  head_pose_threshold: 30.0
  drowsiness_time_window: 30
```

## 🎮 Usage

### Basic Usage
```bash
python main.py
```

### Advanced Usage
```bash
# Use custom configuration
python main.py --config config/my_config.yaml

# Use IP camera
python main.py --camera "rtsp://username:password@ip:port/stream"

# Run in headless mode (no display)
python main.py --no-display

# Load known faces from custom directory
python main.py --load-faces "/path/to/employee/photos"

# Enable debug logging
python main.py --debug
```

### Interactive Controls
- **'q'** - Quit system
- **'s'** - Save screenshot with annotations
- **'r'** - Reset system state
- **'c'** - Calibrate behavior analysis baselines

## 📊 System Architecture

```
Face Tracking & Location Monitoring System
├── modules/
│   ├── face_tracking/
│   │   ├── tracker.py           # DeepSort integration
│   │   ├── zone_monitor.py      # Zone monitoring
│   │   ├── database_manager.py  # Data storage
│   │   └── visualization.py     # Tracking visualization
│   └── behavior_analysis/
│       ├── behavior_detector.py  # MediaPipe behavior detection
│       ├── drowsiness_analyzer.py # Drowsiness scoring
│       └── behavior_visualizer.py # Behavior visualization
├── config/
│   └── config.yaml              # System configuration
├── data/
│   ├── logs/                    # System logs
│   ├── screenshots/             # Saved screenshots
│   └── tracking_data/           # Tracking database
└── main.py                      # Main application
```

## 🔧 API Reference

### Face Tracker
```python
from modules.face_tracking import FaceTracker

tracker = FaceTracker(
    max_age=30,
    min_hits=3,
    face_model="hog",
    recognition_threshold=0.6
)

# Detect and track faces
detections = tracker.detect_faces(frame)
tracks = tracker.update(frame, detections)
```

### Zone Monitor
```python
from modules.face_tracking import ZoneMonitor

monitor = ZoneMonitor(zones_config)
transitions = monitor.update(tracks)
occupancy = monitor.get_all_occupancy()
```

### Behavior Detector
```python
from modules.behavior_analysis import BehaviorDetector

detector = BehaviorDetector(
    ear_threshold=0.25,
    yawn_threshold=20.0
)

behavior_data = detector.detect_behaviors(frame, face_boxes)
```

### Drowsiness Analyzer
```python
from modules.behavior_analysis import DrowsinessAnalyzer

analyzer = DrowsinessAnalyzer(
    time_window=30,
    drowsiness_threshold=60.0
)

score = analyzer.analyze_drowsiness(face_id, behavior_data)
```

## 📈 Performance Optimization

### CPU Optimization
- Use `hog` face detection model for faster processing
- Reduce frame scale factor for faster detection
- Adjust zone complexity (fewer points)
- Enable frame skipping for behavior analysis

### GPU Acceleration
- Use `cnn` face detection model with CUDA
- Install GPU-accelerated OpenCV
- Use GPU-optimized MediaPipe models

### Memory Management
- Regular cleanup of old tracking data
- Limit history buffer sizes
- Use database pagination for large datasets

## 📊 Data Analysis

### Export Data
```python
# Export tracking data to CSV
system.database.export_data_to_csv("exports/", hours=24)

# Export drowsiness analysis
analyzer.export_analysis_data(face_id, hours=1)
```

### Database Queries
```python
# Get tracks by time range
tracks = database.get_tracks_by_timerange(start_time, end_time)

# Get zone transitions
transitions = database.get_zone_transitions_by_timerange(start_time, end_time)

# Get identity statistics
stats = database.get_identity_statistics(hours=24)
```

## 🚨 Alerts and Notifications

### Drowsiness Alerts
- **Normal**: No action required
- **Mild**: Early warning indicators
- **Moderate**: Active drowsiness detected
- **Severe**: Immediate attention required

### Zone Alerts
- Unauthorized area access
- Extended occupancy periods
- Unusual movement patterns

## 🧪 Testing

### Run Unit Tests
```bash
python -m pytest tests/ -v
```

### Test Individual Modules
```bash
# Test face tracking
python -m pytest tests/test_face_tracking.py

# Test behavior analysis
python -m pytest tests/test_behavior_analysis.py
```

## 🔍 Troubleshooting

### Common Issues

1. **MediaPipe Installation Issues**
   ```bash
   pip install --upgrade mediapipe
   ```

2. **DeepSort Not Available**
   ```bash
   pip install deep-sort-realtime
   ```

3. **Camera Access Issues**
   - Check camera permissions
   - Try different camera indices (0, 1, 2...)
   - Verify RTSP URL format

4. **Performance Issues**
   - Reduce frame resolution
   - Use CPU-optimized models
   - Increase detection scale factor

### Debug Mode
Enable detailed logging:
```bash
python main.py --debug
```

## 📝 Integration with Existing Systems

### Face Recognition Integration
The system automatically integrates with the existing face recognition system:
```python
# Load from facial recognition database
system.load_known_faces("../Facial Recognition/employee_photos")
```

### Database Integration
Access existing employee data:
```python
# Connect to existing surveillance database
db_path = "../Facial Recognition/surveillance.db"
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **DeepSort**: Multi-object tracking algorithm
- **MediaPipe**: Real-time face landmark detection
- **face_recognition**: Face recognition library
- **OpenCV**: Computer vision library
- **dlib**: Machine learning algorithms

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation

---

**Note**: This system is designed for legitimate monitoring applications. Please ensure compliance with privacy laws and regulations in your jurisdiction.