# AirCo Surveillance System - Setup Guide

## 📋 Requirements

### Python Dependencies
Create a `requirements.txt` file with the following:

```txt
opencv-python==4.8.1.78
face-recognition==1.3.0
numpy==1.24.3
Pillow==10.0.1
dlib==19.24.2
cmake==3.27.7
```

### System Requirements
- **Python 3.8+**
- **Camera/Webcam** (built-in or USB)
- **Windows/Linux/Mac** (tested on all platforms)

## 🚀 Installation Steps

### 1. Install Dependencies
```bash
# Install requirements
pip install -r requirements.txt

# If face_recognition installation fails on Windows:
pip install --upgrade pip
pip install cmake
pip install dlib
pip install face_recognition
```

### 2. Directory Structure
The system will automatically create these directories:
```
project/
├── surveillance_system.py       # Main application
├── surveillance.db             # SQLite database (auto-created)
├── surveillance.log            # System logs
├── alerts.log                  # Alert logs
├── employee_photos/            # Employee registration photos
├── attendance_photos/          # Daily attendance photos
└── unknown_faces/              # Unknown person alerts
```

## 📖 How to Use

### Step 1: Register Employees
```bash
python surveillance_system.py
# Choose option 1: Register New Employee
# Follow the on-screen instructions to capture face photos
```

### Step 2: Start Surveillance
```bash
python surveillance_system.py
# Choose option 2: Start Surveillance System
# System will start monitoring with your camera
```

### Step 3: Monitor Results
- **Known Person**: Green box, attendance logged
- **Unknown Person**: Red box, alert sent, photo saved
- **Real-time Stats**: FPS, detected faces, system info

## 🎯 Key Features

### ✅ Employee Registration
- Interactive face capture (5 photos per person)
- Multiple angle training for better accuracy
- Stores in SQLite database with metadata

### ✅ Real-time Monitoring
- Live face detection and recognition
- Attendance logging with timestamps
- Unknown person alerts with photo capture

### ✅ Smart Features
- **Cooldown System**: Prevents spam attendance logs
- **Confidence Scoring**: Shows recognition accuracy
- **Performance Optimized**: Processes every 3rd frame
- **Multi-face Support**: Handles multiple people

### ✅ Data Management
- SQLite database for all records
- Today's attendance viewing
- Employee management
- Alert logging

## 🔧 Configuration Options

### Camera Settings
```python
# In surveillance_system.py, modify these values:
camera_id = 0  # Change to 1, 2, etc. for different cameras
recognition_threshold = 0.6  # Lower = stricter recognition
```

### Performance Tuning
```python
# Process every Nth frame (higher = faster but less accurate)
if self.frame_count % 3 == 0:  # Change 3 to 1, 2, 4, 5...
```

## 🎮 Keyboard Controls

### During Surveillance:
- **'q'**: Quit system
- **'r'**: Reload known faces from database
- **'s'**: Show current statistics

### During Registration:
- **SPACE**: Capture face photo
- **'q'**: Quit registration

## 📊 Database Schema

### Employees Table
```sql
CREATE TABLE employees (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT,
    position TEXT,
    face_encoding BLOB,
    phone TEXT,
    email TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Attendance Table
```sql
CREATE TABLE attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    action TEXT DEFAULT 'entry',
    confidence REAL,
    image_path TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees (id)
);
```

### Unknown Faces Table
```sql
CREATE TABLE unknown_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    alert_sent BOOLEAN DEFAULT FALSE,
    notes TEXT
);
```

## 🔍 Troubleshooting

### Common Issues:

1. **Camera not working**
   ```python
   # Try different camera IDs
   camera_id = 1  # or 2, 3, etc.
   ```

2. **Face recognition not accurate**
   ```python
   # Adjust recognition threshold
   recognition_threshold = 0.5  # stricter
   recognition_threshold = 0.7  # more lenient
   ```

3. **Performance issues**
   ```python
   # Process fewer frames
   if self.frame_count % 5 == 0:  # every 5th frame
   ```

4. **Installation problems**
   ```bash
   # On Windows, install Visual C++ Build Tools
   # On Mac, install Xcode Command Line Tools
   # On Linux, install python3-dev
   ```

## 🎯 Usage Examples

### Basic Workflow:
1. **Register 3-5 employees** with their faces
2. **Start surveillance system**
3. **Employees walk by camera** → Automatic attendance
4. **Unknown person appears** → Alert + photo saved
5. **View reports** through menu options

### Expected Output:
```
✓ Attendance: John Doe (0.85)
✓ Attendance: Jane Smith (0.92)
🚨 SECURITY ALERT: Unknown person detected at 2024-01-15 14:30:25
```

## 🚀 Next Steps

### Enhancements you can add:
1. **Email notifications** for unknown persons
2. **Web dashboard** for remote monitoring
3. **Multiple camera support**
4. **Entry/exit tracking**
5. **Integration with access control systems**

##