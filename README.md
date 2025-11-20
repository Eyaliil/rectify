# Rectify - AI-Powered Exercise Analysis System

An intelligent web application that uses AI and sensor data analysis to automatically recognize exercises and assess movement quality in real-time. Features 3D body visualization, quality metrics, automatic feedback, and a clean pipeline ready for live sensor data.

## Features

### Core Functionality

1. **Exercise Recognition**
   - Automatically detects which exercise is being performed (squat, deadlift, plank, push-up)
   - Real-time classification with confidence scores
   - Manual exercise selection option

2. **Quality Assessment**
   - **Depth**: Measures range of motion
   - **Stability**: Evaluates movement control and consistency
   - **Posture**: Analyzes spinal alignment and form
   - **Symmetry**: Detects lateral imbalances
   - **Overall Score**: Composite quality metric

3. **Real-time Feedback**
   - Automatic warnings for poor form (e.g., "Warning: back too rounded")
   - Exercise-specific suggestions
   - Color-coded quality indicators (green/orange/red)

4. **3D Body Visualization**
   - Real-time 3D skeleton visualization using Three.js
   - Visual representation of posture and movement
   - Color-coded feedback based on quality scores

5. **Extensible Data Pipeline**
   - Clean public API for streaming live sensor packets
   - Modular architecture ready for ML model integration
   - UI remains idle until external data is supplied

## Prerequisites

- Node.js (v14 or higher)
- npm (comes with Node.js)

## Installation

1. Install dependencies:
```bash
npm install
```

## Running the Application

Start the server:
```bash
npm start
```

The application will be available at:
```
http://localhost:4000
```

## Usage

1. **Start Analysis**: Click "Start Analysis" to arm the pipeline
2. **Provide Data**: Stream sensor frames via the exposed API (see below)
3. **Select Exercise**: Choose a specific exercise or use "Auto-detect"
4. **View Metrics**: Monitor real-time quality scores and feedback

## Project Structure

```
rectify/
├── server.js              # Express server configuration
├── package.json           # Project dependencies
├── README.md             # This file
└── public/
    ├── index.html        # Main dashboard UI
    ├── style.css         # Styling
    ├── main.js           # Main application controller & data entry point
    ├── exerciseAnalyzer.js    # Exercise recognition & quality assessment
    └── bodyVisualizer.js      # 3D body visualization (Three.js)
```

## Sensor Data

The system processes sensor data from FlexTail sensors. The data format is automatically adapted from the raw sensor format to the internal format.

### Raw Sensor Data Format (from FlexTail)

The application accepts data in the following format (from `.rsf` files processed via `flexlib`):

```javascript
{
  lumbarAngle: number,    // Lumbar spine angle in radians
  sagittal: number,        // Sagittal tilt in radians (forward/backward lean)
  lateral: number,         // Lateral tilt in radians (left/right lean)
  twist: number,           // Rotation/twist in radians
  acceleration: number,    // Acceleration magnitude (single value)
  gyro: number,           // Gyroscope reading (optional)
  thoracicAngle: number    // Thoracic spine angle in radians (optional)
}
```

### Internal Data Format

The system automatically converts the raw data to internal format:
- Angles are converted from radians to degrees
- Acceleration is normalized to a 3D vector format
- Field names are mapped (lumbarAngle → lumbarLordosis, etc.)

See `DATA_FORMAT.md` for detailed documentation on the data structure.

## Exercise Recognition

The system uses feature extraction and rule-based classification (extensible to ML models) to identify exercises based on movement patterns:

- **Squat**: High lordosis change, vertical movement pattern
- **Deadlift**: Forward lean, high lordosis, vertical pull
- **Plank**: Low movement, neutral spine, high stability
- **Push-up**: Horizontal movement, moderate lordosis

## Quality Metrics

Each exercise is assessed on multiple dimensions:

- **Depth**: Range of motion (squat depth, push-up depth, etc.)
- **Stability**: Movement consistency and control
- **Posture**: Spinal alignment and form maintenance
- **Symmetry**: Bilateral balance and alignment

## Feedback System

The system generates contextual feedback based on:

- Overall performance thresholds
- Exercise-specific form requirements
- Real-time error detection
- Quality metric analysis

## Feeding Sensor Data

After clicking "Start Analysis", stream each sensor frame into the global helper. The system accepts data in the FlexTail sensor format (angles in radians):

```js
// FlexTail sensor format (recommended)
window.app.ingestSensorData({
  lumbarAngle: -0.816,    // radians
  sagittal: 0.678,         // radians
  lateral: -0.032,        // radians
  twist: 0.175,           // radians
  acceleration: 0.049,     // magnitude (single value)
  gyro: 0.24              // optional
});
```

The system automatically converts angles from radians to degrees and adapts the field names. You can also use the internal format directly:

```js
// Internal format (also supported)
window.app.ingestSensorData({
  lumbarLordosis: 32.1,    // degrees
  sagittalTilt: 4.5,      // degrees
  lateralTilt: -1.8,     // degrees
  rotation: 0.6,         // degrees
  acceleration: { x: 0.12, y: -1.03, z: 0.05 } // or single magnitude value
});
```

Connect this helper to your preferred transport (WebSocket, BLE, native bridge, etc.). The dashboard will update in real time with whatever data you provide.

See `DATA_FORMAT.md` for complete documentation on the data structure.

## Technologies Used

- **Express.js** - Web server
- **Three.js** - 3D graphics and visualization
- **JavaScript (ES6 Modules)** - Application logic
- **TensorFlow.js** (ready for ML model integration) - Machine learning
- **HTML/CSS** - Frontend interface

## Future Enhancements

- Integration with trained ML models for improved accuracy
- Real sensor data integration (IMU, motion capture)
- Advanced anomaly detection for injury prevention
- Personalized coaching recommendations
- Multi-user support and cloud sync
- Export capabilities for data analysis

## Development Notes

The current implementation uses rule-based classification for exercise recognition. The architecture is designed to easily integrate trained machine learning models (e.g., TensorFlow.js models) for improved accuracy.

Sensor data integration occurs via the `window.app.ingestSensorData()` helper described above.
