# Sensor Data Format

## Data Source
The application receives sensor data from FlexTail sensors, processed from `.rsf` files using the `flexlib` library.

## Data Structure

Each data packet contains the following fields:

### Metadata (optional)
- `User`: User identifier (e.g., "Jonas", "Arwed", "Kiana", "Tom")
- `Activity`: Exercise type (e.g., "Squat", "DL" (Deadlift), "OHP" (Overhead Press), "PushUp", "Row", "Calib")
- `Rating`: Quality rating (e.g., "Better", "Worse", "Correct", "Neutral")
- `Source_File`: Source filename
- `Time`: Timestamp

### Sensor Measurements (required)
- **`lumbarAngle`**: Lumbar spine angle in radians (corresponds to lumbar lordosis)
- **`sagittal`**: Sagittal tilt angle in radians (forward/backward lean)
- **`lateral`**: Lateral tilt angle in radians (left/right lean)
- **`twist`**: Rotation/twist angle in radians
- **`acceleration`**: Acceleration magnitude (single value, not x/y/z components)
- **`gyro`**: Gyroscope reading
- **`thoracicAngle`**: Thoracic spine angle in radians
- **`lateralApprox`**: Approximated lateral tilt
- **`sagittalApprox`**: Approximated sagittal tilt

## Data Format Notes

1. **Angles are in radians**, not degrees
2. **Acceleration is a single magnitude value**, not a 3D vector
3. The data comes from processed `.rsf` files (binary format) that are parsed using `flexlib`
4. Data can be provided as:
   - Real-time streaming data
   - CSV format (from consolidated_measurements.csv)
   - JSON format (if converted)

## Exercise Types in Dataset

- **Squat**: Squat exercises
- **DL**: Deadlift
- **OHP**: Overhead Press
- **PushUp**: Push-up
- **Row**: Row exercises
- **Calib/Calibration**: Calibration/neutral position

## Quality Ratings

- **Better**: Correctly performed exercise
- **Worse**: Incorrectly performed exercise
- **Correct**: Correct form
- **Neutral**: Neutral/calibration position

