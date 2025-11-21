"""
Dataset Preparation for Exercise Classification

This script prepares training data from FlexTail sensor recordings.
It converts RSF files and CSV data into windowed sequences for model training.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import json


class ExerciseDatasetBuilder:
    """
    Builds a dataset from FlexTail sensor recordings.

    The dataset consists of time windows extracted from continuous recordings,
    with each window labeled by the exercise being performed.
    """

    def __init__(
        self,
        data_dir="data/recordings",
        window_size=150,  # 3 seconds at 50Hz
        stride=50,        # 1 second stride
        features=['lumbarAngle', 'twist', 'lateral', 'sagittal', 'lateralApprox', 'sagittalApprox', 'acceleration', 'gyro']
    ):
        """
        Args:
            data_dir (str): Directory containing recording files
            window_size (int): Number of timesteps per window
            stride (int): Stride between consecutive windows
            features (list): List of feature names to extract
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.features = features
        self.label2id = {}
        self.id2label = {}

    def load_csv_recording(self, file_path, exercise_label):
        """
        Load a CSV recording file.

        Expected CSV format:
        timestamp,lumbarAngle,sagittal,lateral,twist,acceleration

        Args:
            file_path (str): Path to CSV file
            exercise_label (str): Exercise label for this recording

        Returns:
            numpy array of shape (timesteps, num_features)
        """
        df = pd.read_csv(file_path)

        # Extract feature columns
        data = df[self.features].values

        return data

    def load_rsf_recording(self, file_path, exercise_label):
        """
        Load an RSF (Raw Sensor File) using flexlib.

        Args:
            file_path (str): Path to RSF file
            exercise_label (str): Exercise label for this recording

        Returns:
            numpy array of shape (timesteps, num_features)
        """
        try:
            import flexlib as fl

            # Try RSFV1Reader (most common)
            recordings = None
            try:
                reader = fl.RSFV1Reader()
                recordings = reader.parse(str(file_path))
            except:
                # Fall back to RSFV2Reader
                try:
                    reader = fl.RSFV2Reader()
                    recordings = reader.parse(str(file_path))
                except Exception as e:
                    print(f"    Error: Could not parse {Path(file_path).name}: {e}")
                    return None

            if recordings is None:
                return None

            data = []

            # AnnotatedRecording has a measurements attribute
            if hasattr(recordings, 'measurements'):
                measurements = recordings.measurements
            elif hasattr(recordings, '__iter__'):
                measurements = recordings
            else:
                measurements = [recordings]

            for measurement in measurements:
                # Get angles from measurement
                angles = measurement.angles if hasattr(measurement, 'angles') else None

                if angles is None:
                    continue

                # Calculate acceleration magnitude (acc is a list [x, y, z])
                acc_mag = 0.0
                if hasattr(measurement, 'acc') and isinstance(measurement.acc, list) and len(measurement.acc) >= 3:
                    acc_mag = (measurement.acc[0]**2 + measurement.acc[1]**2 + measurement.acc[2]**2)**0.5

                # Calculate gyro magnitude (gyro is a list [x, y, z])
                gyro_mag = 0.0
                if hasattr(measurement, 'gyro') and measurement.gyro and isinstance(measurement.gyro, list) and len(measurement.gyro) >= 3:
                    gyro_mag = (measurement.gyro[0]**2 + measurement.gyro[1]**2 + measurement.gyro[2]**2)**0.5

                # Extract all 8 features with CORRECT attribute names
                row = [
                    angles.bend if hasattr(angles, 'bend') else 0.0,                                          # lumbarAngle
                    angles.twist if hasattr(angles, 'twist') else 0.0,                                        # twist
                    measurement.lateral_flexion if hasattr(measurement, 'lateral_flexion') else 0.0,          # lateral
                    measurement.sagittal_flexion if hasattr(measurement, 'sagittal_flexion') else 0.0,        # sagittal
                    measurement.calc_lateral_approx() if hasattr(measurement, 'calc_lateral_approx') else 0.0, # lateralApprox
                    measurement.calc_sagittal_approx() if hasattr(measurement, 'calc_sagittal_approx') else 0.0, # sagittalApprox
                    acc_mag,                                                                                   # acceleration
                    gyro_mag                                                                                   # gyro
                ]
                data.append(row)

            if not data:
                print(f"    Warning: No data extracted from {Path(file_path).name}")
                return None

            return np.array(data)

        except Exception as e:
            print(f"    Error loading {Path(file_path).name}: {e}")
            return None

    def extract_windows(self, data, label):
        """
        Extract sliding windows from a recording.

        Args:
            data (np.array): Recording data of shape (timesteps, features)
            label (str): Exercise label

        Returns:
            tuple: (windows, labels) where windows has shape (num_windows, window_size, features)
        """
        if len(data) < self.window_size:
            print(f"Warning: Recording too short ({len(data)} < {self.window_size})")
            return [], []

        windows = []
        labels = []

        for i in range(0, len(data) - self.window_size + 1, self.stride):
            window = data[i:i + self.window_size]

            # Normalize window (z-score normalization)
            window_mean = window.mean(axis=0, keepdims=True)
            window_std = window.std(axis=0, keepdims=True) + 1e-8
            window_normalized = (window - window_mean) / window_std

            windows.append(window_normalized)
            labels.append(label)

        return windows, labels

    def build_dataset(self):
        """
        Build the complete dataset from all recordings.

        Expected directory structure:
        data/recordings/
            squat/
                recording1.csv
                recording2.rsf
                ...
            deadlift/
                recording1.csv
                ...
            plank/
                ...

        Returns:
            dict: Dataset with keys 'X_train', 'X_val', 'X_test', 'y_train', etc.
        """
        all_windows = []
        all_labels = []

        # Get all exercise directories
        exercise_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        if not exercise_dirs:
            print(f"Error: No exercise directories found in {self.data_dir}")
            print("Please create subdirectories for each exercise type.")
            return None

        # Build label mappings
        exercises = sorted([d.name for d in exercise_dirs])
        self.label2id = {label: idx for idx, label in enumerate(exercises)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        print(f"Found {len(exercises)} exercise types: {exercises}")

        # Process each exercise directory
        for exercise_dir in exercise_dirs:
            exercise_label = exercise_dir.name
            print(f"\nProcessing {exercise_label}...")

            # Get all recording files
            csv_files = list(exercise_dir.glob("*.csv"))
            rsf_files = list(exercise_dir.glob("*.rsf"))

            total_files = len(csv_files) + len(rsf_files)
            print(f"  Found {total_files} recordings ({len(csv_files)} CSV, {len(rsf_files)} RSF)")

            # Process CSV files
            for csv_file in csv_files:
                try:
                    data = self.load_csv_recording(csv_file, exercise_label)
                    windows, labels = self.extract_windows(data, exercise_label)
                    all_windows.extend(windows)
                    all_labels.extend(labels)
                    print(f"    {csv_file.name}: {len(windows)} windows")
                except Exception as e:
                    print(f"    Error loading {csv_file.name}: {e}")

            # Process RSF files
            for rsf_file in rsf_files:
                try:
                    data = self.load_rsf_recording(rsf_file, exercise_label)
                    if data is not None:
                        windows, labels = self.extract_windows(data, exercise_label)
                        all_windows.extend(windows)
                        all_labels.extend(labels)
                        print(f"    {rsf_file.name}: {len(windows)} windows")
                except Exception as e:
                    print(f"    Error loading {rsf_file.name}: {e}")

        if not all_windows:
            print("\nError: No data was loaded. Please check your recording files.")
            return None

        # Convert to numpy arrays
        X = np.array(all_windows)
        y = np.array([self.label2id[label] for label in all_labels])

        print(f"\n=== Dataset Summary ===")
        print(f"Total windows: {len(X)}")
        print(f"Shape: {X.shape}")
        print(f"Class distribution:")
        for label, idx in self.label2id.items():
            count = np.sum(y == idx)
            print(f"  {label}: {count} ({100*count/len(y):.1f}%)")

        # Split into train/val/test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'label2id': self.label2id,
            'id2label': self.id2label,
            'features': self.features,
            'window_size': self.window_size
        }

    def save_dataset(self, dataset, output_dir="data/processed"):
        """Save processed dataset to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save arrays
        np.save(output_path / "X_train.npy", dataset['X_train'])
        np.save(output_path / "y_train.npy", dataset['y_train'])
        np.save(output_path / "X_val.npy", dataset['X_val'])
        np.save(output_path / "y_val.npy", dataset['y_val'])
        np.save(output_path / "X_test.npy", dataset['X_test'])
        np.save(output_path / "y_test.npy", dataset['y_test'])

        # Save metadata
        metadata = {
            'label2id': dataset['label2id'],
            'id2label': dataset['id2label'],
            'features': dataset['features'],
            'window_size': dataset['window_size'],
            'num_classes': len(dataset['label2id']),
            'input_size': len(dataset['features'])
        }

        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nDataset saved to {output_path}")

    def load_dataset(self, data_dir="data/processed"):
        """Load processed dataset from disk."""
        data_path = Path(data_dir)

        dataset = {
            'X_train': np.load(data_path / "X_train.npy"),
            'y_train': np.load(data_path / "y_train.npy"),
            'X_val': np.load(data_path / "X_val.npy"),
            'y_val': np.load(data_path / "y_val.npy"),
            'X_test': np.load(data_path / "X_test.npy"),
            'y_test': np.load(data_path / "y_test.npy")
        }

        with open(data_path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        dataset.update(metadata)
        self.label2id = dataset['label2id']
        self.id2label = {int(k): v for k, v in dataset['id2label'].items()}

        return dataset


def create_sample_data():
    """
    Create sample synthetic data for testing.

    This generates fake sensor data for demonstration purposes.
    Replace this with real FlexTail recordings for actual training.
    """
    print("Creating synthetic sample data...")

    exercises = ['squat', 'deadlift', 'plank', 'pushup', 'row', 'burpee']
    data_dir = Path("data/recordings")

    # Create directory structure
    for exercise in exercises:
        exercise_dir = data_dir / exercise
        exercise_dir.mkdir(parents=True, exist_ok=True)

        # Generate 5 sample recordings per exercise
        for i in range(5):
            # Generate synthetic sensor data
            num_timesteps = 500
            time = np.linspace(0, 10, num_timesteps)

            # Create exercise-specific patterns
            if exercise == 'squat':
                lumbar = 0.3 * np.sin(2 * np.pi * 0.5 * time) + 0.2
                sagittal = 0.2 * np.sin(2 * np.pi * 0.5 * time)
                acceleration = 0.5 * np.abs(np.cos(2 * np.pi * 0.5 * time))
            elif exercise == 'deadlift':
                lumbar = 0.4 * np.sin(2 * np.pi * 0.3 * time) + 0.3
                sagittal = 0.5 * np.sin(2 * np.pi * 0.3 * time) + 0.2
                acceleration = 0.4 * np.abs(np.cos(2 * np.pi * 0.3 * time))
            elif exercise == 'plank':
                lumbar = 0.1 * np.random.randn(num_timesteps) + 0.15
                sagittal = 0.05 * np.random.randn(num_timesteps)
                acceleration = 0.1 * np.abs(np.random.randn(num_timesteps))
            else:
                lumbar = 0.2 * np.sin(2 * np.pi * 0.7 * time) + 0.1
                sagittal = 0.3 * np.sin(2 * np.pi * 0.7 * time)
                acceleration = 0.6 * np.abs(np.cos(2 * np.pi * 0.7 * time))

            lateral = 0.1 * np.random.randn(num_timesteps)
            twist = 0.05 * np.random.randn(num_timesteps)

            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': time,
                'lumbarAngle': lumbar,
                'sagittal': sagittal,
                'lateral': lateral,
                'twist': twist,
                'acceleration': acceleration
            })

            # Save to CSV
            output_file = exercise_dir / f"recording_{i+1}.csv"
            df.to_csv(output_file, index=False)

    print(f"Sample data created in {data_dir}")
    print("Replace this with real FlexTail recordings for production use.")


if __name__ == "__main__":
    # Example usage
    print("=== FlexTail Exercise Dataset Builder ===\n")

    # Create sample data if no data exists
    if not Path("data/recordings").exists() or not any(Path("data/recordings").iterdir()):
        create_sample_data()

    # Build dataset
    builder = ExerciseDatasetBuilder(
        data_dir="data/recordings",
        window_size=150,
        stride=50
    )

    dataset = builder.build_dataset()

    if dataset:
        # Save dataset
        builder.save_dataset(dataset)

        print("\n=== Dataset ready for training! ===")
        print("Next steps:")
        print("1. Replace synthetic data with real FlexTail recordings")
        print("2. Run train_model.py to train the classifier")
        print("3. Integrate the trained model into the Flask backend")
