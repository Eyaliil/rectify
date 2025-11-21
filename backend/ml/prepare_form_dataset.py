"""
Form Quality Dataset Preparation

Extracts form quality labels from filename patterns:
- "Better" / "Correct" → good form
- "Worse" → bad form

Creates a multi-task dataset for both exercise classification AND form quality.
"""

import os
import re
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import flexlib as fl


class FormQualityDatasetBuilder:
    """Builds dataset with both exercise type and form quality labels."""

    def __init__(
        self,
        data_dir="data/recordings",
        window_size=150,
        stride=50,
        features=['lumbarAngle', 'twist', 'lateral', 'sagittal', 'lateralApprox', 'sagittalApprox', 'acceleration', 'gyro']
    ):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.features = features

        # Exercise labels
        self.exercise_label2id = {}
        self.exercise_id2label = {}

        # Form quality labels
        self.form_label2id = {'good': 0, 'bad': 1}
        self.form_id2label = {0: 'good', 1: 'bad'}

    def extract_form_quality(self, filename):
        """
        Extract form quality from filename.

        Patterns:
        - "Better", "Correct", "Good" → good (0)
        - "Worse", "Bad", "Wrong" → bad (1)
        """
        filename_lower = filename.lower()

        if any(word in filename_lower for word in ['better', 'correct', 'good']):
            return 'good'
        elif any(word in filename_lower for word in ['worse', 'bad', 'wrong']):
            return 'bad'
        else:
            # Default to good if no indicator
            return 'good'

    def load_rsf_recording(self, file_path):
        """Load RSF file and extract features."""
        try:
            recordings = None
            try:
                reader = fl.RSFV1Reader()
                recordings = reader.parse(str(file_path))
            except:
                try:
                    reader = fl.RSFV2Reader()
                    recordings = reader.parse(str(file_path))
                except Exception as e:
                    print(f"    Error parsing {Path(file_path).name}: {e}")
                    return None

            if recordings is None:
                return None

            data = []

            if hasattr(recordings, 'measurements'):
                measurements = recordings.measurements
            elif hasattr(recordings, '__iter__'):
                measurements = recordings
            else:
                measurements = [recordings]

            for measurement in measurements:
                angles = measurement.angles if hasattr(measurement, 'angles') else None
                if angles is None:
                    continue

                # Calculate magnitudes
                acc_mag = 0.0
                if hasattr(measurement, 'acc') and isinstance(measurement.acc, list) and len(measurement.acc) >= 3:
                    acc_mag = (measurement.acc[0]**2 + measurement.acc[1]**2 + measurement.acc[2]**2)**0.5

                gyro_mag = 0.0
                if hasattr(measurement, 'gyro') and measurement.gyro and isinstance(measurement.gyro, list) and len(measurement.gyro) >= 3:
                    gyro_mag = (measurement.gyro[0]**2 + measurement.gyro[1]**2 + measurement.gyro[2]**2)**0.5

                row = [
                    angles.bend if hasattr(angles, 'bend') else 0.0,
                    angles.twist if hasattr(angles, 'twist') else 0.0,
                    measurement.lateral_flexion if hasattr(measurement, 'lateral_flexion') else 0.0,
                    measurement.sagittal_flexion if hasattr(measurement, 'sagittal_flexion') else 0.0,
                    measurement.calc_lateral_approx() if hasattr(measurement, 'calc_lateral_approx') else 0.0,
                    measurement.calc_sagittal_approx() if hasattr(measurement, 'calc_sagittal_approx') else 0.0,
                    acc_mag,
                    gyro_mag
                ]
                data.append(row)

            return np.array(data) if data else None

        except Exception as e:
            print(f"    Error loading {Path(file_path).name}: {e}")
            return None

    def extract_windows(self, data):
        """Extract sliding windows from recording."""
        if len(data) < self.window_size:
            return []

        windows = []
        for i in range(0, len(data) - self.window_size + 1, self.stride):
            window = data[i:i + self.window_size]
            # Z-score normalization
            window_mean = window.mean(axis=0, keepdims=True)
            window_std = window.std(axis=0, keepdims=True) + 1e-8
            window_normalized = (window - window_mean) / window_std
            windows.append(window_normalized)

        return windows

    def build_dataset(self):
        """Build multi-task dataset with exercise and form quality labels."""
        all_windows = []
        all_exercise_labels = []
        all_form_labels = []

        exercise_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        if not exercise_dirs:
            print(f"Error: No exercise directories found in {self.data_dir}")
            return None

        # Build exercise label mappings
        exercises = sorted([d.name for d in exercise_dirs])
        self.exercise_label2id = {label: idx for idx, label in enumerate(exercises)}
        self.exercise_id2label = {idx: label for label, idx in self.exercise_label2id.items()}

        print(f"Found {len(exercises)} exercise types: {exercises}")
        print(f"Form quality classes: good, bad")

        form_counts = {'good': 0, 'bad': 0}

        for exercise_dir in exercise_dirs:
            exercise_label = exercise_dir.name
            print(f"\nProcessing {exercise_label}...")

            rsf_files = list(exercise_dir.glob("*.rsf"))
            print(f"  Found {len(rsf_files)} RSF files")

            for rsf_file in rsf_files:
                form_quality = self.extract_form_quality(rsf_file.name)

                data = self.load_rsf_recording(rsf_file)
                if data is None:
                    continue

                windows = self.extract_windows(data)
                if not windows:
                    continue

                all_windows.extend(windows)
                all_exercise_labels.extend([exercise_label] * len(windows))
                all_form_labels.extend([form_quality] * len(windows))
                form_counts[form_quality] += len(windows)

                print(f"    {rsf_file.name}: {len(windows)} windows ({form_quality})")

        if not all_windows:
            print("\nError: No data loaded!")
            return None

        X = np.array(all_windows)
        y_exercise = np.array([self.exercise_label2id[label] for label in all_exercise_labels])
        y_form = np.array([self.form_label2id[label] for label in all_form_labels])

        print(f"\n=== Dataset Summary ===")
        print(f"Total windows: {len(X)}")
        print(f"Shape: {X.shape}")
        print(f"\nExercise distribution:")
        for label, idx in self.exercise_label2id.items():
            count = np.sum(y_exercise == idx)
            print(f"  {label}: {count} ({100*count/len(y_exercise):.1f}%)")
        print(f"\nForm quality distribution:")
        print(f"  good: {form_counts['good']} ({100*form_counts['good']/len(X):.1f}%)")
        print(f"  bad: {form_counts['bad']} ({100*form_counts['bad']/len(X):.1f}%)")

        # Split data
        X_train, X_temp, y_ex_train, y_ex_temp, y_form_train, y_form_temp = train_test_split(
            X, y_exercise, y_form, test_size=0.3, random_state=42, stratify=y_form
        )
        X_val, X_test, y_ex_val, y_ex_test, y_form_val, y_form_test = train_test_split(
            X_temp, y_ex_temp, y_form_temp, test_size=0.5, random_state=42, stratify=y_form_temp
        )

        print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_exercise_train': y_ex_train, 'y_exercise_val': y_ex_val, 'y_exercise_test': y_ex_test,
            'y_form_train': y_form_train, 'y_form_val': y_form_val, 'y_form_test': y_form_test,
            'exercise_label2id': self.exercise_label2id,
            'exercise_id2label': self.exercise_id2label,
            'form_label2id': self.form_label2id,
            'form_id2label': self.form_id2label,
            'features': self.features,
            'window_size': self.window_size
        }

    def save_dataset(self, dataset, output_dir="data/processed_form"):
        """Save multi-task dataset."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save arrays
        np.save(output_path / "X_train.npy", dataset['X_train'])
        np.save(output_path / "X_val.npy", dataset['X_val'])
        np.save(output_path / "X_test.npy", dataset['X_test'])
        np.save(output_path / "y_exercise_train.npy", dataset['y_exercise_train'])
        np.save(output_path / "y_exercise_val.npy", dataset['y_exercise_val'])
        np.save(output_path / "y_exercise_test.npy", dataset['y_exercise_test'])
        np.save(output_path / "y_form_train.npy", dataset['y_form_train'])
        np.save(output_path / "y_form_val.npy", dataset['y_form_val'])
        np.save(output_path / "y_form_test.npy", dataset['y_form_test'])

        # Save metadata
        metadata = {
            'exercise_label2id': dataset['exercise_label2id'],
            'exercise_id2label': dataset['exercise_id2label'],
            'form_label2id': dataset['form_label2id'],
            'form_id2label': dataset['form_id2label'],
            'features': dataset['features'],
            'window_size': dataset['window_size'],
            'num_exercises': len(dataset['exercise_label2id']),
            'num_form_classes': 2,
            'input_size': len(dataset['features'])
        }

        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nDataset saved to {output_path}")


if __name__ == "__main__":
    print("=== Form Quality Dataset Builder ===\n")

    builder = FormQualityDatasetBuilder(
        data_dir="data/recordings",
        window_size=150,
        stride=50
    )

    dataset = builder.build_dataset()

    if dataset:
        builder.save_dataset(dataset)
        print("\n=== Form quality dataset ready! ===")
