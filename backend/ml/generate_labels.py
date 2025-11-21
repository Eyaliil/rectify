"""
Generate initial labels.json from existing recordings.

Run this script to create a template labels.json file,
then manually update the rep counts for each recording.
"""

import json
import re
from pathlib import Path


def extract_info_from_filename(filename):
    """Extract metadata from filename pattern: Person_Exercise_Form_..."""
    # Pattern: Jonas_DL_Better_rsf_v1_2025-11-19_10-58-48.rsf
    parts = filename.replace('.rsf', '').split('_')

    info = {
        'reps': 10,  # Default - UPDATE THIS MANUALLY
        'form': 'good'
    }

    # Try to extract form quality
    filename_lower = filename.lower()
    if 'worse' in filename_lower or 'bad' in filename_lower:
        info['form'] = 'bad'
    elif 'better' in filename_lower or 'correct' in filename_lower or 'good' in filename_lower:
        info['form'] = 'good'

    # Try to extract person name (usually first part)
    if parts:
        info['person'] = parts[0]

    # Try to extract rep count from filename if present (e.g., "5reps")
    rep_match = re.search(r'(\d+)reps?', filename_lower)
    if rep_match:
        info['reps'] = int(rep_match.group(1))

    return info


def generate_labels(data_dir="data/recordings", output_file="data/recordings/labels.json"):
    """Generate labels.json from existing recordings."""
    data_path = Path(data_dir)

    labels = {}

    # Get all exercise directories
    exercise_dirs = [d for d in data_path.iterdir() if d.is_dir()]

    for exercise_dir in sorted(exercise_dirs):
        exercise = exercise_dir.name
        labels[exercise] = {}

        # Get all RSF files
        rsf_files = list(exercise_dir.glob("*.rsf"))

        for rsf_file in sorted(rsf_files):
            filename = rsf_file.name
            info = extract_info_from_filename(filename)
            labels[exercise][filename] = info

    # Save to JSON
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"Generated {output_path}")
    print(f"\nFound {sum(len(v) for v in labels.values())} recordings across {len(labels)} exercises")
    print("\n⚠️  IMPORTANT: Edit labels.json to set correct rep counts!")
    print("   Default is 10 reps - update each recording's 'reps' field")

    return labels


if __name__ == "__main__":
    labels = generate_labels()

    # Print summary
    print("\n=== Summary ===")
    for exercise, files in labels.items():
        print(f"\n{exercise}: {len(files)} recordings")
        for filename, info in list(files.items())[:2]:  # Show first 2
            print(f"  - {filename[:40]}... reps={info['reps']}")
        if len(files) > 2:
            print(f"  ... and {len(files)-2} more")
