#!/usr/bin/env python3
"""
Debug script to check what attributes are available in live sensor measurements
vs what's in RSF files
"""

import flexlib as fl
from pathlib import Path

print("=== Checking RSF File Attributes ===\n")

# Load a sample RSF file
rsf_file = Path("data/recordings/dl/Jonas_DL_Better_rsf_v1_2025-11-19_10-58-48.rsf")
reader = fl.RSFV1Reader()
recording = reader.parse(str(rsf_file))

measurement = recording.measurements[0]

print(f"Measurement type: {type(measurement)}")
print(f"Measurement attributes: {[a for a in dir(measurement) if not a.startswith('_')]}")

print(f"\nAngles type: {type(measurement.angles)}")
print(f"Angles attributes: {[a for a in dir(measurement.angles) if not a.startswith('_')]}")

print(f"\nAngles values:")
angles = measurement.angles
for attr in ['bend', 'sagittal', 'lateral', 'twist']:
    if hasattr(angles, attr):
        print(f"  {attr}: {getattr(angles, attr)}")
    else:
        print(f"  {attr}: NOT FOUND")

print(f"\nAcceleration type: {type(measurement.acc)}")
print(f"Acceleration attributes: {[a for a in dir(measurement.acc) if not a.startswith('_')]}")

if hasattr(measurement.acc, 'norm'):
    print(f"  acc.norm(): {measurement.acc.norm()}")
if hasattr(measurement.acc, 'x'):
    print(f"  acc.x: {measurement.acc.x}")
    print(f"  acc.y: {measurement.acc.y}")
    print(f"  acc.z: {measurement.acc.z}")

print("\n" + "="*50)
print("IMPORTANT: Check if live sensor has same attributes!")
print("Live sensor might use different names.")
print("="*50)
