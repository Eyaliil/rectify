#!/usr/bin/env python3
"""Test how to read RSF files with flexlib"""

import flexlib as fl
from pathlib import Path

# Get a sample RSF file
rsf_file = Path("data/recordings/dl/Jonas_DL_Better_rsf_v1_2025-11-19_10-58-48.rsf")

if not rsf_file.exists():
    print(f"File not found: {rsf_file}")
    exit(1)

print(f"Testing with: {rsf_file.name}\n")

# Try different approaches
print("1. Testing FlexReader...")
try:
    reader = fl.FlexReader(str(rsf_file))
    count = 0
    for measurement in reader:
        if count == 0:
            print(f"  ✓ FlexReader works!")
            print(f"  Measurement type: {type(measurement)}")
            print(f"  Has angles: {hasattr(measurement, 'angles')}")
            if hasattr(measurement, 'angles'):
                angles = measurement.angles
                print(f"  Angles type: {type(angles)}")
                print(f"  Angles attributes: {[a for a in dir(angles) if not a.startswith('_')]}")
        count += 1
        if count > 2:
            break
    print(f"  Total measurements: {count}")
except Exception as e:
    print(f"  ✗ FlexReader failed: {e}")

print("\n2. Testing RSFV1Reader...")
try:
    # Maybe it needs to be used with 'with' statement or opened differently
    with open(str(rsf_file), 'rb') as f:
        reader = fl.RSFV1Reader()
        # Try to see what methods it has
        print(f"  RSFV1Reader methods: {[m for m in dir(reader) if not m.startswith('_')]}")
except Exception as e:
    print(f"  Error: {e}")

print("\n3. Checking flexlib documentation...")
print(f"  FlexReader.__init__ signature:")
import inspect
try:
    sig = inspect.signature(fl.FlexReader.__init__)
    print(f"    {sig}")
except:
    pass

print(f"\n  RSFV1Reader.__init__ signature:")
try:
    sig = inspect.signature(fl.RSFV1Reader.__init__)
    print(f"    {sig}")
except:
    pass
