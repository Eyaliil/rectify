#!/usr/bin/env python3
"""Debug RSF parsing"""

import flexlib as fl
from pathlib import Path

rsf_file = Path("data/recordings/dl/Jonas_DL_Better_rsf_v1_2025-11-19_10-58-48.rsf")

print(f"Parsing: {rsf_file.name}\n")

reader = fl.RSFV1Reader()
result = reader.parse(str(rsf_file))

print(f"Result type: {type(result)}")
print(f"Result attributes: {[a for a in dir(result) if not a.startswith('_')]}")

# Check if it's iterable
if hasattr(result, '__iter__'):
    print(f"\nResult is iterable")
    try:
        for i, item in enumerate(result):
            print(f"\nItem {i}:")
            print(f"  Type: {type(item)}")
            print(f"  Attributes: {[a for a in dir(item) if not a.startswith('_')]}")

            if hasattr(item, 'angles'):
                angles = item.angles
                print(f"  ✓ Has angles!")
                print(f"    bend={getattr(angles, 'bend', 'N/A')}")
                print(f"    sagittal={getattr(angles, 'sagittal', 'N/A')}")
                print(f"    lateral={getattr(angles, 'lateral', 'N/A')}")
                print(f"    twist={getattr(angles, 'twist', 'N/A')}")

            if hasattr(item, 'acc'):
                print(f"  ✓ Has acc!")

            if i >= 2:
                break
    except Exception as e:
        print(f"Error iterating: {e}")
        import traceback
        traceback.print_exc()

# Check if it's an AnnotatedRecording
if hasattr(result, 'measurements'):
    print(f"\n✓ Has measurements attribute")
    measurements = result.measurements
    print(f"  Type: {type(measurements)}")
    print(f"  Length: {len(measurements) if hasattr(measurements, '__len__') else 'unknown'}")

    if hasattr(measurements, '__iter__'):
        for i, m in enumerate(measurements):
            print(f"\n  Measurement {i}:")
            print(f"    Type: {type(m)}")
            if hasattr(m, 'angles'):
                print(f"    ✓ Has angles!")
                angles = m.angles
                print(f"      bend={getattr(angles, 'bend', 'N/A')}")
            if i >= 2:
                break
