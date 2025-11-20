#!/usr/bin/env python3
"""Test RSF parsing with flexlib"""

import flexlib as fl
from pathlib import Path

# Get a sample RSF file
rsf_file = Path("data/recordings/dl/Jonas_DL_Better_rsf_v1_2025-11-19_10-58-48.rsf")

print(f"Testing with: {rsf_file.name}\n")

# Try using parse method
print("Testing RSFV1Reader.parse()...")
try:
    with open(str(rsf_file), 'rb') as f:
        data = f.read()

    reader = fl.RSFV1Reader()
    result = reader.parse(data)

    print(f"  ✓ Parse successful!")
    print(f"  Result type: {type(result)}")
    print(f"  Result attributes: {[a for a in dir(result) if not a.startswith('_')]}")

    # Try to iterate
    if hasattr(result, '__iter__'):
        print(f"  Result is iterable")
        count = 0
        for measurement in result:
            if count == 0:
                print(f"\n  First measurement:")
                print(f"    Type: {type(measurement)}")
                print(f"    Attributes: {[a for a in dir(measurement) if not a.startswith('_')]}")
                if hasattr(measurement, 'angles'):
                    angles = measurement.angles
                    print(f"    Angles: bend={angles.bend}, sagittal={angles.sagittal}, lateral={angles.lateral}, twist={angles.twist}")
                if hasattr(measurement, 'acc'):
                    print(f"    Has acceleration data")
            count += 1
            if count >= 3:
                break
        print(f"\n  Total measurements tested: {count}")

except Exception as e:
    import traceback
    print(f"  ✗ Failed: {e}")
    traceback.print_exc()

# Try RSFV2Reader as well
print("\n\nTesting RSFV2Reader.parse()...")
try:
    with open(str(rsf_file), 'rb') as f:
        data = f.read()

    reader = fl.RSFV2Reader()
    result = reader.parse(data)
    print(f"  ✓ V2 Parse successful!")
    print(f"  (File is actually V2 format)")
except Exception as e:
    print(f"  ✗ V2 Failed (expected for V1 files): {e}")
