#!/usr/bin/env python3
"""Test what's available in flexlib"""

import sys

try:
    import flexlib as fl
    print("✓ flexlib imported successfully")
    print(f"Version: {fl.__version__ if hasattr(fl, '__version__') else 'unknown'}")
    print(f"\nAvailable attributes:")
    for attr in sorted(dir(fl)):
        if not attr.startswith('_'):
            print(f"  - {attr}")

    # Check for RSF
    if hasattr(fl, 'RSF'):
        print("\n✓ RSF class is available")
    else:
        print("\n✗ RSF class NOT found")
        print("  Looking for alternatives...")

        # Check for common alternatives
        for name in ['RawSensorFile', 'read_rsf', 'load_rsf', 'Recording', 'File']:
            if hasattr(fl, name):
                print(f"  ✓ Found: {name}")

except ImportError as e:
    print(f"✗ Failed to import flexlib: {e}")
    sys.exit(1)
