#!/usr/bin/env python3
"""Verify minitorch installation"""
import sys

try:
    import minitorch
    print("✓ minitorch module imported successfully")
    print(f"  Location: {minitorch.__file__}")

    # Check if key components are available
    if hasattr(minitorch, 'Scalar'):
        print("✓ minitorch.Scalar is available")

    if hasattr(minitorch, 'Tensor'):
        print("✓ minitorch.Tensor is available")

    print("\n✓ Installation verified successfully!")
    sys.exit(0)

except ImportError as e:
    print(f"✗ Failed to import minitorch: {e}")
    print("\nPlease run: pip install -e .")
    sys.exit(1)

