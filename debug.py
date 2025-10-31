import torch
import torch.nn as nn
import os
import sys

# --- This is the key ---
# We MUST import your models.py file.
# This assumes debug_dimensions.py is in the same parent folder as models.py
# If models.py is in a subfolder, adjust this path.
try:
    from models import CycleTransMorph, ScalingAndSquaring, SpatialTransformer
except ImportError:
    print("ERROR: Could not find models.py.")
    print("Make sure this script is in the same directory as models.py")
    sys.exit()

print("--- Dimension Debugger START ---")

# We will use a NON-SQUARE size: (D, H, W)
# We make H and W different to expose the bug.
DEBUG_SIZE = (128, 128, 96) 
D, H, W = DEBUG_SIZE

print(f"Target (D, H, W) size: {DEBUG_SIZE}")

# ---
# 1. Modify the models.py file (see Step 2)
# ---
print("\nPlease modify models.py as per the instructions, then re-run.")
print("Waiting for user to modify models.py...")
print("-------------------------------------------------\n")

# This is a placeholder to run *after* you modify the file.
# We will temporarily modify the models.py file to add prints.
# (See Step 2)