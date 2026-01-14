import numpy as np
import os
import glob

# Search for the first npz file in results_confirmed
files = glob.glob(r"d:\Exoplanet_AI\notebooks\results_confirmed\*.npz")
if files:
    fpath = files[0]
    try:
        data = np.load(fpath)
        print(f"File: {fpath}")
        print(f"Keys: {list(data.keys())}")
        if 'label' in data:
            print(f"Label: {data['label']}")
        else:
            print("Label key NOT found.")
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print("No .npz files found in d:\\Exoplanet_AI\\notebooks\\results_confirmed")
