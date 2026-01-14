import numpy as np
import glob

# Check files from results_koi
koi_files = glob.glob('notebooks/results_koi/*.npz')[:5]
print('=== results_koi files ===')
for f in koi_files:
    try:
        d = np.load(f)
        print(f'File: {f}')
        print(f'Keys: {list(d.keys())}')
        if 'global_view' in d:
            gv = d['global_view']
            print(f'global_view shape: {gv.shape}')
        if 'local_view' in d:
            lv = d['local_view']
            print(f'local_view shape: {lv.shape}')
        print('---')
    except Exception as e:
        print(f'Error: {e}')

print()

# Check files from results_confirmed
conf_files = glob.glob('notebooks/results_confirmed/*.npz')[:5]
print('=== results_confirmed files ===')
for f in conf_files:
    try:
        d = np.load(f)
        print(f'File: {f}')
        print(f'Keys: {list(d.keys())}')
        if 'global_view' in d:
            gv = d['global_view']
            print(f'global_view shape: {gv.shape}')
        if 'local_view' in d:
            lv = d['local_view']
            print(f'local_view shape: {lv.shape}')
        print('---')
    except Exception as e:
        print(f'Error: {e}')
