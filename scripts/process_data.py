import os
import glob
import numpy as np
import pandas as pd
import lightkurve as lk
import multiprocessing
from tqdm import tqdm
import warnings
import logging

# Suppress warnings from lightkurve/astropy
warnings.filterwarnings('ignore')

# ==========================================
# Configuration
# ==========================================
CACHE_DIR = r"D:\.lightkurve\cache\mastDownload\Kepler"
METADATA_PATH = "data/all_df.csv"
OUTPUT_DIR = "notebooks/results_koi"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "processing_errors.txt")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(filename=LOG_FILE, level=logging.ERROR, 
                    format='%(asctime)s - %(message)s')

# ==========================================
# Worker Function
# ==========================================
def process_star(args):
    kic_id, file_path, meta, output_dir = args
    
    try:
        # 1. Read & Clean
        # lightkurve.read returns a LightCurveCollection or LightCurve
        # We use lk.read() which handles FITS files automatically
        lc_data = lk.read(file_path)
        
        # If it's a collection (multiple quarters), stitch them
        if isinstance(lc_data, lk.LightCurveCollection):
            lc = lc_data.stitch()
        else:
            lc = lc_data
            
        # Remove NaNs first
        lc = lc.remove_nans()
        
        # Flatten (remove stellar variability trends)
        # window_length=101 as per requirements
        lc = lc.flatten(window_length=101)
        
        # Remove outliers (sigma=3)
        lc = lc.remove_outliers(sigma=3)
        
        # Metadata extraction
        period = meta.get('period', np.nan)
        t0 = meta.get('epoch', np.nan)
        duration_hours = meta.get('koi_duration', np.nan)
        
        # Validation
        if np.isnan(period) or np.isnan(t0) or np.isnan(duration_hours):
            raise ValueError(f"Missing metadata (Period: {period}, Epoch: {t0}, Duration: {duration_hours})")

        # 2. Phase Fold
        folded_lc = lc.fold(period=period, epoch_time=t0)
        
        # 3. View 1: Global
        # Bin to exactly 2001 bins
        global_binned = folded_lc.bin(bins=2001)
        
        # Normalize (Center at 0.0)
        # Flattened flux is normalized to 1.0, so we subtract 1.0
        global_flux = global_binned.flux.value - 1.0
        
        # Handle NaNs (fill with 0.0)
        global_flux = np.nan_to_num(global_flux, nan=0.0)
        
        # Ensure shape is exactly 2001 (Interpolate if binning was slightly off)
        if len(global_flux) != 2001:
             x_old = np.linspace(0, 1, len(global_flux))
             x_new = np.linspace(0, 1, 2001)
             global_flux = np.interp(x_new, x_old, global_flux)

        global_view = global_flux.reshape(-1, 1)

        # 4. View 2: Local
        duration_days = duration_hours / 24.0
        duration_phase = duration_days / period
        phase_limit = 2 * duration_phase
        
        # Crop to [-2*dur, +2*dur]
        mask = (folded_lc.phase.value >= -phase_limit) & (folded_lc.phase.value <= phase_limit)
        local_lc = folded_lc[mask]
        
        if len(local_lc) == 0:
             local_view = np.zeros((201, 1))
        else:
            # Bin to exactly 201 bins
            local_binned = local_lc.bin(bins=201)
            local_flux = local_binned.flux.value - 1.0
            local_flux = np.nan_to_num(local_flux, nan=0.0)
            
            if len(local_flux) != 201:
                 x_old = np.linspace(0, 1, len(local_flux))
                 x_new = np.linspace(0, 1, 201)
                 local_flux = np.interp(x_new, x_old, local_flux)
            
            local_view = local_flux.reshape(-1, 1)

        # 5. Scalars
        # [Period, Duration, Depth, Prad, Srad, Teff, logg]
        scalars = np.array([
            meta.get('period', 0),
            meta.get('koi_duration', 0),
            meta.get('koi_depth', 0),
            meta.get('koi_prad', 0),
            meta.get('koi_srad', 0),
            meta.get('koi_steff', 0),
            meta.get('koi_slogg', 0)
        ])
        # Replace NaNs with 0
        scalars = np.nan_to_num(scalars, nan=0.0)

        # 6. Label
        label = 1 if meta.get('koi_disposition') == 'CONFIRMED' else 0

        # 7. Save
        # Use Period in filename to allow multiple planets per star
        output_filename = os.path.join(output_dir, f"KIC_{kic_id}_P{period:.2f}.npz")
        np.savez_compressed(
            output_filename,
            global_view=global_view,
            local_view=local_view,
            scalars=scalars,
            label=label
        )
        
        return True

    except Exception as e:
        logging.error(f"KIC {kic_id}: {str(e)}")
        return False

# ==========================================
# Main Execution
# ==========================================
def main():
    print("[INFO] Loading metadata...")
    if not os.path.exists(METADATA_PATH):
        print(f"[Error] Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH, low_memory=False)
    
    # Ensure target_id is numeric and clean
    df['target_id'] = pd.to_numeric(df['target_id'], errors='coerce')
    df = df.dropna(subset=['target_id'])
    df['target_id'] = df['target_id'].astype(int)
    
    # --- FIX: HANDLE MULTI-PLANET SYSTEMS ---
    # Drop duplicates based on target_id AND period to keep all candidates
    if 'period' in df.columns:
        df = df.drop_duplicates(subset=['target_id', 'period'], keep='first')
    else:
        # Fallback if period is missing (unlikely given previous steps)
        df = df.drop_duplicates(subset=['target_id'], keep='first')
    
    # Create metadata lookup dictionary
    # Structure: { target_id: [ {candidate1_meta}, {candidate2_meta}, ... ] }
    cols_to_keep = ['target_id', 'period', 'epoch', 'koi_duration', 'koi_depth', 
                    'koi_prad', 'koi_srad', 'koi_steff', 'koi_slogg', 'koi_disposition']
    
    available_cols = [c for c in cols_to_keep if c in df.columns]
    
    meta_dict = {}
    for _, row in df[available_cols].iterrows():
        tid = int(row['target_id'])
        if tid not in meta_dict:
            meta_dict[tid] = []
        meta_dict[tid].append(row.to_dict())
    
    print(f"[INFO] Scanning {CACHE_DIR} for FITS files...")
    # Recursive scan for .fits files
    fits_files = glob.glob(os.path.join(CACHE_DIR, "**", "*.fits"), recursive=True)
    
    # Map KIC ID to file path
    # Filename format expected: kplr010797460-2016..._llc.fits -> ID: 10797460
    file_map = {}
    for f in fits_files:
        basename = os.path.basename(f)
        # Filter for light curve files (usually contain 'llc' or 'slc')
        if basename.startswith('kplr') and ('llc' in basename or 'slc' in basename):
            try:
                # Extract ID: kplr + 9 digits
                id_str = basename[4:13]
                kic_id = int(id_str)
                # Prefer 'llc' (Long Cadence) over 'slc' (Short Cadence) if duplicates exist
                if kic_id not in file_map or 'llc' in basename:
                    file_map[kic_id] = f
            except:
                continue
    
    print(f"[INFO] Found {len(file_map)} unique KIC files in cache.")
    
    # Prepare tasks
    tasks = []
    for kic_id, file_path in file_map.items():
        if kic_id in meta_dict:
            # Create a task for EACH candidate associated with this star
            candidates = meta_dict[kic_id]
            for candidate_meta in candidates:
                tasks.append((kic_id, file_path, candidate_meta, OUTPUT_DIR))
    
    print(f"[INFO] Matched {len(tasks)} candidate planets. Starting processing...")
    
    # Run Multiprocessing
    # Leave 2 cores free for system stability
    num_processes = max(1, multiprocessing.cpu_count() - 2)
    print(f"[INFO] Using {num_processes} worker processes.")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use imap to get an iterator for tqdm
        results = list(tqdm(pool.imap(process_star, tasks), total=len(tasks), unit="star"))
    
    success_count = sum(results)
    print(f"[INFO] Processing complete.")
    print(f"   Success: {success_count}")
    print(f"   Failed:  {len(tasks) - success_count}")
    print(f"   Errors logged to: {LOG_FILE}")

if __name__ == '__main__':
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    main()
