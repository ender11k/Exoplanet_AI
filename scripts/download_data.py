import os
import pandas as pd
import numpy as np
import lightkurve as lk
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# ==========================================
# Configuration
# ==========================================
DATA_FILE = 'data/all_df.csv'
LOG_DIR = 'logs'
LOG_FILE = os.path.join(LOG_DIR, 'download_errors.txt')
MAX_WORKERS = 20
N_SAMPLES = None  # Set to an integer (e.g., 100) for testing, or None for all

# ==========================================
# Setup Logging
# ==========================================
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE, 
    level=logging.ERROR, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_star(target_id, catalog):
    """
    Downloads light curve data for a single star using Lightkurve.
    Relies on Lightkurve's built-in caching to avoid re-downloading.
    """
    try:
        search_id = ""
        search = None
        
        if catalog == 'KOI':
            # Kepler Object of Interest: Use KIC ID
            search_id = f"KIC {int(target_id)}"
            # Search for Kepler Mission data, Long Cadence
            search = lk.search_lightcurve(search_id, author='Kepler', cadence='long')
            
        elif catalog == 'TOI':
            # TESS Object of Interest: Use TIC ID
            search_id = f"TIC {int(target_id)}"
            # Search for SPOC data (Science Processing Operations Center)
            search = lk.search_lightcurve(search_id, author='SPOC')
            
        else:
            # Should be filtered out before calling this
            return

        if search and len(search) > 0:
            # download_all() downloads all found products (quarters/sectors)
            # Lightkurve handles caching automatically
            search.download_all()
        else:
            # Log as warning if no data found, but don't raise exception
            logging.warning(f"No light curves found for {search_id}")

    except Exception as e:
        # Log the error and re-raise to be caught by the executor if needed
        # or just log it here.
        error_msg = f"Failed to download {target_id} ({catalog}): {str(e)}"
        logging.error(error_msg)
        # We don't raise here to keep the thread alive, but the task is "failed" in a sense.

def main():
    print("Loading catalog...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE, low_memory=False)
    
    # ==========================================
    # Data Selector Logic
    # ==========================================
    print("Applying Data Selector Logic...")
    
    # 1. Clean Filtering: Keep only KOI and TOI (Drop TCE)
    df = df[df['catalog'].isin(['KOI', 'TOI'])].copy()
    print(f"Filtered to {len(df)} KOI/TOI candidates.")

    # 2. Unified Duration Column
    # Create 'final_duration' based on catalog type
    df['final_duration'] = np.nan
    
    # KOI: Use 'koi_duration'
    mask_koi = df['catalog'] == 'KOI'
    if 'koi_duration' in df.columns:
        df.loc[mask_koi, 'final_duration'] = df.loc[mask_koi, 'koi_duration']
    
    # TOI: Use 'pl_trandurh'
    mask_toi = df['catalog'] == 'TOI'
    if 'pl_trandurh' in df.columns:
        df.loc[mask_toi, 'final_duration'] = df.loc[mask_toi, 'pl_trandurh']
        
    # Drop rows where duration is still NaN
    initial_len = len(df)
    df = df.dropna(subset=['final_duration'])
    print(f"Dropped {initial_len - len(df)} rows with missing duration. Remaining: {len(df)}")

    # 3. Drop duplicates based on target_id AND period
    # This ensures we keep unique planet candidates (multi-planet systems), 
    # not just unique stars.
    # Note: Filename logic downstream (in processing) should use KIC_{ID}_P{Period}.npz
    if 'period' in df.columns:
        df = df.drop_duplicates(subset=['target_id', 'period'])
        print(f"Dropped duplicates (Target + Period). Final Candidates: {len(df)}")
    else:
        print("[Warning] 'period' column not found. Skipping duplicate drop by period.")

    # ==========================================
    # Download Execution
    # ==========================================
    # For downloading, we only need unique stars.
    # (Multiple planets on the same star use the same light curve file)
    targets = df[['target_id', 'catalog']].drop_duplicates()
    
    # Optional: Limit samples
    if N_SAMPLES is not None:
        targets = targets.head(N_SAMPLES)
        print(f"Limiting to first {N_SAMPLES} stars.")
    
    total_targets = len(targets)
    print(f"Found {total_targets} unique stars to download.")
    print(f"Starting download with {MAX_WORKERS} worker threads...")
    
    # Use ThreadPoolExecutor for I/O bound tasks
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_target = {
            executor.submit(download_star, row['target_id'], row['catalog']): row['target_id']
            for _, row in targets.iterrows()
        }
        
        # Process as they complete with a progress bar
        for future in tqdm(as_completed(future_to_target), total=total_targets, desc="Downloading"):
            target_id = future_to_target[future]
            try:
                future.result()
            except Exception as exc:
                logging.error(f"Unhandled exception for {target_id}: {exc}")

    print("\nDownload process completed.")
    print(f"Check {LOG_FILE} for any errors.")

if __name__ == "__main__":
    main()
