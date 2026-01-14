#!/usr/bin/env python3
"""
=============================================================================
EXOPLANET DATA PROCESSOR V2 - NASA ExoMiner Complete Feature Set
=============================================================================

This processor generates ALL input views used by NASA ExoMiner:
1. Global View      - Full orbital phase (2001 bins)
2. Local View       - Transit-centered (201 bins)
3. Secondary View   - Phase-shifted by 0.5 to detect eclipsing binaries
4. Odd Transit View - Only odd-numbered transits (depth consistency check)
5. Even Transit View- Only even-numbered transits
6. Centroid View    - MOM_CENTR1/2 to detect background eclipsing binaries
7. Stellar Scalars  - Extended stellar parameters

Key Improvements over V1:
- 6 additional diagnostic views for false positive rejection
- Centroid extraction from raw FITS files
- Extended scalar features (15 vs 7)
- Better error handling and validation
- Multiprocessing for speed (CPU-bound operations)

Note: GPU is NOT used because:
- FITS file I/O is disk-bound, not compute-bound
- Phase folding and binning are simple operations
- CPU multiprocessing is more effective for this workload

Author: Exoplanet AI Research Team
Date: January 2026
=============================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import lightkurve as lk
from lightkurve import LightCurve
import multiprocessing
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
import logging
from astropy.io import fits
from scipy import interpolate
import traceback

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
# Multiple cache directories to scan for FITS files
CACHE_DIRS = [
    r"D:\Exoplanet_AI\notebooks\mastDownload\Kepler",
    r"C:\Users\amard\.lightkurve\cache\mastDownload\Kepler",
    r"D:\.lightkurve\cache\mastDownload\Kepler",
]

METADATA_PATH = r"D:\Exoplanet_AI\data\all_df.csv"

# NEW OUTPUT DIRECTORY - keeps old data intact
OUTPUT_DIR = r"D:\Exoplanet_AI\notebooks\results_koi_v2"

LOG_DIR = r"D:\Exoplanet_AI\logs"
LOG_FILE = os.path.join(LOG_DIR, "processing_v2_errors.txt")
SUCCESS_LOG = os.path.join(LOG_DIR, "processing_v2_success.txt")

# View dimensions (matching NASA ExoMiner paper)
GLOBAL_BINS = 2001   # Full orbit
LOCAL_BINS = 201     # Transit-centered
SECONDARY_BINS = 201 # Secondary eclipse
ODD_EVEN_BINS = 201  # Odd/Even transit views
CENTROID_BINS = 201  # Centroid time series

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=LOG_FILE, 
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def get_valid_value(meta, candidates, default=np.nan):
    """Get first non-NaN value from candidate column names."""
    for col in candidates:
        val = meta.get(col, np.nan)
        if val is not None:
            try:
                if not np.isnan(float(val)):
                    return float(val)
            except (TypeError, ValueError):
                if isinstance(val, str):
                    return val
    return default


def safe_interpolate(flux, target_length):
    """Safely interpolate flux to target length."""
    if len(flux) == 0:
        return np.zeros(target_length)
    if len(flux) == target_length:
        return flux
    
    x_old = np.linspace(0, 1, len(flux))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, flux)


def normalize_flux(flux):
    """Normalize flux: center at 0, handle NaNs."""
    flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
    # Subtract 1 because flattened flux is normalized to 1.0
    flux = flux - 1.0
    # Clip extreme values
    flux = np.clip(flux, -0.5, 0.5)
    return flux


def extract_centroid_from_fits(fits_path):
    """
    Extract centroid time series from Kepler FITS file.
    
    Kepler FITS files contain:
    - MOM_CENTR1: Moment-derived column centroid
    - MOM_CENTR2: Moment-derived row centroid
    
    Returns: time, centr1, centr2 arrays
    """
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            time = data['TIME']
            centr1 = data['MOM_CENTR1']
            centr2 = data['MOM_CENTR2']
            
            # Remove NaNs
            mask = ~np.isnan(time) & ~np.isnan(centr1) & ~np.isnan(centr2)
            return time[mask], centr1[mask], centr2[mask]
    except Exception as e:
        return None, None, None


def phase_fold_array(time, values, period, t0):
    """Phase fold any time series array."""
    phase = ((time - t0) / period) % 1.0
    # Center at 0 (-0.5 to 0.5)
    phase[phase > 0.5] -= 1.0
    
    # Sort by phase
    sort_idx = np.argsort(phase)
    return phase[sort_idx], values[sort_idx]


def bin_phase_folded(phase, values, n_bins, phase_min=-0.5, phase_max=0.5):
    """Bin phase-folded data into fixed number of bins."""
    bins = np.linspace(phase_min, phase_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    binned_values = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])
        if np.sum(mask) > 0:
            binned_values[i] = np.nanmedian(values[mask])
        else:
            binned_values[i] = 0.0
    
    return binned_values


def get_transit_indices(time, period, t0, duration_days):
    """
    Identify individual transit events and return their indices.
    
    Returns list of (start_idx, end_idx) for each transit.
    """
    # Calculate phase
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    
    # Transit window in phase
    transit_half_width = (duration_days / period) * 2  # 2x duration for safety
    
    # Find transit number for each point
    transit_num = np.floor((time - t0) / period).astype(int)
    unique_transits = np.unique(transit_num)
    
    transits = []
    for tn in unique_transits:
        mask = (transit_num == tn) & (np.abs(phase) < transit_half_width)
        if np.sum(mask) > 5:  # Need at least 5 points
            indices = np.where(mask)[0]
            transits.append((indices.min(), indices.max()))
    
    return transits, transit_num


# =============================================================================
# View Generation Functions
# =============================================================================

def generate_global_view(folded_lc, n_bins=GLOBAL_BINS):
    """Generate global view - full orbital phase."""
    try:
        binned = folded_lc.bin(bins=n_bins)
        flux = normalize_flux(binned.flux.value)
        flux = safe_interpolate(flux, n_bins)
        return flux.reshape(-1, 1)
    except:
        return np.zeros((n_bins, 1))


def generate_local_view(folded_lc, duration_phase, n_bins=LOCAL_BINS):
    """Generate local view - transit-centered."""
    try:
        phase_limit = 2 * duration_phase
        mask = (folded_lc.phase.value >= -phase_limit) & (folded_lc.phase.value <= phase_limit)
        local_lc = folded_lc[mask]
        
        if len(local_lc) < 10:
            return np.zeros((n_bins, 1))
        
        binned = local_lc.bin(bins=n_bins)
        flux = normalize_flux(binned.flux.value)
        flux = safe_interpolate(flux, n_bins)
        return flux.reshape(-1, 1)
    except:
        return np.zeros((n_bins, 1))


def generate_secondary_view(folded_lc, duration_phase, n_bins=SECONDARY_BINS):
    """
    Generate secondary eclipse view - phase shifted by 0.5.
    
    This view is CRITICAL for detecting eclipsing binaries:
    - Real planets: No significant dip at phase 0.5
    - Eclipsing binaries: Show secondary eclipse at phase 0.5
    """
    try:
        # Shift phase by 0.5 to center on secondary eclipse position
        phase = folded_lc.phase.value.copy()
        shifted_phase = (phase + 0.5) % 1.0
        shifted_phase[shifted_phase > 0.5] -= 1.0
        
        # Extract region around secondary (same window as local view)
        phase_limit = 2 * duration_phase
        mask = (shifted_phase >= -phase_limit) & (shifted_phase <= phase_limit)
        
        if np.sum(mask) < 10:
            return np.zeros((n_bins, 1))
        
        # Bin the secondary region
        secondary_flux = folded_lc.flux.value[mask]
        secondary_phase = shifted_phase[mask]
        
        binned_flux = bin_phase_folded(secondary_phase, secondary_flux, n_bins, 
                                        -phase_limit, phase_limit)
        flux = normalize_flux(binned_flux)
        return flux.reshape(-1, 1)
    except:
        return np.zeros((n_bins, 1))


def generate_odd_even_views(lc, period, t0, duration_days, n_bins=ODD_EVEN_BINS):
    """
    Generate odd and even transit views.
    
    This is CRITICAL for detecting:
    - V-shaped eclipsing binaries with different eclipse depths
    - Blended binaries where odd/even depths differ
    
    Real planets should have identical odd/even depths.
    """
    try:
        time = lc.time.value
        flux = lc.flux.value
        
        # Get transit numbers
        transit_num = np.floor((time - t0) / period).astype(int)
        
        # Phase fold
        phase = ((time - t0) / period) % 1.0
        phase[phase > 0.5] -= 1.0
        
        # Transit window
        duration_phase = duration_days / period
        phase_limit = 2 * duration_phase
        
        # Separate odd and even
        in_transit = np.abs(phase) < phase_limit
        odd_mask = in_transit & (transit_num % 2 == 1)
        even_mask = in_transit & (transit_num % 2 == 0)
        
        # Generate odd view
        if np.sum(odd_mask) > 10:
            odd_flux = bin_phase_folded(phase[odd_mask], flux[odd_mask], n_bins,
                                        -phase_limit, phase_limit)
            odd_view = normalize_flux(odd_flux).reshape(-1, 1)
        else:
            odd_view = np.zeros((n_bins, 1))
        
        # Generate even view
        if np.sum(even_mask) > 10:
            even_flux = bin_phase_folded(phase[even_mask], flux[even_mask], n_bins,
                                         -phase_limit, phase_limit)
            even_view = normalize_flux(even_flux).reshape(-1, 1)
        else:
            even_view = np.zeros((n_bins, 1))
        
        return odd_view, even_view
    except:
        return np.zeros((n_bins, 1)), np.zeros((n_bins, 1))


def generate_centroid_view(fits_path, period, t0, duration_days, n_bins=CENTROID_BINS):
    """
    Generate centroid shift time series.
    
    This is CRITICAL for detecting background eclipsing binaries:
    - Real planets: No centroid shift during transit
    - Background EBs: Centroid shifts toward contaminating star
    
    We compute the centroid offset magnitude during transit.
    """
    try:
        time, centr1, centr2 = extract_centroid_from_fits(fits_path)
        
        if time is None or len(time) < 100:
            return np.zeros((n_bins, 1))
        
        # Normalize centroids (remove mean, scale by std)
        centr1 = (centr1 - np.nanmean(centr1)) / (np.nanstd(centr1) + 1e-10)
        centr2 = (centr2 - np.nanmean(centr2)) / (np.nanstd(centr2) + 1e-10)
        
        # Compute centroid offset magnitude
        centroid_offset = np.sqrt(centr1**2 + centr2**2)
        
        # Phase fold
        phase, centroid_folded = phase_fold_array(time, centroid_offset, period, t0)
        
        # Extract transit region
        duration_phase = duration_days / period
        phase_limit = 2 * duration_phase
        mask = (phase >= -phase_limit) & (phase <= phase_limit)
        
        if np.sum(mask) < 10:
            return np.zeros((n_bins, 1))
        
        # Bin
        binned = bin_phase_folded(phase[mask], centroid_folded[mask], n_bins,
                                   -phase_limit, phase_limit)
        
        # Normalize
        binned = np.nan_to_num(binned, nan=0.0)
        binned = np.clip(binned, -5, 5)  # Clip extreme values
        
        return binned.reshape(-1, 1)
    except:
        return np.zeros((n_bins, 1))


# =============================================================================
# Main Processing Function
# =============================================================================

def process_star_v2(args):
    """
    Process a single star/candidate and generate all NASA ExoMiner views.
    
    Output NPZ structure:
    - global_view:    (2001, 1) - Full orbital phase
    - local_view:     (201, 1)  - Transit-centered
    - secondary_view: (201, 1)  - Secondary eclipse (phase + 0.5)
    - odd_view:       (201, 1)  - Odd-numbered transits
    - even_view:      (201, 1)  - Even-numbered transits
    - centroid_view:  (201, 1)  - Centroid offset during transit
    - scalars:        (15,)     - Extended stellar/transit parameters
    - label:          scalar    - 0 or 1
    """
    kic_id, fits_path, meta, output_dir = args
    
    try:
        # ===================
        # 1. Load Light Curve
        # ===================
        lc_data = lk.read(fits_path)
        
        if isinstance(lc_data, lk.LightCurveCollection):
            lc = lc_data.stitch()
        else:
            lc = lc_data
        
        # Clean
        lc = lc.remove_nans()
        lc_flat = lc.flatten(window_length=101)
        lc_clean = lc_flat.remove_outliers(sigma=3)
        
        # ===================
        # 2. Extract Metadata
        # ===================
        period = get_valid_value(meta, ['period', 'koi_period', 'pl_orbper', 'tce_period'])
        t0 = get_valid_value(meta, ['epoch', 'koi_time0bk', 'pl_tranmid', 'tce_time0bk', 't0'])
        duration_hours = get_valid_value(meta, ['koi_duration', 'pl_trandurh', 'tce_duration', 'duration'])
        
        if np.isnan(period) or np.isnan(t0) or np.isnan(duration_hours):
            raise ValueError(f"Missing critical metadata: P={period}, t0={t0}, dur={duration_hours}")
        
        duration_days = duration_hours / 24.0
        duration_phase = duration_days / period
        
        # ===================
        # 3. Phase Fold
        # ===================
        folded_lc = lc_clean.fold(period=period, epoch_time=t0)
        
        # ===================
        # 4. Generate All Views
        # ===================
        
        # View 1: Global (full orbit)
        global_view = generate_global_view(folded_lc, GLOBAL_BINS)
        
        # View 2: Local (transit-centered)
        local_view = generate_local_view(folded_lc, duration_phase, LOCAL_BINS)
        
        # View 3: Secondary Eclipse (CRITICAL for EB detection)
        secondary_view = generate_secondary_view(folded_lc, duration_phase, SECONDARY_BINS)
        
        # Views 4 & 5: Odd/Even (CRITICAL for EB detection)
        odd_view, even_view = generate_odd_even_views(lc_clean, period, t0, duration_days, ODD_EVEN_BINS)
        
        # View 6: Centroid (CRITICAL for background EB detection)
        centroid_view = generate_centroid_view(fits_path, period, t0, duration_days, CENTROID_BINS)
        
        # ===================
        # 5. Extended Scalars (15 features)
        # ===================
        depth = get_valid_value(meta, ['koi_depth', 'pl_trandep', 'tce_depth'], 0.0)
        
        scalars = np.array([
            # Transit parameters
            period,                                                              # 0: Orbital period (days)
            duration_hours,                                                       # 1: Transit duration (hours)
            depth,                                                                # 2: Transit depth (ppm)
            get_valid_value(meta, ['koi_impact', 'pl_imppar', 'tce_impact'], 0.0), # 3: Impact parameter
            get_valid_value(meta, ['koi_ror', 'pl_ratror', 'tce_ror'], 0.0),       # 4: Planet/star radius ratio
            
            # Planet parameters
            get_valid_value(meta, ['koi_prad', 'pl_rade', 'tce_prad'], 0.0),       # 5: Planet radius (Earth)
            get_valid_value(meta, ['koi_sma', 'pl_orbsmax', 'tce_sma'], 0.0),      # 6: Semi-major axis (AU)
            get_valid_value(meta, ['koi_incl', 'pl_orbincl', 'tce_incl'], 90.0),   # 7: Orbital inclination (deg)
            get_valid_value(meta, ['koi_teq', 'pl_eqt', 'tce_eqt'], 0.0),          # 8: Equilibrium temp (K)
            get_valid_value(meta, ['koi_insol', 'pl_insol', 'tce_insol'], 0.0),    # 9: Insolation flux
            
            # Stellar parameters
            get_valid_value(meta, ['koi_srad', 'st_rad', 'tce_sradius'], 1.0),     # 10: Stellar radius (Solar)
            get_valid_value(meta, ['koi_steff', 'st_teff', 'tce_steff'], 5500.0),  # 11: Effective temp (K)
            get_valid_value(meta, ['koi_slogg', 'st_logg', 'tce_slogg'], 4.4),     # 12: Surface gravity (log g)
            get_valid_value(meta, ['koi_smet', 'st_met', 'tce_smet'], 0.0),        # 13: Metallicity [Fe/H]
            get_valid_value(meta, ['koi_smass', 'st_mass', 'tce_smass'], 1.0),     # 14: Stellar mass (Solar)
        ], dtype=np.float32)
        
        # Replace NaNs with reasonable defaults
        scalars = np.nan_to_num(scalars, nan=0.0)
        
        # ===================
        # 6. Label
        # ===================
        disposition = get_valid_value(meta, ['koi_disposition', 'tfopwg_disp', 'koi_pdisposition'], '')
        
        is_confirmed = False
        if isinstance(disposition, str):
            disp_upper = disposition.upper()
            if 'CONFIRMED' in disp_upper or disp_upper in ['CP', 'KP']:
                is_confirmed = True
        
        label = np.int32(1 if is_confirmed else 0)
        
        # ===================
        # 7. Save
        # ===================
        output_filename = os.path.join(output_dir, f"KIC_{kic_id}_P{period:.2f}.npz")
        
        np.savez_compressed(
            output_filename,
            # Views
            global_view=global_view.astype(np.float32),
            local_view=local_view.astype(np.float32),
            secondary_view=secondary_view.astype(np.float32),
            odd_view=odd_view.astype(np.float32),
            even_view=even_view.astype(np.float32),
            centroid_view=centroid_view.astype(np.float32),
            # Scalars and label
            scalars=scalars,
            label=label
        )
        
        return True, kic_id, period
        
    except Exception as e:
        error_msg = f"KIC {kic_id}: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        return False, kic_id, str(e)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 70)
    print("EXOPLANET DATA PROCESSOR V2 - NASA ExoMiner Complete Feature Set")
    print("=" * 70)
    print()
    print("This will generate 6 views + 15 scalars for each candidate:")
    print("  1. Global View      (2001 bins) - Full orbital phase")
    print("  2. Local View       (201 bins)  - Transit-centered")
    print("  3. Secondary View   (201 bins)  - Secondary eclipse detection")
    print("  4. Odd Transit View (201 bins)  - Odd-numbered transits")
    print("  5. Even Transit View(201 bins)  - Even-numbered transits")
    print("  6. Centroid View    (201 bins)  - Centroid shift detection")
    print()
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"(Old data in results_koi will be preserved)")
    print()
    
    # Load metadata
    print("[1/4] Loading metadata...")
    if not os.path.exists(METADATA_PATH):
        print(f"ERROR: Metadata file not found at {METADATA_PATH}")
        return
    
    df = pd.read_csv(METADATA_PATH, low_memory=False)
    df['target_id'] = pd.to_numeric(df['target_id'], errors='coerce')
    df = df.dropna(subset=['target_id'])
    df['target_id'] = df['target_id'].astype(int)
    
    # Handle multi-planet systems
    if 'period' in df.columns:
        df = df.drop_duplicates(subset=['target_id', 'period'], keep='first')
    
    print(f"   Found {len(df)} candidates in metadata")
    
    # Build metadata dictionary
    meta_dict = {}
    for _, row in df.iterrows():
        tid = int(row['target_id'])
        if tid not in meta_dict:
            meta_dict[tid] = []
        meta_dict[tid].append(row.to_dict())
    
    # Scan for FITS files from ALL cache directories
    print(f"\n[2/4] Scanning for FITS files in multiple directories...")
    
    file_map = {}
    total_fits = 0
    
    for cache_dir in CACHE_DIRS:
        if not os.path.exists(cache_dir):
            print(f"   Skipping (not found): {cache_dir}")
            continue
            
        fits_files = glob.glob(os.path.join(cache_dir, "**", "*_llc.fits"), recursive=True)
        print(f"   Found {len(fits_files)} files in {cache_dir}")
        total_fits += len(fits_files)
        
        # Map KIC ID to file path (prefer long cadence)
        for f in fits_files:
            basename = os.path.basename(f)
            if basename.startswith('kplr'):
                try:
                    id_str = basename[4:13]
                    kic_id = int(id_str)
                    # Only add if not already mapped (first found wins)
                    if kic_id not in file_map:
                        file_map[kic_id] = f
                except:
                    continue
    
    print(f"   Total: {total_fits} FITS files, {len(file_map)} unique KIC IDs")
    
    # Prepare tasks
    print("\n[3/4] Preparing processing tasks...")
    tasks = []
    for kic_id, fits_path in file_map.items():
        if kic_id in meta_dict:
            for candidate_meta in meta_dict[kic_id]:
                tasks.append((kic_id, fits_path, candidate_meta, OUTPUT_DIR))
    
    print(f"   Matched {len(tasks)} candidates for processing")
    
    if len(tasks) == 0:
        print("ERROR: No candidates matched. Check file paths and metadata.")
        return
    
    # Process with multiprocessing
    print(f"\n[4/4] Processing with multiprocessing...")
    
    # Use all but 2 CPU cores
    num_workers = max(1, cpu_count() - 2)
    print(f"   Using {num_workers} CPU workers")
    print(f"   (GPU not used - I/O bound operations are faster on CPU)")
    print()
    
    success_count = 0
    fail_count = 0
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_star_v2, tasks),
            total=len(tasks),
            desc="Processing",
            unit="candidate"
        ))
    
    # Count results
    for result in results:
        if result[0]:
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"   Successful: {success_count}")
    print(f"   Failed:     {fail_count}")
    print(f"   Output dir: {OUTPUT_DIR}")
    print(f"   Error log:  {LOG_FILE}")
    print()
    print("Next step: Update train_exotransformer_ultimate.py to use new data")
    print("=" * 70)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
