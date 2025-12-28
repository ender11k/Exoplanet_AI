import os
import logging
from astropy.io import fits
from tqdm import tqdm
import warnings

# Suppress astropy warnings for cleaner output (e.g. non-standard keywords)
warnings.filterwarnings('ignore', category=UserWarning, append=True)

# ==========================================
# Configuration
# ==========================================
CACHE_DIR = r"D:\.lightkurve\cache"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "cache_cleanup.log")

# ==========================================
# Setup Logging
# ==========================================
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # Overwrite log each run
)

def get_all_fits_files(root_dir):
    """Recursively find all .fits files in the directory."""
    fits_files = []
    print(f"Scanning {root_dir} for .fits files...")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".fits"):
                fits_files.append(os.path.join(root, file))
    return fits_files

def validate_and_clean(filepath):
    """
    Validates a FITS file for consistency and truncation.
    Returns (is_corrupt, bytes_reclaimed).
    """
    try:
        # Open with checksum verification to catch bit-rot or partial writes
        with fits.open(filepath, mode='readonly', checksum=True) as hdul:
            # Crucial Step: Force-read the data payload.
            # Merely opening the header is not enough to detect truncation.
            # Most Lightkurve files have data in extension 1 (LIGHTCURVE).
            if len(hdul) > 1:
                _ = hdul[1].data
            else:
                # If it's a single-HDU file, check primary data
                _ = hdul[0].data
                
        return False, 0
        
    except (OSError, ValueError, TypeError, Exception) as e:
        # File is corrupt or truncated
        try:
            file_size = os.path.getsize(filepath)
            os.remove(filepath)
            logging.error(f"CORRUPT: {filepath} | Error: {str(e)}")
            return True, file_size
        except Exception as del_e:
            logging.error(f"FAILED TO DELETE: {filepath} | Error: {str(del_e)}")
            return False, 0

def main():
    print("Starting Cache Audit & Scrub...")
    print(f"Target Directory: {CACHE_DIR}")
    
    if not os.path.exists(CACHE_DIR):
        print(f"Error: Cache directory {CACHE_DIR} does not exist.")
        return

    files = get_all_fits_files(CACHE_DIR)
    total_files = len(files)
    
    if total_files == 0:
        print("No FITS files found.")
        return

    print(f"Found {total_files} FITS files. Beginning validation...")

    corrupt_count = 0
    reclaimed_bytes = 0

    # Use tqdm for progress bar
    for filepath in tqdm(files, desc="Validating", unit="file"):
        is_corrupt, size = validate_and_clean(filepath)
        if is_corrupt:
            corrupt_count += 1
            reclaimed_bytes += size

    reclaimed_mb = reclaimed_bytes / (1024 * 1024)
    
    # Final Summary
    print("\n" + "="*40)
    print("AUDIT COMPLETE")
    print("="*40)
    print(f"Scanned files: {total_files}")
    print(f"Corrupt files detected & deleted: {corrupt_count}")
    print(f"Space reclaimed: {reclaimed_mb:.2f} MB")
    print(f"Detailed log saved to: {LOG_FILE}")
    print("="*40)

if __name__ == "__main__":
    main()
