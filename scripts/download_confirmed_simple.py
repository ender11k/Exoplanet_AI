#!/usr/bin/env python3
"""
=============================================================================
DOWNLOAD CONFIRMED EXOPLANETS - FIXED VERSION
=============================================================================
Downloads ALL confirmed exoplanet light curves using lightkurve library
which handles NASA/MAST queries properly.

This is the optimal approach - uses lightkurve's robust search functionality
instead of raw TAP queries.

Author: Exoplanet AI Research Team
Date: 2026
=============================================================================
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import argparse
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Install/import lightkurve
try:
    import lightkurve as lk
except ImportError:
    os.system(f"{sys.executable} -m pip install lightkurve")
    import lightkurve as lk

# Setup logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/download_exoplanets.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_confirmed_kepler_planets() -> pd.DataFrame:
    """
    Get confirmed Kepler planets from NASA Exoplanet Archive
    Using the correct API endpoint and query format
    """
    import requests
    
    logger.info("Fetching confirmed Kepler planets from NASA Exoplanet Archive...")
    
    # Correct TAP query URL format
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    # Simpler query that works
    query = "SELECT pl_name,hostname,kepid,pl_orbper,pl_rade,disc_facility FROM ps WHERE kepid IS NOT NULL AND default_flag=1"
    
    params = {
        "query": query,
        "format": "csv"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=120)
        
        if response.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            # Filter to Kepler discoveries
            df = df[df['disc_facility'].str.contains('Kepler', na=False, case=False)]
            logger.info(f"Found {len(df)} confirmed Kepler planets")
            return df
        else:
            logger.warning(f"API returned status {response.status_code}, using local KOI data")
            return pd.DataFrame()
            
    except Exception as e:
        logger.warning(f"Could not fetch from NASA Archive: {e}")
        return pd.DataFrame()


def get_local_confirmed_kois(koi_file: str = "data/koi_df.csv") -> pd.DataFrame:
    """
    Get confirmed KOIs from local data file
    """
    koi_path = Path(koi_file)
    
    if not koi_path.exists():
        logger.warning(f"Local KOI file not found: {koi_file}")
        return pd.DataFrame()
    
    df = pd.read_csv(koi_path)
    
    # Filter to confirmed planets
    if 'koi_disposition' in df.columns:
        confirmed = df[df['koi_disposition'] == 'CONFIRMED']
    elif 'disposition' in df.columns:
        confirmed = df[df['disposition'].str.contains('CONFIRMED', case=False, na=False)]
    else:
        confirmed = df
    
    logger.info(f"Found {len(confirmed)} confirmed KOIs in local data")
    return confirmed


def download_light_curve(
    target_id: str,
    period: float,
    output_dir: Path,
    mission: str = "Kepler"
) -> bool:
    """
    Download and process a single light curve
    
    Args:
        target_id: Target identifier (e.g., "KIC 12345678")
        period: Orbital period in days
        output_dir: Output directory for NPZ files
        mission: Mission name ("Kepler" or "TESS")
    
    Returns:
        True if successful, False otherwise
    """
    # Parse KIC ID
    if isinstance(target_id, str):
        kic_id = target_id.replace("KIC ", "").replace("KIC", "").strip()
    else:
        kic_id = str(int(target_id))
    
    # Create output filename
    filename = f"KIC_{kic_id}_P{period:.2f}.npz"
    filepath = output_dir / filename
    
    # Skip if already exists
    if filepath.exists():
        return True
    
    try:
        # Search for light curves
        search_result = lk.search_lightcurve(
            f"KIC {kic_id}",
            mission=mission
        )
        
        if len(search_result) == 0:
            logger.debug(f"No light curves found for KIC {kic_id}")
            return False
        
        # Download (get first available)
        try:
            lc_collection = search_result[:5].download_all()  # Limit to first 5 quarters
        except Exception as e:
            logger.debug(f"Download failed for KIC {kic_id}: {e}")
            return False
        
        if lc_collection is None or len(lc_collection) == 0:
            return False
        
        # Stitch quarters together
        try:
            lc = lc_collection.stitch()
        except:
            lc = lc_collection[0]
        
        # Process light curve
        lc = lc.remove_nans()
        lc = lc.normalize()
        
        try:
            lc_flat = lc.flatten(window_length=301)
        except:
            lc_flat = lc
        
        # Get arrays
        time_arr = lc_flat.time.value
        flux_arr = lc_flat.flux.value
        
        # Phase fold if we have period
        if period > 0:
            try:
                folded = lc_flat.fold(period=period)
                phase = folded.phase.value
                folded_flux = folded.flux.value
                
                # Sort by phase
                sort_idx = np.argsort(phase)
                phase = phase[sort_idx]
                folded_flux = folded_flux[sort_idx]
            except:
                # Create artificial phase array
                phase = np.linspace(-0.5, 0.5, len(flux_arr))
                folded_flux = flux_arr
        else:
            phase = np.linspace(-0.5, 0.5, len(flux_arr))
            folded_flux = flux_arr
        
        # Create global view (201 bins)
        global_view = bin_lightcurve(phase, folded_flux, n_bins=201)
        
        # Create local view (61 bins around transit)
        local_mask = np.abs(phase) < 0.1
        if np.sum(local_mask) > 10:
            local_view = bin_lightcurve(
                phase[local_mask], 
                folded_flux[local_mask], 
                n_bins=61
            )
        else:
            local_view = global_view[:61]
        
        # Save NPZ file
        np.savez(
            filepath,
            global_view=global_view.astype(np.float32),
            local_view=local_view.astype(np.float32),
            period=period,
            kepid=int(kic_id),
            label=1,  # Confirmed exoplanet
            mission=mission
        )
        
        logger.info(f"✓ Downloaded KIC {kic_id} (P={period:.2f}d)")
        return True
        
    except Exception as e:
        logger.debug(f"Error processing KIC {kic_id}: {e}")
        return False


def bin_lightcurve(phase: np.ndarray, flux: np.ndarray, n_bins: int = 201) -> np.ndarray:
    """Bin a light curve into fixed number of bins"""
    binned = np.ones(n_bins, dtype=np.float32)
    
    # Remove NaN/inf
    valid = np.isfinite(phase) & np.isfinite(flux)
    phase = phase[valid]
    flux = flux[valid]
    
    if len(phase) == 0:
        return binned
    
    # Create bin edges
    edges = np.linspace(phase.min(), phase.max(), n_bins + 1)
    
    for i in range(n_bins):
        mask = (phase >= edges[i]) & (phase < edges[i + 1])
        if np.sum(mask) > 0:
            binned[i] = np.median(flux[mask])
    
    return binned


def download_all_confirmed(
    output_dir: str,
    max_downloads: Optional[int] = None,
    use_local: bool = True
):
    """
    Download all confirmed exoplanet light curves
    
    Args:
        output_dir: Directory to save NPZ files
        max_downloads: Maximum number to download (None for all)
        use_local: Use local KOI data if NASA API fails
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("DOWNLOADING CONFIRMED EXOPLANET LIGHT CURVES")
    logger.info("="*60)
    
    # Get planet list
    planets_df = get_confirmed_kepler_planets()
    
    # Fall back to local data if API failed
    if planets_df.empty and use_local:
        planets_df = get_local_confirmed_kois()
        
        # Map column names
        if 'kepid' not in planets_df.columns and 'kic_id' in planets_df.columns:
            planets_df['kepid'] = planets_df['kic_id']
        if 'pl_orbper' not in planets_df.columns and 'koi_period' in planets_df.columns:
            planets_df['pl_orbper'] = planets_df['koi_period']
    
    if planets_df.empty:
        logger.error("No planet data available!")
        return
    
    # Filter to valid entries
    if 'kepid' in planets_df.columns and 'pl_orbper' in planets_df.columns:
        planets_df = planets_df.dropna(subset=['kepid', 'pl_orbper'])
    elif 'kepid' in planets_df.columns:
        planets_df = planets_df.dropna(subset=['kepid'])
        planets_df['pl_orbper'] = 1.0  # Default period
    else:
        logger.error("Missing required columns (kepid)")
        return
    
    # Limit if specified
    if max_downloads:
        planets_df = planets_df.head(max_downloads)
    
    total = len(planets_df)
    logger.info(f"Will attempt to download {total} confirmed planets")
    
    # Download stats
    downloaded = 0
    skipped = 0
    failed = 0
    
    for idx, row in planets_df.iterrows():
        kepid = int(row['kepid'])
        period = row.get('pl_orbper', row.get('koi_period', 1.0))
        
        if pd.isna(period) or period <= 0:
            period = 1.0
        
        progress = (idx + 1) / total * 100
        
        # Check if already exists
        filename = f"KIC_{kepid}_P{period:.2f}.npz"
        if (output_path / filename).exists():
            skipped += 1
            continue
        
        logger.info(f"[{progress:.1f}%] Downloading KIC {kepid}...")
        
        success = download_light_curve(
            target_id=str(kepid),
            period=float(period),
            output_dir=output_path,
            mission="Kepler"
        )
        
        if success:
            downloaded += 1
        else:
            failed += 1
        
        # Rate limiting
        time.sleep(0.3)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("="*60)
    logger.info(f"Downloaded: {downloaded}")
    logger.info(f"Skipped (already exists): {skipped}")
    logger.info(f"Failed: {failed}")
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "total_attempted": total
    }
    
    with open(output_path / "download_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download confirmed exoplanet light curves")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="notebooks/results_confirmed",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--max_downloads",
        type=int,
        default=None,
        help="Maximum number of planets to download"
    )
    parser.add_argument(
        "--use_local",
        action="store_true",
        default=True,
        help="Use local KOI data if NASA API fails"
    )
    
    args = parser.parse_args()
    
    download_all_confirmed(
        output_dir=args.output_dir,
        max_downloads=args.max_downloads,
        use_local=args.use_local
    )


if __name__ == "__main__":
    main()
