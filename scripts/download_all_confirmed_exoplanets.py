#!/usr/bin/env python3
"""
=============================================================================
COMPREHENSIVE EXOPLANET DATA DOWNLOADER
=============================================================================
Downloads ALL confirmed exoplanet light curves from:
1. Kepler Mission (primary source)
2. K2 Mission (extended Kepler)
3. TESS Mission (newest data)

Goal: Maximize positive samples for training a NASA-level ExoMiner/ExoTransformer model

Author: Exoplanet AI Research Team
Date: 2026
=============================================================================
"""

import os
import sys
import time
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple
import argparse
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try importing lightkurve (primary library for Kepler/TESS data)
try:
    import lightkurve as lk
    LIGHTKURVE_AVAILABLE = True
except ImportError:
    LIGHTKURVE_AVAILABLE = False
    print("⚠️ lightkurve not installed. Installing...")
    os.system(f"{sys.executable} -m pip install lightkurve")
    import lightkurve as lk
    LIGHTKURVE_AVAILABLE = True

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/download_exoplanets.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NASAExoplanetArchive:
    """
    Interface to NASA Exoplanet Archive TAP Service
    Downloads confirmed exoplanet catalog with host star information
    """
    
    BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    @staticmethod
    def query_confirmed_planets(mission: str = "all") -> pd.DataFrame:
        """
        Query NASA Exoplanet Archive for confirmed planets
        
        Args:
            mission: "kepler", "k2", "tess", or "all"
            
        Returns:
            DataFrame with confirmed exoplanets
        """
        logger.info(f"Querying NASA Exoplanet Archive for {mission} confirmed planets...")
        
        # Build query based on mission
        if mission == "kepler":
            # Kepler confirmed planets with KIC ID
            query = """
            SELECT pl_name, hostname, kepid, pl_orbper, pl_rade, pl_bmasse,
                   pl_eqt, st_teff, st_rad, st_mass, disc_facility, disc_year,
                   sy_kepmag, pl_trandur, pl_tranmid
            FROM ps
            WHERE kepid IS NOT NULL 
            AND disc_facility LIKE '%Kepler%'
            AND default_flag = 1
            ORDER BY kepid
            """
        elif mission == "k2":
            # K2 confirmed planets with EPIC ID
            query = """
            SELECT pl_name, hostname, pl_orbper, pl_rade, pl_bmasse,
                   pl_eqt, st_teff, st_rad, st_mass, disc_facility, disc_year,
                   pl_trandur, pl_tranmid
            FROM ps
            WHERE disc_facility LIKE '%K2%'
            AND default_flag = 1
            ORDER BY hostname
            """
        elif mission == "tess":
            # TESS confirmed planets with TIC ID
            query = """
            SELECT pl_name, hostname, tic_id, pl_orbper, pl_rade, pl_bmasse,
                   pl_eqt, st_teff, st_rad, st_mass, disc_facility, disc_year,
                   pl_trandur, pl_tranmid
            FROM ps
            WHERE tic_id IS NOT NULL
            AND disc_facility LIKE '%TESS%'
            AND default_flag = 1
            ORDER BY tic_id
            """
        else:
            # All confirmed planets
            query = """
            SELECT pl_name, hostname, kepid, tic_id, pl_orbper, pl_rade, pl_bmasse,
                   pl_eqt, st_teff, st_rad, st_mass, disc_facility, disc_year,
                   sy_kepmag, pl_trandur, pl_tranmid
            FROM ps
            WHERE default_flag = 1
            ORDER BY disc_year DESC
            """
        
        params = {
            "query": query,
            "format": "csv"
        }
        
        try:
            response = requests.get(NASAExoplanetArchive.BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            
            # Parse CSV response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            logger.info(f"Found {len(df)} confirmed planets from {mission}")
            return df
            
        except Exception as e:
            logger.error(f"Error querying NASA Exoplanet Archive: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_kepler_candidates() -> pd.DataFrame:
        """
        Get Kepler Objects of Interest (KOI) - includes candidates and confirmed
        This gives us more data for training
        """
        logger.info("Querying Kepler Objects of Interest (KOI)...")
        
        query = """
        SELECT kepoi_name, kepid, koi_period, koi_ror, koi_duration, koi_depth,
               koi_disposition, koi_pdisposition, koi_score, koi_time0bk,
               koi_steff, koi_srad, koi_smass, koi_kepmag, koi_prad, koi_teq
        FROM koi
        WHERE koi_disposition IN ('CONFIRMED', 'CANDIDATE')
        ORDER BY kepid
        """
        
        params = {
            "query": query,
            "format": "csv"
        }
        
        try:
            response = requests.get(NASAExoplanetArchive.BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            logger.info(f"Found {len(df)} KOIs (confirmed + candidates)")
            return df
            
        except Exception as e:
            logger.error(f"Error querying KOI table: {e}")
            return pd.DataFrame()


class LightCurveDownloader:
    """
    Downloads and processes light curves from MAST using lightkurve
    """
    
    def __init__(self, output_dir: str, mission: str = "Kepler"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mission = mission
        self.downloaded = 0
        self.failed = 0
        self.skipped = 0
        
    def download_kepler_lightcurve(
        self, 
        kepid: int, 
        period: float,
        t0: Optional[float] = None,
        duration: Optional[float] = None
    ) -> bool:
        """
        Download and process Kepler light curve for a confirmed exoplanet
        
        Args:
            kepid: Kepler Input Catalog ID
            period: Orbital period in days
            t0: Transit epoch (BKJD)
            duration: Transit duration in hours
            
        Returns:
            True if successful, False otherwise
        """
        # Create filename
        filename = f"KIC_{kepid}_P{period:.2f}.npz"
        filepath = self.output_dir / filename
        
        # Skip if already exists
        if filepath.exists():
            self.skipped += 1
            return True
            
        try:
            # Search for light curves
            search_result = lk.search_lightcurve(
                f"KIC {kepid}",
                mission="Kepler",
                author="Kepler"
            )
            
            if len(search_result) == 0:
                logger.warning(f"No light curves found for KIC {kepid}")
                self.failed += 1
                return False
            
            # Download all quarters and stitch together
            lc_collection = search_result.download_all()
            
            if lc_collection is None or len(lc_collection) == 0:
                logger.warning(f"Failed to download light curves for KIC {kepid}")
                self.failed += 1
                return False
            
            # Stitch all quarters together
            lc = lc_collection.stitch()
            
            # Remove NaN values
            lc = lc.remove_nans()
            
            # Normalize flux
            lc = lc.normalize()
            
            # Flatten to remove stellar variability
            lc_flat = lc.flatten(window_length=401)
            
            # Get time and flux arrays
            time = lc_flat.time.value
            flux = lc_flat.flux.value
            
            # Phase-fold if we have period
            if period and period > 0:
                # Fold the light curve
                folded_lc = lc_flat.fold(period=period)
                phase = folded_lc.phase.value
                folded_flux = folded_lc.flux.value
                
                # Sort by phase
                sort_idx = np.argsort(phase)
                phase = phase[sort_idx]
                folded_flux = folded_flux[sort_idx]
                
                # Bin the folded light curve to standard size (201 points like NASA)
                n_bins = 201
                phase_bins = np.linspace(phase.min(), phase.max(), n_bins + 1)
                binned_flux = np.zeros(n_bins)
                
                for i in range(n_bins):
                    mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
                    if np.sum(mask) > 0:
                        binned_flux[i] = np.median(folded_flux[mask])
                    else:
                        binned_flux[i] = 1.0  # Default normalized value
                
                # Also create global view (larger phase range)
                global_view = self._create_global_view(phase, folded_flux, n_bins=201)
                
                # Create local view (zoomed around transit)
                local_view = self._create_local_view(phase, folded_flux, n_bins=61)
                
            else:
                # No period - use raw binned light curve
                n_bins = 201
                binned_flux = self._bin_lightcurve(flux, n_bins)
                global_view = binned_flux
                local_view = binned_flux[:61]
            
            # Save as NPZ file
            np.savez(
                filepath,
                global_view=global_view,
                local_view=local_view,
                time=time[:10000] if len(time) > 10000 else time,  # Limit size
                flux=flux[:10000] if len(flux) > 10000 else flux,
                period=period,
                kepid=kepid,
                label=1,  # Confirmed exoplanet
                mission="Kepler"
            )
            
            self.downloaded += 1
            logger.info(f"✓ Downloaded KIC {kepid} (period={period:.2f}d)")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading KIC {kepid}: {e}")
            self.failed += 1
            return False
    
    def download_tess_lightcurve(
        self,
        tic_id: int,
        period: float,
        t0: Optional[float] = None
    ) -> bool:
        """
        Download and process TESS light curve
        """
        filename = f"TIC_{tic_id}_P{period:.2f}.npz"
        filepath = self.output_dir / filename
        
        if filepath.exists():
            self.skipped += 1
            return True
            
        try:
            # Search for TESS light curves
            search_result = lk.search_lightcurve(
                f"TIC {tic_id}",
                mission="TESS",
                author="SPOC"
            )
            
            if len(search_result) == 0:
                # Try QLP pipeline
                search_result = lk.search_lightcurve(
                    f"TIC {tic_id}",
                    mission="TESS"
                )
            
            if len(search_result) == 0:
                self.failed += 1
                return False
            
            # Download and stitch
            lc_collection = search_result.download_all()
            if lc_collection is None:
                self.failed += 1
                return False
                
            lc = lc_collection.stitch()
            lc = lc.remove_nans().normalize().flatten(window_length=101)
            
            time = lc.time.value
            flux = lc.flux.value
            
            # Phase fold
            if period > 0:
                folded_lc = lc.fold(period=period)
                phase = folded_lc.phase.value
                folded_flux = folded_lc.flux.value
                
                sort_idx = np.argsort(phase)
                phase = phase[sort_idx]
                folded_flux = folded_flux[sort_idx]
                
                global_view = self._create_global_view(phase, folded_flux, 201)
                local_view = self._create_local_view(phase, folded_flux, 61)
            else:
                global_view = self._bin_lightcurve(flux, 201)
                local_view = global_view[:61]
            
            np.savez(
                filepath,
                global_view=global_view,
                local_view=local_view,
                time=time[:10000] if len(time) > 10000 else time,
                flux=flux[:10000] if len(flux) > 10000 else flux,
                period=period,
                tic_id=tic_id,
                label=1,
                mission="TESS"
            )
            
            self.downloaded += 1
            logger.info(f"✓ Downloaded TIC {tic_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading TIC {tic_id}: {e}")
            self.failed += 1
            return False
    
    def _create_global_view(self, phase: np.ndarray, flux: np.ndarray, n_bins: int = 201) -> np.ndarray:
        """Create global view binned light curve"""
        phase_bins = np.linspace(phase.min(), phase.max(), n_bins + 1)
        binned = np.ones(n_bins)
        
        for i in range(n_bins):
            mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
            if np.sum(mask) > 0:
                binned[i] = np.median(flux[mask])
                
        return binned
    
    def _create_local_view(self, phase: np.ndarray, flux: np.ndarray, n_bins: int = 61) -> np.ndarray:
        """Create local view centered on transit (phase ~0)"""
        # Zoom to ±0.1 phase (around transit)
        local_mask = np.abs(phase) < 0.1
        
        if np.sum(local_mask) < 10:
            # Not enough points, use global
            return self._create_global_view(phase, flux, n_bins)
        
        local_phase = phase[local_mask]
        local_flux = flux[local_mask]
        
        phase_bins = np.linspace(-0.1, 0.1, n_bins + 1)
        binned = np.ones(n_bins)
        
        for i in range(n_bins):
            mask = (local_phase >= phase_bins[i]) & (local_phase < phase_bins[i + 1])
            if np.sum(mask) > 0:
                binned[i] = np.median(local_flux[mask])
                
        return binned
    
    def _bin_lightcurve(self, flux: np.ndarray, n_bins: int) -> np.ndarray:
        """Bin a light curve to fixed number of bins"""
        binned = np.ones(n_bins)
        bin_size = len(flux) // n_bins
        
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size
            if end <= len(flux):
                binned[i] = np.median(flux[start:end])
                
        return binned


def download_all_kepler_confirmed(output_dir: str, max_workers: int = 4):
    """
    Download ALL confirmed Kepler exoplanet light curves
    """
    logger.info("=" * 60)
    logger.info("DOWNLOADING ALL CONFIRMED KEPLER EXOPLANETS")
    logger.info("=" * 60)
    
    # Get confirmed planets from NASA archive
    planets_df = NASAExoplanetArchive.query_confirmed_planets("kepler")
    
    if planets_df.empty:
        logger.error("Failed to get confirmed planets list")
        return
    
    # Also get KOI data for additional info
    koi_df = NASAExoplanetArchive.get_kepler_candidates()
    
    # Filter to only confirmed in KOI table too
    if not koi_df.empty:
        confirmed_koi = koi_df[koi_df['koi_disposition'] == 'CONFIRMED']
        logger.info(f"Found {len(confirmed_koi)} confirmed KOIs")
    
    # Get unique KepIDs with their periods
    planets_df = planets_df.dropna(subset=['kepid', 'pl_orbper'])
    planets_df['kepid'] = planets_df['kepid'].astype(int)
    
    logger.info(f"Total Kepler planets to download: {len(planets_df)}")
    
    # Initialize downloader
    downloader = LightCurveDownloader(output_dir, mission="Kepler")
    
    # Download with progress
    total = len(planets_df)
    
    for idx, row in planets_df.iterrows():
        kepid = int(row['kepid'])
        period = row['pl_orbper']
        t0 = row.get('pl_tranmid', None)
        duration = row.get('pl_trandur', None)
        
        progress = (idx + 1) / total * 100
        logger.info(f"[{progress:.1f}%] Processing {row['pl_name']} (KIC {kepid})")
        
        downloader.download_kepler_lightcurve(kepid, period, t0, duration)
        
        # Be nice to MAST servers
        time.sleep(0.5)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info(f"Downloaded: {downloader.downloaded}")
    logger.info(f"Skipped (already exists): {downloader.skipped}")
    logger.info(f"Failed: {downloader.failed}")
    logger.info("=" * 60)
    
    return downloader.downloaded


def download_all_tess_confirmed(output_dir: str):
    """
    Download ALL confirmed TESS exoplanet light curves
    """
    logger.info("=" * 60)
    logger.info("DOWNLOADING ALL CONFIRMED TESS EXOPLANETS")
    logger.info("=" * 60)
    
    planets_df = NASAExoplanetArchive.query_confirmed_planets("tess")
    
    if planets_df.empty:
        logger.error("Failed to get TESS confirmed planets")
        return
    
    planets_df = planets_df.dropna(subset=['tic_id', 'pl_orbper'])
    planets_df['tic_id'] = planets_df['tic_id'].astype(int)
    
    logger.info(f"Total TESS planets to download: {len(planets_df)}")
    
    downloader = LightCurveDownloader(output_dir, mission="TESS")
    
    for idx, row in planets_df.iterrows():
        tic_id = int(row['tic_id'])
        period = row['pl_orbper']
        t0 = row.get('pl_tranmid', None)
        
        progress = (idx + 1) / len(planets_df) * 100
        logger.info(f"[{progress:.1f}%] Processing {row['pl_name']} (TIC {tic_id})")
        
        downloader.download_tess_lightcurve(tic_id, period, t0)
        time.sleep(0.5)
    
    logger.info(f"Downloaded: {downloader.downloaded}")
    logger.info(f"Skipped: {downloader.skipped}")
    logger.info(f"Failed: {downloader.failed}")
    
    return downloader.downloaded


def download_koi_candidates(output_dir: str, include_candidates: bool = True):
    """
    Download KOI candidates (high-confidence candidates can augment training)
    Only downloads candidates with koi_score > 0.9 (very likely planets)
    """
    logger.info("=" * 60)
    logger.info("DOWNLOADING HIGH-CONFIDENCE KOI CANDIDATES")
    logger.info("=" * 60)
    
    koi_df = NASAExoplanetArchive.get_kepler_candidates()
    
    if koi_df.empty:
        return 0
    
    # Filter to high-confidence candidates
    if include_candidates:
        # Include confirmed + high-score candidates
        high_conf = koi_df[
            (koi_df['koi_disposition'] == 'CONFIRMED') |
            ((koi_df['koi_disposition'] == 'CANDIDATE') & (koi_df['koi_score'] > 0.9))
        ]
    else:
        high_conf = koi_df[koi_df['koi_disposition'] == 'CONFIRMED']
    
    high_conf = high_conf.dropna(subset=['kepid', 'koi_period'])
    
    logger.info(f"High-confidence KOIs to download: {len(high_conf)}")
    
    downloader = LightCurveDownloader(output_dir, mission="Kepler")
    
    for idx, row in high_conf.iterrows():
        kepid = int(row['kepid'])
        period = row['koi_period']
        
        progress = (idx + 1) / len(high_conf) * 100
        logger.info(f"[{progress:.1f}%] Processing {row['kepoi_name']} (KIC {kepid})")
        
        downloader.download_kepler_lightcurve(kepid, period)
        time.sleep(0.5)
    
    logger.info(f"Downloaded: {downloader.downloaded}")
    return downloader.downloaded


def verify_data_quality(data_dir: str) -> Dict:
    """
    Verify downloaded data quality and statistics
    """
    logger.info("Verifying data quality...")
    
    data_path = Path(data_dir)
    files = list(data_path.glob("*.npz"))
    
    stats = {
        "total_files": len(files),
        "kepler_files": 0,
        "tess_files": 0,
        "valid_files": 0,
        "corrupt_files": [],
        "size_distribution": []
    }
    
    for f in files:
        try:
            data = np.load(f)
            
            if 'global_view' in data and 'local_view' in data:
                global_view = data['global_view']
                local_view = data['local_view']
                
                # Check for valid data
                if len(global_view) == 201 and len(local_view) == 61:
                    if not np.any(np.isnan(global_view)) and not np.any(np.isnan(local_view)):
                        stats['valid_files'] += 1
                        
                        if 'KIC' in f.name:
                            stats['kepler_files'] += 1
                        elif 'TIC' in f.name:
                            stats['tess_files'] += 1
                            
                        stats['size_distribution'].append(f.stat().st_size)
                    else:
                        stats['corrupt_files'].append(f.name)
                else:
                    stats['corrupt_files'].append(f.name)
            else:
                stats['corrupt_files'].append(f.name)
                
        except Exception as e:
            stats['corrupt_files'].append(f.name)
    
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Valid files: {stats['valid_files']}")
    logger.info(f"Kepler files: {stats['kepler_files']}")
    logger.info(f"TESS files: {stats['tess_files']}")
    logger.info(f"Corrupt files: {len(stats['corrupt_files'])}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download ALL confirmed exoplanet light curves for training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="notebooks/results_confirmed",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--missions",
        type=str,
        nargs="+",
        default=["kepler", "tess"],
        choices=["kepler", "tess", "k2", "koi"],
        help="Missions to download data from"
    )
    parser.add_argument(
        "--include_candidates",
        action="store_true",
        help="Include high-confidence candidates (score > 0.9)"
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify existing data, don't download"
    )
    parser.add_argument(
        "--max_per_mission",
        type=int,
        default=None,
        help="Maximum planets to download per mission (for testing)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    if args.verify_only:
        stats = verify_data_quality(args.output_dir)
        print(json.dumps(stats, indent=2, default=str))
        return
    
    # Download from each mission
    total_downloaded = 0
    
    if "kepler" in args.missions:
        n = download_all_kepler_confirmed(args.output_dir)
        total_downloaded += n or 0
    
    if "tess" in args.missions:
        n = download_all_tess_confirmed(args.output_dir)
        total_downloaded += n or 0
    
    if "koi" in args.missions:
        n = download_koi_candidates(args.output_dir, args.include_candidates)
        total_downloaded += n or 0
    
    # Final verification
    logger.info("\n" + "=" * 60)
    logger.info("FINAL DATA VERIFICATION")
    logger.info("=" * 60)
    
    stats = verify_data_quality(args.output_dir)
    
    # Save download report
    report = {
        "download_date": datetime.now().isoformat(),
        "missions": args.missions,
        "total_downloaded": total_downloaded,
        "statistics": stats
    }
    
    with open(output_dir / "download_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\nDownload complete! Total new files: {total_downloaded}")
    logger.info(f"Report saved to: {output_dir / 'download_report.json'}")


if __name__ == "__main__":
    main()
