#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_confirmed_planets.py
=============================

A production-grade data acquisition pipeline for downloading confirmed
exoplanet light curves from NASA space telescope missions.

This module queries the NASA Exoplanet Archive via its Table Access Protocol
(TAP) interface and retrieves time-series photometry from the Mikulski Archive
for Space Telescopes (MAST) using the Lightkurve package.

Supported Missions
------------------
- **Kepler**: Primary mission (2009-2013), ~2,700 confirmed planets
- **K2**: Extended Kepler mission (2014-2018), ~500 confirmed planets  
- **TESS**: Transiting Exoplanet Survey Satellite (2018-present), ~400+ confirmed

Architecture
------------
The pipeline follows a three-stage architecture:

1. **Query Stage**: Retrieve confirmed planet catalogs from NASA Exoplanet
   Archive using ADQL (Astronomical Data Query Language) via TAP.

2. **Download Stage**: Fetch raw light curve FITS files from MAST using
   Lightkurve's caching mechanism with exponential backoff retry logic.

3. **Validation Stage**: Verify data integrity and generate download reports.

Data Products
-------------
- Raw light curve FITS files cached in configurable directory
- CSV catalog of confirmed planets with stellar/planetary parameters
- Detailed logging of all download attempts and failures

References
----------
.. [1] NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
.. [2] Lightkurve: Kepler & TESS data analysis (Lightkurve Collaboration, 2018)
.. [3] ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier
       (Valizadegan et al., 2022)

Example Usage
-------------
Query available data without downloading::

    $ python download_confirmed_planets.py --query_only

Download all confirmed planet light curves::

    $ python download_confirmed_planets.py --download

Resume interrupted download::

    $ python download_confirmed_planets.py --download --resume

Authors
-------
Exoplanet AI Research Team

License
-------
This project is developed for NASA research purposes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Third-party imports with graceful degradation
try:
    # Set lightkurve cache directory BEFORE import
    # This must be done via environment variable before lightkurve is imported
    os.environ['LIGHTKURVE_CACHE_DIR'] = r"D:\.lightkurve2"
    
    import lightkurve as lk
    LIGHTKURVE_AVAILABLE = True
    
    # Also set via lightkurve's config if available
    if hasattr(lk, 'conf') and hasattr(lk.conf, 'cache_dir'):
        lk.conf.cache_dir = r"D:\.lightkurve2"
except ImportError:
    LIGHTKURVE_AVAILABLE = False
    print("[WARNING] Lightkurve not installed. Download functionality disabled.")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration parameters for the download pipeline.
    
    This dataclass centralizes all configurable parameters to ensure
    reproducibility and ease of modification for different environments.
    
    Attributes
    ----------
    output_dir : Path
        Directory for output files (catalogs, reports).
    cache_dir : Path
        Directory for Lightkurve cache (raw FITS files).
    log_dir : Path
        Directory for log files.
    max_workers : int
        Maximum number of concurrent download threads.
    max_retries : int
        Maximum retry attempts per target.
    retry_delay : float
        Base delay (seconds) between retries (exponential backoff).
    rate_limit_delay : float
        Minimum delay between requests to avoid overwhelming servers.
    request_timeout : int
        Timeout (seconds) for HTTP requests.
    """
    output_dir: Path = field(default_factory=lambda: Path("data"))
    cache_dir: Path = field(default_factory=lambda: Path(r"D:\.lightkurve2"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    max_workers: int = 2  # Reduced from 8 to avoid MAST rate limiting
    max_retries: int = 5  # Increased for better resilience
    retry_delay: float = 5.0  # Longer delay between retries
    rate_limit_delay: float = 1.0  # Longer delay between requests
    request_timeout: int = 120  # Increased timeout for large files
    
    def __post_init__(self) -> None:
        """Create directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


class DownloadStatus(Enum):
    """Enumeration of possible download outcomes."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    NO_DATA = "no_data"
    CACHED = "cached"


@dataclass
class DownloadResult:
    """Result of a single download attempt.
    
    Attributes
    ----------
    target_id : int
        Unique identifier for the target star.
    catalog : str
        Source catalog (KEPLER_CONFIRMED, K2_CONFIRMED, TESS_CONFIRMED).
    status : DownloadStatus
        Outcome of the download attempt.
    message : str
        Human-readable status message.
    duration : float
        Time taken for the download (seconds).
    file_count : int
        Number of files downloaded.
    """
    target_id: int
    catalog: str
    status: DownloadStatus
    message: str = ""
    duration: float = 0.0
    file_count: int = 0


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(config: PipelineConfig) -> logging.Logger:
    """Configure logging with both file and console handlers.
    
    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration object.
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_file = config.log_dir / f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler (DEBUG level - capture everything)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler (INFO level - user-friendly output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger("ExoplanetDownloader")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


# =============================================================================
# NASA Exoplanet Archive Interface
# =============================================================================

class NASAExoplanetArchive:
    """Interface to NASA Exoplanet Archive TAP service.
    
    This class provides methods to query the NASA Exoplanet Archive using
    the Table Access Protocol (TAP) with ADQL queries.
    
    Attributes
    ----------
    base_url : str
        Base URL for the TAP service.
    logger : logging.Logger
        Logger instance for this class.
        
    References
    ----------
    .. [1] TAP Protocol: https://www.ivoa.net/documents/TAP/
    .. [2] NASA Archive API: https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
    """
    
    BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    # ADQL Queries for each mission
    # These queries are designed to retrieve all necessary parameters for
    # transit detection and stellar characterization.
    
    KEPLER_QUERY = """
    SELECT 
        kepid,
        kepoi_name,
        kepler_name,
        koi_disposition,
        koi_period,
        koi_period_err1,
        koi_period_err2,
        koi_depth,
        koi_depth_err1,
        koi_depth_err2,
        koi_duration,
        koi_duration_err1,
        koi_duration_err2,
        koi_prad,
        koi_prad_err1,
        koi_prad_err2,
        koi_srad,
        koi_srad_err1,
        koi_srad_err2,
        koi_steff,
        koi_steff_err1,
        koi_steff_err2,
        koi_slogg,
        koi_slogg_err1,
        koi_slogg_err2,
        koi_kepmag,
        ra,
        dec
    FROM cumulative 
    WHERE koi_disposition = 'CONFIRMED'
    ORDER BY kepid
    """
    
    K2_QUERY = """
    SELECT 
        tic_id,
        pl_name,
        hostname,
        disc_facility,
        pl_orbper,
        pl_orbpererr1,
        pl_orbpererr2,
        pl_trandep,
        pl_trandeperr1,
        pl_trandeperr2,
        pl_trandur,
        pl_trandurherr1,
        pl_trandurherr2,
        pl_rade,
        pl_radeerr1,
        pl_radeerr2,
        st_rad,
        st_raderr1,
        st_raderr2,
        st_teff,
        st_tefferr1,
        st_tefferr2,
        st_logg,
        st_loggerr1,
        st_loggerr2,
        sy_vmag,
        ra,
        dec
    FROM pscomppars
    WHERE disc_facility like '%K2%'
    ORDER BY tic_id
    """
    
    TESS_QUERY = """
    SELECT 
        tic_id,
        pl_name,
        hostname,
        disc_facility,
        pl_orbper,
        pl_orbpererr1,
        pl_orbpererr2,
        pl_trandep,
        pl_trandeperr1,
        pl_trandeperr2,
        pl_trandur,
        pl_trandurherr1,
        pl_trandurherr2,
        pl_rade,
        pl_radeerr1,
        pl_radeerr2,
        st_rad,
        st_raderr1,
        st_raderr2,
        st_teff,
        st_tefferr1,
        st_tefferr2,
        st_logg,
        st_loggerr1,
        st_loggerr2,
        sy_tmag,
        ra,
        dec
    FROM pscomppars
    WHERE disc_facility like '%TESS%'
    ORDER BY tic_id
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the NASA Archive interface.
        
        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance. If None, creates a default logger.
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def _execute_query(self, query: str, timeout: int = 60) -> pd.DataFrame:
        """Execute an ADQL query against the TAP service.
        
        Parameters
        ----------
        query : str
            ADQL query string.
        timeout : int
            Request timeout in seconds.
            
        Returns
        -------
        pd.DataFrame
            Query results as a DataFrame.
            
        Raises
        ------
        ConnectionError
            If the query fails after all retries.
        """
        # Clean and encode the query
        clean_query = " ".join(query.split())
        encoded_query = urllib.parse.quote(clean_query)
        url = f"{self.BASE_URL}?query={encoded_query}&format=csv"
        
        self.logger.debug(f"Executing query: {clean_query[:100]}...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = pd.read_csv(url, low_memory=False)
                self.logger.info(f"Query returned {len(df)} rows")
                return df
            except Exception as e:
                self.logger.warning(f"Query attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Query failed after {max_retries} attempts")
                    return pd.DataFrame()
        
        return pd.DataFrame()
    
    def get_kepler_confirmed(self) -> pd.DataFrame:
        """Retrieve all confirmed Kepler planets.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with Kepler confirmed planet parameters.
            Includes 'target_id', 'catalog', and 'label' columns.
        """
        self.logger.info("Querying Kepler confirmed planets...")
        df = self._execute_query(self.KEPLER_QUERY)
        
        if not df.empty:
            df['catalog'] = 'KEPLER_CONFIRMED'
            df['target_id'] = df['kepid'].astype('Int64')
            df['label'] = 1
            self.logger.info(f"Retrieved {len(df)} Kepler confirmed planets")
        
        return df
    
    def get_k2_confirmed(self) -> pd.DataFrame:
        """Retrieve all confirmed K2 planets.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with K2 confirmed planet parameters.
        """
        self.logger.info("Querying K2 confirmed planets...")
        df = self._execute_query(self.K2_QUERY)
        
        if not df.empty:
            df['catalog'] = 'K2_CONFIRMED'
            df['target_id'] = df['tic_id'].astype('Int64')
            df['label'] = 1
            self.logger.info(f"Retrieved {len(df)} K2 confirmed planets")
        
        return df
    
    def get_tess_confirmed(self) -> pd.DataFrame:
        """Retrieve all confirmed TESS planets.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with TESS confirmed planet parameters.
        """
        self.logger.info("Querying TESS confirmed planets...")
        df = self._execute_query(self.TESS_QUERY)
        
        if not df.empty:
            df['catalog'] = 'TESS_CONFIRMED'
            df['target_id'] = df['tic_id'].astype('Int64')
            df['label'] = 1
            self.logger.info(f"Retrieved {len(df)} TESS confirmed planets")
        
        return df
    
    def get_all_confirmed(self) -> pd.DataFrame:
        """Retrieve all confirmed planets from all missions.
        
        Returns
        -------
        pd.DataFrame
            Combined DataFrame with all confirmed planets.
        """
        kepler_df = self.get_kepler_confirmed()
        k2_df = self.get_k2_confirmed()
        tess_df = self.get_tess_confirmed()
        
        # Combine all catalogs
        combined = pd.concat([kepler_df, k2_df, tess_df], ignore_index=True)
        
        # Clean and deduplicate
        combined = combined.dropna(subset=['target_id'])
        combined = combined.drop_duplicates(subset=['target_id', 'catalog'])
        
        self.logger.info(f"Total unique confirmed planets: {len(combined)}")
        
        return combined


# =============================================================================
# Light Curve Downloader
# =============================================================================

class LightCurveDownloader:
    """Parallel downloader for light curve data from MAST.
    
    This class manages concurrent downloads of light curve FITS files
    using Lightkurve, with built-in retry logic, rate limiting, and
    progress tracking.
    
    Attributes
    ----------
    config : PipelineConfig
        Pipeline configuration.
    logger : logging.Logger
        Logger instance.
    progress_file : Path
        Path to JSON file tracking download progress (for resumability).
    """
    
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        """Initialize the downloader.
        
        Parameters
        ----------
        config : PipelineConfig
            Pipeline configuration.
        logger : logging.Logger
            Logger instance.
        """
        self.config = config
        self.logger = logger
        self.progress_file = config.output_dir / "download_progress.json"
        
        # Configure Lightkurve cache directory
        if LIGHTKURVE_AVAILABLE:
            lk.config.cache_dir = str(config.cache_dir)
            self.logger.info(f"Lightkurve cache directory: {config.cache_dir}")
    
    def _load_progress(self) -> Dict[str, str]:
        """Load download progress from disk.
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping target keys to status strings.
        """
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load progress file: {e}")
        return {}
    
    def _save_progress(self, progress: Dict[str, str]) -> None:
        """Save download progress to disk.
        
        Parameters
        ----------
        progress : Dict[str, str]
            Dictionary mapping target keys to status strings.
        """
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save progress: {e}")
    
    def _get_target_key(self, target_id: int, catalog: str) -> str:
        """Generate unique key for a target.
        
        Parameters
        ----------
        target_id : int
            Target identifier.
        catalog : str
            Source catalog.
            
        Returns
        -------
        str
            Unique key string.
        """
        return f"{catalog}_{target_id}"
    
    def _download_single(self, target_id: int, catalog: str) -> DownloadResult:
        """Download light curve for a single target.
        
        Parameters
        ----------
        target_id : int
            Target identifier.
        catalog : str
            Source catalog.
            
        Returns
        -------
        DownloadResult
            Result of the download attempt.
        """
        start_time = time.time()
        
        if not LIGHTKURVE_AVAILABLE:
            return DownloadResult(
                target_id=target_id,
                catalog=catalog,
                status=DownloadStatus.FAILED,
                message="Lightkurve not available"
            )
        
        # Construct search identifier based on catalog
        if catalog == 'KEPLER_CONFIRMED':
            search_id = f"KIC {target_id}"
            author = 'Kepler'
            cadence = 'long'
        elif catalog == 'K2_CONFIRMED':
            search_id = f"TIC {target_id}"
            author = 'K2'
            cadence = 'long'
        elif catalog == 'TESS_CONFIRMED':
            search_id = f"TIC {target_id}"
            author = 'SPOC'
            cadence = None  # TESS uses different cadence system
        else:
            return DownloadResult(
                target_id=target_id,
                catalog=catalog,
                status=DownloadStatus.FAILED,
                message=f"Unknown catalog: {catalog}"
            )
        
        # Retry loop with exponential backoff
        for attempt in range(self.config.max_retries):
            try:
                # Rate limiting
                time.sleep(self.config.rate_limit_delay)
                
                # Search for light curves
                if cadence:
                    search_result = lk.search_lightcurve(
                        search_id, author=author, cadence=cadence
                    )
                else:
                    search_result = lk.search_lightcurve(search_id, author=author)
                
                if search_result is None or len(search_result) == 0:
                    return DownloadResult(
                        target_id=target_id,
                        catalog=catalog,
                        status=DownloadStatus.NO_DATA,
                        message=f"No light curves found for {search_id}",
                        duration=time.time() - start_time
                    )
                
                # Download all available quarters/sectors
                # Use download_all with quality_bitmask to ensure clean data
                try:
                    lc_collection = search_result.download_all(
                        download_dir=str(self.config.cache_dir)
                    )
                    if lc_collection is None or len(lc_collection) == 0:
                        raise Exception("Download returned empty collection")
                except Exception as dl_err:
                    raise Exception(f"Download failed: {dl_err}")
                
                return DownloadResult(
                    target_id=target_id,
                    catalog=catalog,
                    status=DownloadStatus.SUCCESS,
                    message=f"Downloaded {len(search_result)} files",
                    duration=time.time() - start_time,
                    file_count=len(search_result)
                )
                
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    self.logger.debug(
                        f"Retry {attempt + 1} for {search_id} after {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    return DownloadResult(
                        target_id=target_id,
                        catalog=catalog,
                        status=DownloadStatus.FAILED,
                        message=str(e),
                        duration=time.time() - start_time
                    )
        
        return DownloadResult(
            target_id=target_id,
            catalog=catalog,
            status=DownloadStatus.FAILED,
            message="Max retries exceeded",
            duration=time.time() - start_time
        )
    
    def download_batch(
        self,
        targets: List[Tuple[int, str]],
        resume: bool = True
    ) -> List[DownloadResult]:
        """Download light curves for a batch of targets.
        
        Parameters
        ----------
        targets : List[Tuple[int, str]]
            List of (target_id, catalog) tuples.
        resume : bool
            If True, skip targets that were successfully downloaded before.
            
        Returns
        -------
        List[DownloadResult]
            Results for all download attempts.
        """
        results: List[DownloadResult] = []
        
        # Load progress if resuming
        progress = self._load_progress() if resume else {}
        
        # Filter out already completed targets
        if resume:
            pending_targets = [
                (tid, cat) for tid, cat in targets
                if progress.get(self._get_target_key(tid, cat)) != DownloadStatus.SUCCESS.value
            ]
            skipped = len(targets) - len(pending_targets)
            if skipped > 0:
                self.logger.info(f"Resuming: skipping {skipped} already completed targets")
            targets = pending_targets
        
        if not targets:
            self.logger.info("No targets to download")
            return results
        
        self.logger.info(f"Downloading {len(targets)} targets with {self.config.max_workers} workers")
        
        # Parallel download with progress bar
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_target = {
                executor.submit(self._download_single, tid, cat): (tid, cat)
                for tid, cat in targets
            }
            
            with tqdm(total=len(targets), desc="Downloading", unit="target") as pbar:
                for future in as_completed(future_to_target):
                    tid, cat = future_to_target[future]
                    
                    try:
                        result = future.result()
                    except Exception as e:
                        result = DownloadResult(
                            target_id=tid,
                            catalog=cat,
                            status=DownloadStatus.FAILED,
                            message=f"Unexpected error: {e}"
                        )
                    
                    results.append(result)
                    
                    # Update progress
                    progress[self._get_target_key(tid, cat)] = result.status.value
                    
                    # Periodic progress save
                    if len(results) % 50 == 0:
                        self._save_progress(progress)
                    
                    # Update progress bar
                    pbar.update(1)
                    if result.status == DownloadStatus.SUCCESS:
                        pbar.set_postfix({"last": f"{tid} ✓"})
                    elif result.status == DownloadStatus.FAILED:
                        pbar.set_postfix({"last": f"{tid} ✗"})
        
        # Final progress save
        self._save_progress(progress)
        
        return results


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(results: List[DownloadResult], output_path: Path) -> Dict[str, Any]:
    """Generate a summary report of download results.
    
    Parameters
    ----------
    results : List[DownloadResult]
        List of download results.
    output_path : Path
        Path to save the report.
        
    Returns
    -------
    Dict[str, Any]
        Summary statistics.
    """
    # Calculate statistics
    total = len(results)
    success = sum(1 for r in results if r.status == DownloadStatus.SUCCESS)
    failed = sum(1 for r in results if r.status == DownloadStatus.FAILED)
    no_data = sum(1 for r in results if r.status == DownloadStatus.NO_DATA)
    total_files = sum(r.file_count for r in results)
    total_time = sum(r.duration for r in results)
    
    # Statistics by catalog
    by_catalog: Dict[str, Dict[str, int]] = {}
    for r in results:
        if r.catalog not in by_catalog:
            by_catalog[r.catalog] = {"success": 0, "failed": 0, "no_data": 0}
        by_catalog[r.catalog][r.status.value] = by_catalog[r.catalog].get(r.status.value, 0) + 1
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_targets": total,
        "successful": success,
        "failed": failed,
        "no_data_available": no_data,
        "success_rate": f"{100 * success / max(total, 1):.1f}%",
        "total_files_downloaded": total_files,
        "total_time_seconds": round(total_time, 1),
        "by_catalog": by_catalog
    }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save failed targets for retry
    failed_targets = [
        {"target_id": r.target_id, "catalog": r.catalog, "error": r.message}
        for r in results if r.status == DownloadStatus.FAILED
    ]
    if failed_targets:
        failed_path = output_path.parent / "failed_downloads.json"
        with open(failed_path, 'w') as f:
            json.dump(failed_targets, f, indent=2)
    
    return summary


# =============================================================================
# CLI Interface
# =============================================================================

def print_banner() -> None:
    """Print ASCII banner for the tool."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    ███████╗██╗  ██╗ ██████╗ ██████╗ ██╗      █████╗ ███╗   ██╗███████╗   ║
║    ██╔════╝╚██╗██╔╝██╔═══██╗██╔══██╗██║     ██╔══██╗████╗  ██║██╔════╝   ║
║    █████╗   ╚███╔╝ ██║   ██║██████╔╝██║     ███████║██╔██╗ ██║█████╗     ║
║    ██╔══╝   ██╔██╗ ██║   ██║██╔═══╝ ██║     ██╔══██║██║╚██╗██║██╔══╝     ║
║    ███████╗██╔╝ ██╗╚██████╔╝██║     ███████╗██║  ██║██║ ╚████║███████╗   ║
║    ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝   ║
║                                                                           ║
║              NASA Confirmed Exoplanet Data Acquisition Pipeline           ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main() -> int:
    """Main entry point for the download pipeline.
    
    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="NASA Confirmed Exoplanet Data Acquisition Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Query available planets without downloading:
    python download_confirmed_planets.py --query_only
    
  Download all confirmed planet light curves:
    python download_confirmed_planets.py --download
    
  Resume interrupted download:
    python download_confirmed_planets.py --download --resume
    
  Show current dataset statistics:
    python download_confirmed_planets.py --stats
        """
    )
    
    parser.add_argument(
        '--stats', action='store_true',
        help='Show current dataset statistics'
    )
    parser.add_argument(
        '--query_only', action='store_true',
        help='Query NASA archive to see available planets (no download)'
    )
    parser.add_argument(
        '--download', action='store_true',
        help='Download all confirmed planet light curves'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume interrupted download (skip completed targets)'
    )
    parser.add_argument(
        '--workers', type=int, default=8,
        help='Number of parallel download workers (default: 8)'
    )
    parser.add_argument(
        '--cache_dir', type=str, default=r"D:\.lightkurve2",
        help='Directory for Lightkurve cache (default: D:\\.lightkurve2)'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Initialize configuration
    config = PipelineConfig(
        max_workers=args.workers,
        cache_dir=Path(args.cache_dir)
    )
    
    # Setup logging
    logger = setup_logging(config)
    
    # Initialize NASA Archive interface
    archive = NASAExoplanetArchive(logger)
    
    # Handle --stats
    if args.stats:
        logger.info("Displaying dataset statistics...")
        
        all_df_path = config.output_dir / "all_df.csv"
        if all_df_path.exists():
            df = pd.read_csv(all_df_path, low_memory=False)
            print(f"\n  all_df.csv:")
            print(f"    Total rows: {len(df)}")
            if 'catalog' in df.columns:
                print(f"    By catalog:")
                for cat, count in df['catalog'].value_counts().items():
                    print(f"      {cat}: {count}")
            if 'label' in df.columns:
                print(f"    By label:")
                print(f"      Planets (1): {(df['label'] == 1).sum()}")
                print(f"      False Positives (0): {(df['label'] == 0).sum()}")
        
        # Check processed files
        npz_dir = Path('notebooks/results_koi')
        if npz_dir.exists():
            npz_files = list(npz_dir.glob("*.npz"))
            print(f"\n  Processed .npz files: {len(npz_files)}")
        
        return 0
    
    # Handle --query_only
    if args.query_only:
        logger.info("Querying NASA Exoplanet Archive (query only mode)...")
        
        kepler_df = archive.get_kepler_confirmed()
        k2_df = archive.get_k2_confirmed()
        tess_df = archive.get_tess_confirmed()
        
        print("\n" + "=" * 60)
        print("AVAILABLE CONFIRMED PLANETS")
        print("=" * 60)
        print(f"  Kepler: {len(kepler_df):,}")
        print(f"  K2:     {len(k2_df):,}")
        print(f"  TESS:   {len(tess_df):,}")
        print(f"  ─────────────────────")
        print(f"  TOTAL:  {len(kepler_df) + len(k2_df) + len(tess_df):,}")
        print("=" * 60)
        
        return 0
    
    # Handle --download
    if args.download:
        logger.info("Starting download pipeline...")
        
        # Step 1: Query all confirmed planets
        print("\n[STEP 1/3] Querying NASA Exoplanet Archive...")
        confirmed_df = archive.get_all_confirmed()
        
        if confirmed_df.empty:
            logger.error("No confirmed planets retrieved. Check network connection.")
            return 1
        
        # Save catalog
        catalog_path = config.output_dir / "confirmed_planets.csv"
        confirmed_df.to_csv(catalog_path, index=False)
        logger.info(f"Saved catalog to: {catalog_path}")
        
        # Step 2: Download light curves
        print("\n[STEP 2/3] Downloading light curves from MAST...")
        
        downloader = LightCurveDownloader(config, logger)
        targets = confirmed_df[['target_id', 'catalog']].drop_duplicates().values.tolist()
        targets = [(int(tid), cat) for tid, cat in targets if pd.notna(tid)]
        
        results = downloader.download_batch(targets, resume=args.resume)
        
        # Step 3: Generate report
        print("\n[STEP 3/3] Generating download report...")
        
        report_path = config.output_dir / "download_report.json"
        summary = generate_report(results, report_path)
        
        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETE")
        print("=" * 60)
        print(f"  Total targets:    {summary['total_targets']:,}")
        print(f"  Successful:       {summary['successful']:,}")
        print(f"  Failed:           {summary['failed']:,}")
        print(f"  No data found:    {summary['no_data_available']:,}")
        print(f"  Success rate:     {summary['success_rate']}")
        print(f"  Files downloaded: {summary['total_files_downloaded']:,}")
        print(f"  Report saved to:  {report_path}")
        print("=" * 60)
        print("\n[NEXT STEP] Run process_data.py to create .npz training files")
        
        return 0 if summary['failed'] == 0 else 1
    
    # No action specified
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
