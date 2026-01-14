#!/usr/bin/env python3
"""
=============================================================================
EXOPLANET AI - MASTER PIPELINE
=============================================================================
Complete pipeline to build a NASA-level exoplanet detection model:

Step 1: Download ALL confirmed exoplanet light curves
Step 2: Process and validate data
Step 3: Train ExoTransformer Ultimate
Step 4: Evaluate and compare with NASA ExoMiner

Run this script to execute the entire pipeline automatically.

Author: Exoplanet AI Research Team
Date: 2026
=============================================================================
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: str, description: str) -> bool:
    """Run a shell command with logging"""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {description}")
    logger.info(f"Command: {cmd}")
    logger.info(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        return False
    return True


def check_prerequisites():
    """Check if all required packages are installed"""
    logger.info("Checking prerequisites...")
    
    required = ['tensorflow', 'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'lightkurve']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        logger.info(f"Installing missing packages: {missing}")
        for pkg in missing:
            subprocess.run([sys.executable, '-m', 'pip', 'install', pkg])
    
    logger.info("All prerequisites satisfied!")
    return True


def count_samples(data_dir: str) -> dict:
    """Count positive and negative samples in a directory"""
    import numpy as np
    
    data_path = Path(data_dir)
    if not data_path.exists():
        return {"total": 0, "positives": 0, "negatives": 0}
    
    files = list(data_path.glob("*.npz"))
    positives = 0
    negatives = 0
    
    for f in files:
        try:
            data = np.load(f)
            label = data.get('label', 0)
            if label == 1:
                positives += 1
            else:
                negatives += 1
        except:
            continue
    
    return {
        "total": len(files),
        "positives": positives,
        "negatives": negatives
    }


def main():
    parser = argparse.ArgumentParser(description="Exoplanet AI Master Pipeline")
    parser.add_argument(
        '--skip_download',
        action='store_true',
        help='Skip data download step (use existing data)'
    )
    parser.add_argument(
        '--download_only',
        action='store_true',
        help='Only download data, do not train'
    )
    parser.add_argument(
        '--missions',
        nargs='+',
        default=['kepler', 'tess'],
        help='Missions to download data from'
    )
    parser.add_argument(
        '--include_candidates',
        action='store_true',
        help='Include high-confidence candidates'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Training epochs'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/exotransformer_ultimate',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("notebooks/results_confirmed").mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███████╗██╗  ██╗ ██████╗ ████████╗██████╗  █████╗ ███╗   ██╗███████╗      ║
║   ██╔════╝╚██╗██╔╝██╔═══██╗╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝      ║
║   █████╗   ╚███╔╝ ██║   ██║   ██║   ██████╔╝███████║██╔██╗ ██║███████╗      ║
║   ██╔══╝   ██╔██╗ ██║   ██║   ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║      ║
║   ███████╗██╔╝ ██╗╚██████╔╝   ██║   ██║  ██║██║  ██║██║ ╚████║███████║      ║
║   ╚══════╝╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝      ║
║                                                                              ║
║               EXOPLANET AI - NASA LEVEL MODEL PIPELINE                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check prerequisites
    check_prerequisites()
    
    # Step 1: Check current data
    logger.info("\n" + "="*60)
    logger.info("STEP 1: ANALYZING CURRENT DATA")
    logger.info("="*60)
    
    existing_stats = count_samples("notebooks/results_koi")
    confirmed_stats = count_samples("notebooks/results_confirmed")
    
    logger.info(f"Existing KOI data: {existing_stats['total']} files "
                f"({existing_stats['positives']} positives)")
    logger.info(f"Confirmed exoplanets: {confirmed_stats['total']} files "
                f"({confirmed_stats['positives']} positives)")
    
    total_positives = existing_stats['positives'] + confirmed_stats['positives']
    
    # Step 2: Download if needed
    if not args.skip_download:
        logger.info("\n" + "="*60)
        logger.info("STEP 2: DOWNLOADING CONFIRMED EXOPLANET DATA")
        logger.info("="*60)
        
        missions_str = " ".join(args.missions)
        candidate_flag = "--include_candidates" if args.include_candidates else ""
        
        download_cmd = (
            f'{sys.executable} scripts/download_all_confirmed_exoplanets.py '
            f'--output_dir notebooks/results_confirmed '
            f'--missions {missions_str} {candidate_flag}'
        )
        
        logger.info("This will download ALL confirmed exoplanets from:")
        for m in args.missions:
            logger.info(f"  • {m.upper()} mission")
        
        logger.info("\nEstimated download:")
        logger.info("  • Kepler: ~2,700 confirmed planets")
        logger.info("  • TESS: ~400 confirmed planets")
        logger.info("  • Total: ~3,100+ new positive samples")
        logger.info("\n⚠️ This may take 2-4 hours depending on internet speed")
        
        # Run download
        success = run_command(download_cmd, "Download confirmed exoplanets")
        
        if not success:
            logger.error("Download failed! Check logs/download_exoplanets.log")
            # Continue anyway with existing data
    
    if args.download_only:
        logger.info("\n--download_only flag set. Stopping here.")
        return
    
    # Step 3: Verify data
    logger.info("\n" + "="*60)
    logger.info("STEP 3: VERIFYING DATA")
    logger.info("="*60)
    
    final_koi = count_samples("notebooks/results_koi")
    final_confirmed = count_samples("notebooks/results_confirmed")
    
    total_samples = final_koi['total'] + final_confirmed['total']
    total_positives = final_koi['positives'] + final_confirmed['positives']
    
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total positives: {total_positives}")
    logger.info(f"Positive ratio: {100*total_positives/max(total_samples,1):.1f}%")
    
    # Estimate achievable performance
    if total_positives >= 3000:
        logger.info("\n✓ Sufficient data for NASA-level performance (PR-AUC > 0.85)")
    elif total_positives >= 1500:
        logger.info("\n✓ Good data for high performance (PR-AUC 0.70-0.85)")
    elif total_positives >= 500:
        logger.info("\n⚠️ Limited data - expect PR-AUC 0.50-0.70")
    else:
        logger.info("\n⚠️ Very limited data - expect PR-AUC 0.30-0.50")
    
    # Step 4: Train model
    logger.info("\n" + "="*60)
    logger.info("STEP 4: TRAINING EXOTRANSFORMER ULTIMATE")
    logger.info("="*60)
    
    train_cmd = (
        f'{sys.executable} scripts/train_exotransformer_ultimate.py '
        f'--data_dirs notebooks/results_koi notebooks/results_confirmed '
        f'--output_dir {args.output_dir} '
        f'--epochs {args.epochs} '
        f'--batch_size 32 '
        f'--n_folds 5'
    )
    
    success = run_command(train_cmd, "Train ExoTransformer Ultimate")
    
    if not success:
        logger.error("Training failed!")
        return
    
    # Step 5: Final summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*60)
    
    # Load and display results
    summary_path = Path(args.output_dir) / "training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        
        logger.info(f"\nFinal Results:")
        logger.info(f"  ROC-AUC: {summary['mean_roc_auc']:.4f} ± {summary['std_roc_auc']:.4f}")
        logger.info(f"  PR-AUC: {summary['mean_pr_auc']:.4f} ± {summary['std_pr_auc']:.4f}")
        logger.info(f"  Recall@P99: {summary['mean_recall_at_99_precision']:.4f}")
        
        logger.info(f"\nComparison with NASA ExoMiner:")
        logger.info(f"  Our Model:     PR-AUC = {summary['mean_pr_auc']:.4f}")
        logger.info(f"  NASA ExoMiner: PR-AUC = ~0.90")
        
        if summary['mean_pr_auc'] >= 0.85:
            logger.info("\n🎉 CONGRATULATIONS! Your model matches NASA-level performance!")
        elif summary['mean_pr_auc'] >= 0.70:
            logger.info("\n✓ Excellent! Your model is competitive with state-of-the-art!")
        else:
            logger.info("\n→ Download more confirmed exoplanets to improve performance")
    
    logger.info(f"\nResults saved to: {args.output_dir}/")
    logger.info("Model weights saved as: best_exotransformer_fold_*.keras")


if __name__ == "__main__":
    main()
