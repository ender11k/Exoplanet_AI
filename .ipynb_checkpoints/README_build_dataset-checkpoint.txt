How to build a 1D-CNN dataset from Kepler light curves (modern Lightkurve API)

1) Install dependencies (in a fresh environment is recommended):
   pip install lightkurve==2.* astropy numpy pandas

2) Put your catalog CSV (e.g., all_df.csv) on disk. The script expects columns:
   - target_id: Kepler ID (KIC)
   - label: binary label (0/1)
   Optional:
   - period: transit period [days]
   - epoch: transit epoch [BKJD days]

3) Run the script (internet required to fetch light curves):
   python build_kepler_cnn_dataset.py --catalog /path/to/all_df.csv --outdir dataset --max_targets 500

4) Outputs:
   - dataset/global/<KIC>.npy     (global time-domain resampled series)
   - dataset/folded/<KIC>.npy     (phase-folded resampled series, if period/epoch available)
   - dataset/manifest.csv         (index with flags and labels)

5) Training hint:
   - Use class weights to address imbalance (many 0s vs fewer 1s).
   - Consider combining 'global' and 'folded' views (like Astronet) for best results.
