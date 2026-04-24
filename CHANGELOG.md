# RxHARM Changelog

## [0.1.1] — 2025-04

### Fixed (v0.1.1 Stability + Accuracy Fixes)

- **[Fix 1a]** `SeasonalDetector.detect()` now expands the Landsat composite window
  from `N_HOTTEST_MONTHS` up to `MAX_WINDOW_MONTHS` (new config constant = 5) when
  fewer than `MIN_LANDSAT_SCENES` valid scenes are found. Prevents silent flat HVI
  maps in cloud-prone or data-sparse regions.

- **[Fix 1b]** Added `MAX_WINDOW_MONTHS = 5` to `config.py` Section 6.

- **[Fix 1c]** `HazardFetcher.get_lst()` now checks scene count before reducing.
  If fewer than `MIN_LANDSAT_SCENES` scenes exist, the cloud mask is relaxed to
  keep cloud-adjacent pixels instead of raising a RuntimeError.

- **[Fix 1d]** `fetch_all_indicators()` progress tracking upgraded from simple
  `print()` to a task-level tracker showing elapsed time (seconds) and ✓/✗ status
  per step. `fetch_metadata` now includes `'step_times_s'` dict.

- **[Fix 2a]** `config.GEE_COLLECTIONS['ghs_built_s']` corrected from community
  catalog path to JRC native: `JRC/GHSL/P2023A/GHS_BUILT_S`. Added
  `GHS_BUILT_H_EPOCH = 2018` constant.

- **[Fix 2b]** `ExposureFetcher.get_built_fraction()` rewritten to use
  `ee.Image(prefix/epoch)` not `ee.ImageCollection().mosaic()`. Added new
  `ExposureFetcher.get_building_height()` method for GHS-BUILT-H.

- **[Fix 3a]** Created `rxharm/fetch/worldpop_fetcher.py` — new
  `WorldPopFetcher` class downloads WorldPop Global2 R2025A directly from
  worldpop.org. Fetches only the age-sex bands required for HVI. ISO3 lookup
  via `get_iso3_from_centroid()`.

- **[Fix 3b]** `SensitivityFetcher.get_age_fractions()` now tries WorldPop Global2
  direct download (2015–2030) as primary, GEE WorldPop 2020 as fallback. Returns
  GEE placeholder image when using local data. Added `merge_worldpop_local()` to
  `fetch/__init__.py` to merge local arrays after export.

- **[Fix 4]** `VIIRSDownscaler._stack_covariates()` now enforces canonical
  `COVARIATE_ORDER = ('population', 'ndvi', 'built_frac')` regardless of dict
  key ordering. Prevents RF training on one column order and predicting on another.

- **[Fix 5a]** `normalizer.py` `INDICATOR_DIRECTIONS` — `viirs_dnb` direction
  clarified and confirmed as `'positive'` (high light = more infrastructure = more
  coping capacity, handled naturally by AC denominator).

- **[Fix 5b]** `HRIEngine.__init__()` now accepts an ISO3 country code string (e.g.
  `'IND'`) in addition to a float CDR value. Added `HRIEngine.cdr_from_iso3()` class
  method that looks up from `cdr_lookup.csv`. Accepts `None` → global fallback
  (0.0074). Eliminates manual CDR lookup and wrong-country errors.

- **[Fix 6a]** `fetch_all_indicators()` fetches Dynamic World ONCE, stores result on
  `sensitivity_fetcher._dw_label_mosaic`, and passes it to `HazardFetcher.get_uhi()`
  via the new optional `dw_built_mask` parameter. Eliminates one redundant GEE call.

- **[Fix 6b]** `SensitivityFetcher.fetch_all()` now caches the DW built mask on
  `self._dw_label_mosaic` for sharing.

- **[Fix 6c]** `AdaptiveCapacityFetcher._build_s2_collection()` caches its result on
  `self._s2_col` so `get_ndvi()` and `get_ndwi()` share the same collection object.

- **[Fix 6d]** `AdaptiveCapacityFetcher.get_viirs_dnb_raw()` now calls `.toFloat()`
  before `.clamp()` to prevent uint16 overflow, and handles both `'maximum'` and
  `'avg_rad'` band names across VIIRS collection versions.

- **[Fix 6e]** `SeasonalDetector._cache_path()` now stores cache on Google Drive
  when mounted (persists across Colab sessions), and includes `N_HOTTEST_MONTHS` in
  the filename so changing the config value auto-invalidates the cache.

### Added

- `rxharm/fetch/worldpop_fetcher.py` — `WorldPopFetcher`, `get_iso3_from_centroid()`
- `rxharm/fetch/__init__.py` — `merge_worldpop_local()`
- `rxharm/index/hri.py` — `HRIEngine.cdr_from_iso3()`
- `requirements.txt` — Added `Pillow`, `reverse-geocoder`, `pycountry`

---

## [0.1.0] — Initial Release — 2025-04-17

### Fixed (v0.1.0 Stability Fixes)

- **[Critical — Group 1]** Added `rxharm.fetch.validator.validate_indicator_arrays()` to detect
  silent GEE collection failures. Previously, failed fetches (zero-filled or NaN-filled exports)
  produced plausible-looking but completely wrong HVI maps with no error message.
  Validator is now called automatically at the start of `HVIEngine.compute_all()`.

- **[Critical — Group 2]** Fixed `f5_scenario_robustness` returning negative or NaN values.
  Old denominator was `min_s(AD)` which hit zero or negative from Monte Carlo noise in
  low-population cells. New denominator is `max(|min_AD|, 1.0)` — always positive.
  Also added `np.maximum(0, ad)` clip to prevent physically impossible negative deaths.

- **[Critical — Group 2]** Fixed `f4_cobenefit_efficiency` returning NaN in early NSGA-III
  generations when total allocation cost is zero. Old code divided by `total_cost`; new code
  divides by `budget` (a user-supplied constant), making the metric always well-defined.
  Added final `np.isfinite` guard to prevent optimizer archive corruption.

- **[Critical — Group 3]** Fixed Moore's neighborhood boundary cell corruption.
  `fetch_all_indicators()` now buffers the GEE export region by 5 × `CELL_SIZE_M` (500m) so that
  all Moore neighborhood cells always have valid data rather than nodata fill values.

- **[Critical — Group 3]** Fixed `apply_majority_filter()` incorrectly smoothing boundary cells
  to code 0 (no intervention). Added `np.pad(…, mode='edge')` before filtering so boundary
  cells always have 8 valid neighbors drawn from real data.

- **[Critical — Group 3]** Added nodata sentinel detection in `filter_non_prescribable()`.
  Cells with values of `-9999`, `-9998`, `-32768`, or `65535` are now explicitly marked
  non-prescribable, preventing fill values from corrupting optimization with extreme negatives.

- **[Critical — Group 4]** `BayesianCalibrator.calibrate_lambda()` now emits an explicit
  `!`-banner WARNING when returning the prior without MCC calibration. Previously silent,
  every downstream mortality calculation used an unvalidated default without user awareness.
  Returns new `calibration_status: "PRIOR_ONLY"` field for machine-readable status checks.

- **[Critical — Group 6c]** Added explicit `AC_FLOOR > 0` guard in `HVIEngine._compute_hvi_formula()`.
  A misconfigured `config.py` with `AC_FLOOR = 0` would have caused division by zero in all
  cells where AC normalises to exactly 0. Now raises `ValueError` with actionable message.
  Also added `np.isfinite` guard to prevent any residual Inf from propagating.

### Added (v0.1.0 Usability)

- **[Group 1]** `rxharm.fetch.validator.validate_indicator_arrays(arrays, aoi_name)` —
  validates all 14 indicator bands against expected physical ranges. Raises `ValueError`
  with a 5-point diagnostic if any critical band fails.

- **[Group 1]** `rxharm.fetch.validator.print_validation_report(arrays)` — prints a
  human-readable per-band validation table without raising.

- **[Group 1]** `rxharm.fetch.load_existing_export(tiff_path, …)` — **resume mode**.
  Loads an existing GeoTIFF instead of re-running GEE. Accepts glob patterns, converts
  nodata fills to NaN, and runs validation automatically. Enables iterative HVI analysis
  in ~30 seconds instead of 30 minutes.

- **[Group 5]** `rxharm.optimize.runner.pareto_to_dataframe(result, objective_names)` —
  converts a pymoo Pareto front to a pandas DataFrame with strategic solution flags
  (`is_health_focused`, `is_budget_focused`, `is_balanced`) and a `mortality_reduction_pct`
  convenience column. Enables direct CSV export for papers and non-Python users.

- **[Group 5]** `rxharm.optimize.runner.save_pareto_to_csv(result, filepath)` —
  convenience wrapper that saves the Pareto front DataFrame to CSV.

- **[Group 5]** `rxharm.spatial.prescriber.Prescriber.to_geodataframe(prescription, transform)` —
  converts a prescription code array to a GeoDataFrame (QGIS, ArcGIS, R `sf`-readable).
  Columns: `cell_id`, `geometry`, `intervention_code`, `intervention_name`, `zone_id`, `hri_value`.

- **[Group 5]** `rxharm.spatial.prescriber.Prescriber.save_prescription(…, output_dir)` —
  saves prescriptions in three formats simultaneously: GeoPackage (`.gpkg`), CSV with
  lat/lon coordinates, and a summary statistics CSV.

- **[Group 6a]** `rxharm.config.GEE_COLLECTIONS` — centralised dictionary of all 13 GEE
  collection IDs. All four fetch modules now import from this dict instead of hardcoding.
  To update a collection ID, change one line in `config.py`.

### Changed

- All GEE collection ID strings moved from individual fetch modules to `config.GEE_COLLECTIONS`.
  Affected files: `fetch/hazard.py`, `fetch/exposure.py`, `fetch/sensitivity.py`,
  `fetch/adaptive_capacity.py`.

- Export region in `fetch_all_indicators()` now buffered by 500m (5 × 100m cells).

- `calibrate_mmt()` now prints an informational message confirming the MMT source
  and adds `calibration_status` and `note` fields to the return dict.
