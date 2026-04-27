"""
rxharm/config.py
================
Project RxHARM — Central Configuration File

This file is the single source of truth for every constant, threshold,
weight, and parameter used across the entire RxHARM pipeline.

DESIGN RULE: No magic numbers anywhere else in the codebase.
             Every value lives here with a comment explaining:
               - what it is
               - what unit it is measured in
               - why that value was chosen

Organized into 10 sections:
    1.  Resolution and CRS
    2.  AOI Size Classification Thresholds
    3.  Default Sub-index Indicator Weights
    4.  HVI / HRI Formula Parameters
    5.  Epidemiological Parameters
    6.  Seasonal Detection Parameters
    7.  NSGA-III Optimizer Settings
    8.  Intervention Feasibility Thresholds
    9.  Weighting Method Options
    10. File Paths

A non-programmer should be able to read this file and understand
every value without consulting any other source.
"""

from __future__ import annotations
import os

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Resolution and CRS
# ══════════════════════════════════════════════════════════════════════════════

CELL_SIZE_M  = 100          # output raster resolution in meters
                             # (100m chosen: WorldPop native res, practical GEE limit)

OUTPUT_CRS   = "EPSG:4326"  # WGS84 geographic coordinates
                             # used for all exported rasters and vector outputs

GEE_SCALE    = 100          # Google Earth Engine export / reproject scale in meters
                             # must match CELL_SIZE_M for consistent aggregation

# ── Landsat Collection 2 Level-2 processing constants ────────────────────────
# Applied as: reflectance = raw_DN * LANDSAT_SR_SCALE + LANDSAT_SR_OFFSET
# Source: USGS Landsat Collection 2 Product Guide
LANDSAT_SR_SCALE  = 0.0000275   # scale factor for surface reflectance bands
LANDSAT_SR_OFFSET = -0.2        # additive offset for surface reflectance bands

# Broadband albedo formula coefficients (Liang 2000)
# Source: Liang (2000), Remote Sensing of Environment, 76(1):213-238
# Formula: α = B2*0.356 + B4*0.130 + B5*0.373 + B6*0.085 + B7*0.072 - 0.0018
# Applied to scaled surface reflectance (after LANDSAT_SR_SCALE + OFFSET)
ALBEDO_COEFFS = {
    "B2": 0.356,          # blue band
    "B4": 0.130,          # red band
    "B5": 0.373,          # near-infrared band
    "B6": 0.085,          # SWIR1 band
    "B7": 0.072,          # SWIR2 band
    "intercept": -0.0018,
}

# GHS-BUILT-S available epoch years (for nearest-epoch selection)
# Source: JRC GHS-BUILT-S R2023A — 5-year epochs 1975 to 2025
GHS_BUILT_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]

# FIX 0.1.1: GHS-BUILT-H (building height) — only 2018 epoch available
GHS_BUILT_H_EPOCH = 2018   # the sole available epoch for building heights

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — AOI Size Classification Thresholds
# ══════════════════════════════════════════════════════════════════════════════

# Cells below this count: optimize directly at cell level (fastest path)
MAX_CELLS_DIRECT = 500

# Cells below this count: one level of meso-zone clustering before optimization
MAX_CELLS_MESO = 5_000

# Above MAX_CELLS_MESO: full two-level hierarchical decomposition is triggered.
# The user is warned and asked to confirm before proceeding, because runtime
# can exceed 2 hours for very large AOIs.

# Moore's neighborhood: if an AOI has this many cells or fewer,
# expand the analysis to the 3×3 Moore neighborhood automatically.
# REASON: Tiny AOIs (e.g. a single market square) gain contextual richness
#         from adjacent cells; the optimizer then accounts for spillover cooling.
MOORE_MAX_CELLS = 9

# Spatial weights for Moore's neighborhood objective function aggregation.
# Primary selected cell (center) gets full weight; face-adjacent cells
# (N/S/E/W) get 0.65 weight; diagonal cells (NE/NW/SE/SW) get 0.45.
# Values sourced from spatial autocorrelation decay analysis in urban LST fields.
MOORE_WEIGHT_PRIMARY  = 1.00   # the cell the user selected
MOORE_WEIGHT_FACE     = 0.65   # 4 face-adjacent cells (share an edge)
MOORE_WEIGHT_DIAGONAL = 0.45   # 4 diagonal cells (share only a corner)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Default Sub-index Indicator Weights
# ══════════════════════════════════════════════════════════════════════════════
#
# These are EQUAL-WEIGHT defaults designed for global comparability.
# Equal weighting is preferred when no local calibration data exists,
# because it makes no implicit claim about relative importance.
#
# All weights within each sub-index MUST sum to exactly 1.0.
# The test suite in tests/test_config.py enforces this invariant.
#
# User-customizable weighting (PCA, entropy, CRITIC, manual) is available
# through the rxharm/index/weighter.py module. Change WEIGHTING_DEFAULT
# in Section 9 to activate a different method.

# ── Hazard sub-index (H_s) ────────────────────────────────────────────────────
# "What physically makes this location thermally dangerous?"
HAZARD_WEIGHTS = {
    "lst":    0.50,  # Landsat LST: direct thermal hazard magnitude (dominant driver)
    "albedo": 0.25,  # Landsat albedo (inverted): low albedo = high solar absorption
    "uhi":    0.25,  # UHI intensity: locally-induced amplification above rural baseline
}
# sum = 1.00 ✓

# ── Exposure sub-index (E) ────────────────────────────────────────────────────
# "How many people are physically present and at risk?"
EXPOSURE_WEIGHTS = {
    "population": 0.70,   # WorldPop 100m count: primary exposure metric
    "built_frac": 0.30,   # GHS-BUILT-S: concentrates exposure weight on built surfaces
}
# sum = 1.00 ✓

# ── Sensitivity sub-index (S) ─────────────────────────────────────────────────
# "Who among those exposed is physiologically more susceptible?"
SENSITIVITY_WEIGHTS = {
    "elderly_frac": 0.30,  # WorldPop 65+: reduced thermoregulation capacity
    "child_frac":   0.25,  # WorldPop <5: immature thermoregulation, carer-dependent
    "impervious":   0.25,  # Dynamic World built: no evaporative surface cooling
    "cropland":     0.20,  # Dynamic World crops: mandatory outdoor occupational exposure
}
# sum = 1.00 ✓

# ── Adaptive Capacity sub-index (AC) ─────────────────────────────────────────
# "How well can people and places cope with and recover from heat stress?"
# NOTE: High AC = lower vulnerability. AC appears in the denominator of HVI.
ADAPTIVE_CAPACITY_WEIGHTS = {
    "ndvi":          0.20,  # Sentinel-2: vegetation greenness / ET cooling capacity
    "tree_cover":    0.20,  # Hansen GFW: persistent woody canopy extent
    "canopy_height": 0.20,  # Potapov GLAD: shade quality (tall > short)
    "ndwi":          0.20,  # Sentinel-2: vegetation moisture / ET efficiency
    "viirs_dnb":     0.20,  # VIIRS downscaled: economic/electrification proxy
}
# sum = 1.00 ✓

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HVI / HRI Formula Parameters
# ══════════════════════════════════════════════════════════════════════════════
#
# HVI formula:   HVI_i = (E_i × S_i) / AC_i
#   where E, S, and AC are all normalized to [0, 1] before computation.
#   Rationale: vulnerability arises from high exposure AND sensitivity
#              but is REDUCED by adaptive capacity.
#
# HRI formula:   HRI_i = H_s_i × HVI_i
#   where H_s is the normalized Hazard sub-index.
#   Rationale: risk = the physical hazard acting on the local vulnerability.

# Minimum value to which AC is clipped AFTER normalization.
# REASON: Prevents division-by-zero in HVI = E*S/AC when AC ≈ 0.
#         0.01 is small enough not to distort rankings but avoids Inf/NaN.
AC_FLOOR = 0.01   # dimensionless, fraction of normalized [0,1] scale

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Epidemiological Parameters
# ══════════════════════════════════════════════════════════════════════════════

# ── Beta coefficients by Köppen-Geiger climate zone ───────────────────────────
# Source: Gasparrini et al. (2017), Lancet Planetary Health
# Meaning: fractional increase in all-cause mortality risk per 1°C above
#          the Minimum Mortality Temperature (MMT) for each climate zone.
# Units: fraction per °C   (e.g., 0.0072 means +0.72% mortality per +1°C)
BETA_BY_CLIMATE_ZONE = {
    "A": 0.0072,   # Tropical — highest sensitivity; bodies less adapted to extremes
    "B": 0.0058,   # Arid/semi-arid — heat acclimatization partially offsets
    "C": 0.0041,   # Temperate (Mediterranean, humid subtropical)
    "D": 0.0033,   # Continental — cold-adapted populations, moderate heat response
    "E": 0.0025,   # Polar — edge case, very few study locations
}

# ── Minimum Mortality Temperature (MMT) estimation ────────────────────────────
# The MMT is the temperature at which all-cause mortality is minimized.
# It is location-adaptive: computed as a percentile of the local ERA5 LST
# climatology during the hottest analysis period (defined by N_HOTTEST_MONTHS).
MMT_PERCENTILE = 75   # percentile of ERA5 LST distribution — typical literature range: 70–85

# Standard deviation used in Bayesian MMT estimation (degrees Celsius).
# Represents the uncertainty in the MMT prior:
#   MMT_i ~ Normal(ERA5_p75_i, MMT_SIGMA)
MMT_SIGMA = 3.0   # degrees Celsius

# ── HVI vulnerability modifier (lambda) ───────────────────────────────────────
# Scales how strongly HVI amplifies the baseline mortality response.
# Attributable Deaths formula:
#   AD_i = Pop_i × CDR × AF_i × days
#   AF_i = 1 - exp(-beta_z × deltaT_i × (1 + lambda × HVI_norm_i))
# where deltaT_i = observed_temp - MMT_i
# Source: Khatana et al. (2022), Circulation — spatial vulnerability interaction
LAMBDA_HVI_DEFAULT = 0.50                # central estimate (dimensionless)
LAMBDA_HVI_RANGE   = (0.30, 0.80)        # min/max range used in sensitivity analysis

# ── Economic co-benefit monetization ─────────────────────────────────────────
# Shadow carbon price for co-benefit valuation of interventions.
# Source: IPCC AR6 Chapter 8 (2022), central estimate for 2030.
# Units: USD per metric tonne CO2-equivalent
CARBON_PRICE_USD_PER_TCO2 = 80.0

# Stormwater cost avoidance value (per cubic metre of runoff avoided).
# Source: US EPA green infrastructure cost-benefit data (2023).
# Units: USD per cubic metre
STORMWATER_VALUE_USD_PER_M3 = 2.50

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Seasonal Detection Parameters
# ══════════════════════════════════════════════════════════════════════════════

# Number of hottest calendar months to use as the composite analysis window.
# e.g., 2 means the two hottest months form the Landsat + Sentinel-2 composite.
# REASON: Using only the hottest period maximises signal-to-noise for heat risk.
N_HOTTEST_MONTHS = 2   # months (integer)

# FIX 0.1.1: Maximum number of months the composite window can expand to
# when insufficient Landsat scenes are found in the initial hottest months.
# Setting this to 5 means if 2 months have no data, up to 5 months are used.
MAX_WINDOW_MONTHS = 5   # months (integer)

# Minimum number of valid (cloud-free) Landsat scenes required for a composite.
# If fewer scenes are available, the user receives a warning and is prompted
# to expand the date range or relax the cloud mask.
MIN_LANDSAT_SCENES = 3   # scenes (integer)

# Sentinel-2 maximum scene cloud percentage filter applied before compositing.
# Scenes with more cloud cover than this value are excluded entirely.
S2_MAX_CLOUD_PCT = 30   # percent (0–100)

# SCL (Scene Classification Layer) values to MASK in Sentinel-2 processing
# 3=cloud shadow, 8=cloud med prob, 9=cloud high prob, 10=thin cirrus, 11=snow/ice
# Keep all other SCL classes (vegetation, bare soil, water, etc.)
S2_SCL_MASK_VALUES = [3, 8, 9, 10, 11]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — NSGA-III Optimizer Settings
# ══════════════════════════════════════════════════════════════════════════════

# ── Population sizes ──────────────────────────────────────────────────────────
# Number of candidate solutions maintained per generation.
# REASON: 100 is the recommended minimum for 3-objective problems with
#         NSGA-III (Deb & Jain, 2014). Larger populations improve Pareto
#         coverage but increase per-generation cost.
NSGA3_POP_SIZE_SR = 100   # short-run mode (3 objectives)
NSGA3_POP_SIZE_LR = 100   # long-run mode (5 objectives)

# ── Number of generations ─────────────────────────────────────────────────────
NSGA3_N_GEN_SR = 150   # short-run: fewer generations — time is critical
NSGA3_N_GEN_LR = 300   # long-run: more generations — thoroughness preferred

# ── Monte Carlo samples per objective function evaluation ─────────────────────
# Each objective function call draws MC_SAMPLES from the triangular
# distributions in the intervention library to propagate uncertainty.
# Higher = more accurate expected values but slower per-generation cost.
# REASON: 200 balances statistical stability (~3% CI width) with runtime.
MC_SAMPLES = 200   # samples per evaluation (integer)

# ── Epsilon tolerances per objective ─────────────────────────────────────────
# Controls what counts as a "meaningful improvement" when comparing solutions.
# Solutions within epsilon of each other are treated as equivalent on that objective.
# Prevents convergence stagnation on numerically noisy objectives.
#
# Short-run objectives (3):
#   [mortality_fraction_reduced, cost_fraction_of_budget, equity_gini_coefficient]
EPSILON_SR = [0.005, 0.010, 0.010]

# Long-run objectives (5):
#   [mortality_fraction, cost_fraction, equity_gini, cobenefit_efficiency, robustness]
EPSILON_LR = [0.005, 0.010, 0.010, 0.010, 0.010]

# ── Random seeds for reproducibility ─────────────────────────────────────────
# Run the optimizer with each seed and check that Pareto fronts are consistent.
# Inconsistency across seeds signals insufficient generations or population size.
RANDOM_SEEDS = [42, 123, 456, 789, 1011]   # list of integers

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Intervention Feasibility Thresholds
# ══════════════════════════════════════════════════════════════════════════════

# ── Dynamic World land cover class integer labels ─────────────────────────────
# Source: Brown et al. (2022), Scientific Data; Google/WRI Dynamic World V1
# Used to create binary eligibility masks for each intervention.
DW_LABEL_WATER   = 0
DW_LABEL_TREES   = 1
DW_LABEL_GRASS   = 2
DW_LABEL_FLOODED = 3
DW_LABEL_CROPS   = 4
DW_LABEL_SHRUB   = 5
DW_LABEL_BUILT   = 6
DW_LABEL_BARE    = 7
DW_LABEL_SNOW    = 8

# Cells with a Dynamic World modal class in this set are non-prescribable.
# REASON: Water, snow, and flooded land cannot physically receive interventions.
NON_PRESCRIBABLE_DW_LABELS = {DW_LABEL_WATER, DW_LABEL_SNOW, DW_LABEL_FLOODED}

# Minimum WorldPop population per cell for a cell to be considered prescribable.
# REASON: Intervening in uninhabited cells produces no mortality benefit.
# Units: persons per 100m cell
MIN_POPULATION_THRESHOLD = 1.0   # persons

# ── Intervention-specific feasibility thresholds ──────────────────────────────
# These mirror the group_E_feasibility entries in intervention_library.json.
# Storing them here as well allows Python code to reference them without
# loading the JSON every time.

COOL_ROOF_MIN_BUILT_FRAC   = 0.30   # fraction — cell must be >30% built surface
COOL_ROOF_MAX_BUILT_H_M    = 20.0   # metres — building must be <20m tall (flat/low roof)
TREE_MIN_OPEN_FRAC         = 0.10   # fraction — must have >10% non-built open fraction
BGI_MAX_BUILT_H_M          = 5.0    # metres — blue-green infra is ground-level only
SHADE_MAX_EXISTING_CANOPY_M= 3.0    # metres — only prescribe shade where canopy is <3m
COOLING_CENTER_RADIUS_M    = 300.0  # metres — walking-distance coverage zone
PAVEMENT_MIN_BUILT_FRAC    = 0.40   # fraction — cell must be mostly paved

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Weighting Method Options
# ══════════════════════════════════════════════════════════════════════════════

# String keys that rxharm/index/weighter.py recognises.
# - "equal"   : uniform weights (default; globally comparable)
# - "pca"     : Principal Component Analysis-derived weights
# - "entropy" : Shannon entropy weighting (emphasises discriminating indicators)
# - "critic"  : CRITIC method (correlation-adjusted entropy)
# - "manual"  : user supplies a weights dict directly
WEIGHTING_METHODS = ["equal", "pca", "entropy", "critic", "manual"]
WEIGHTING_DEFAULT = "equal"   # used when user does not specify

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — File Paths
# ══════════════════════════════════════════════════════════════════════════════
#
# These paths resolve relative to the installed package location so they work
# whether the package is installed via `pip install -e .` (development) or
# via `pip install git+...` (production). The DATA_DIR path should be used
# in all modules instead of hardcoded relative paths.

# Absolute path to the rxharm package source directory
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute path to the rxharm/data/ directory
DATA_DIR = os.path.join(PACKAGE_DIR, "data")

# Individual data file paths
INDICATOR_REGISTRY_PATH  = os.path.join(DATA_DIR, "indicator_registry.json")
INTERVENTION_LIBRARY_PATH= os.path.join(DATA_DIR, "intervention_library.json")
BETA_COEFFICIENTS_PATH   = os.path.join(DATA_DIR, "beta_coefficients.csv")
CDR_LOOKUP_PATH          = os.path.join(DATA_DIR, "cdr_lookup.csv")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — GEE Collection Version Registry
# ══════════════════════════════════════════════════════════════════════════════
#
# FIX v0.1.0: Centralizes all GEE collection IDs so that if a collection
# changes its ID or version, only this file needs updating. Previously,
# collection IDs were hardcoded across multiple fetch modules.
#
# UPDATE PROCESS: If a collection ID changes, update it here and run:
#   pytest tests/test_fetch.py -m gee -k "collection_access"
# to verify the new ID is accessible before committing.

GEE_COLLECTIONS: dict = {
    # Landsat Collection 2 Level-2 Surface Reflectance
    "landsat8":          "LANDSAT/LC08/C02/T1_L2",
    "landsat9":          "LANDSAT/LC09/C02/T1_L2",
    # Sentinel-2 Harmonised Surface Reflectance
    "sentinel2":         "COPERNICUS/S2_SR_HARMONIZED",
    # Dynamic World near-real-time land use / land cover
    "dynamic_world":     "GOOGLE/DYNAMICWORLD/V1",
    # WorldPop population (100m grid)
    "worldpop_pop":      "WorldPop/GP/100m/pop",
    "worldpop_agesex":   "WorldPop/GP/100m/pop_age_sex_cons_unadj",
    "worldpop_global2":  "projects/sat-io/open-datasets/WorldPop/Global2",
    # Global Human Settlement Layer
    # FIX 0.1.1: Corrected from community catalog to JRC native catalog.
    # Each epoch is accessed as ee.Image(prefix + '/' + epoch_year).
    # Old (WRONG): 'projects/sat-io/open-datasets/GHS/GHS_BUILT_S'
    "ghs_built_s":       "JRC/GHSL/P2023A/GHS_BUILT_S",  # append /{epoch_year}
    "ghs_built_h":       "JRC/GHSL/P2023A/GHS_BUILT_H",  # append /{epoch_year}
    # Global Forest Watch Hansen tree cover (update version string annually)
    "hansen_gfw":        "UMD/hansen/global_forest_change_2023_v1_11",
    # Potapov canopy height (GLAD / GEDI)
    "potapov_ch":        "projects/sat-io/open-datasets/GLAD/GEDI_V27",
    # VIIRS Day/Night Band (annual composite)
    "viirs_dnb":         "NOAA/VIIRS/DNB/ANNUAL_V22",
    # ERA5-Land monthly reanalysis
    "era5_land":         "ECMWF/ERA5_LAND/MONTHLY_AGGR",
}

# Version string embedded in Hansen collection ID — update when GFW releases new version.
GFW_DATASET_VERSION: str = "global_forest_change_2023_v1_11"

# Community catalog base prefix — if sat-io moves, update this one string.
SAT_IO_PREFIX: str = "projects/sat-io/open-datasets"
