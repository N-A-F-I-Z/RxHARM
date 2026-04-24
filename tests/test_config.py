"""
tests/test_config.py
====================
Tests for rxharm/config.py internal consistency.

These tests verify the configuration is self-consistent before any
data fetching or computation is attempted.

All tests in this file must pass immediately after Step I installation.
They do not require GEE authentication, network access, or any external data.

Run with:
    pytest tests/test_config.py -v
"""

import json
import os
import pytest
import rxharm.config as cfg


# ── Weight Consistency Tests ──────────────────────────────────────────────────

class TestWeightConsistency:
    """All weight dictionaries must sum to exactly 1.0."""

    def test_hazard_weights_sum_to_one(self):
        total = sum(cfg.HAZARD_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-10, (
            f"HAZARD_WEIGHTS sum to {total}, expected 1.0. "
            f"Values: {cfg.HAZARD_WEIGHTS}"
        )

    def test_hazard_weights_all_positive(self):
        for key, val in cfg.HAZARD_WEIGHTS.items():
            assert val > 0, f"HAZARD_WEIGHTS['{key}'] = {val} must be positive"

    def test_exposure_weights_sum_to_one(self):
        total = sum(cfg.EXPOSURE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-10, (
            f"EXPOSURE_WEIGHTS sum to {total}, expected 1.0. "
            f"Values: {cfg.EXPOSURE_WEIGHTS}"
        )

    def test_exposure_weights_all_positive(self):
        for key, val in cfg.EXPOSURE_WEIGHTS.items():
            assert val > 0, f"EXPOSURE_WEIGHTS['{key}'] = {val} must be positive"

    def test_sensitivity_weights_sum_to_one(self):
        total = sum(cfg.SENSITIVITY_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-10, (
            f"SENSITIVITY_WEIGHTS sum to {total}, expected 1.0. "
            f"Values: {cfg.SENSITIVITY_WEIGHTS}"
        )

    def test_sensitivity_weights_all_positive(self):
        for key, val in cfg.SENSITIVITY_WEIGHTS.items():
            assert val > 0, f"SENSITIVITY_WEIGHTS['{key}'] = {val} must be positive"

    def test_adaptive_capacity_weights_sum_to_one(self):
        total = sum(cfg.ADAPTIVE_CAPACITY_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-10, (
            f"ADAPTIVE_CAPACITY_WEIGHTS sum to {total}, expected 1.0. "
            f"Values: {cfg.ADAPTIVE_CAPACITY_WEIGHTS}"
        )

    def test_adaptive_capacity_weights_all_positive(self):
        for key, val in cfg.ADAPTIVE_CAPACITY_WEIGHTS.items():
            assert val > 0, (
                f"ADAPTIVE_CAPACITY_WEIGHTS['{key}'] = {val} must be positive"
            )

    def test_all_four_sub_indices_have_weights(self):
        """All four sub-indices must have at least one indicator weight."""
        assert len(cfg.HAZARD_WEIGHTS) >= 1
        assert len(cfg.EXPOSURE_WEIGHTS) >= 1
        assert len(cfg.SENSITIVITY_WEIGHTS) >= 1
        assert len(cfg.ADAPTIVE_CAPACITY_WEIGHTS) >= 1


# ── Indicator Registry Tests ──────────────────────────────────────────────────

class TestIndicatorRegistryLoads:
    """Verify the JSON files load and have required fields."""

    def test_indicator_registry_file_exists(self):
        assert os.path.isfile(cfg.INDICATOR_REGISTRY_PATH), (
            f"indicator_registry.json not found at: {cfg.INDICATOR_REGISTRY_PATH}"
        )

    def test_indicator_registry_loads(self):
        with open(cfg.INDICATOR_REGISTRY_PATH) as f:
            registry = json.load(f)
        assert "indicators" in registry, "Registry must have 'indicators' key"
        assert len(registry["indicators"]) == 14, (
            f"Expected 14 indicators, found {len(registry['indicators'])}. "
            f"Keys: {list(registry['indicators'].keys())}"
        )

    def test_all_indicators_have_required_fields(self):
        required_fields = [
            "name", "sub_index", "direction", "native_resolution_m",
            "temporal_tier", "automatable", "status", "citation"
        ]
        with open(cfg.INDICATOR_REGISTRY_PATH) as f:
            registry = json.load(f)
        for key, indicator in registry["indicators"].items():
            for field in required_fields:
                assert field in indicator, (
                    f"Indicator '{key}' is missing required field '{field}'"
                )

    def test_indicator_directions_are_valid(self):
        """direction must be 'positive' or 'negative' for every indicator."""
        with open(cfg.INDICATOR_REGISTRY_PATH) as f:
            registry = json.load(f)
        valid_directions = {"positive", "negative"}
        for key, indicator in registry["indicators"].items():
            assert indicator["direction"] in valid_directions, (
                f"Indicator '{key}' has invalid direction: '{indicator['direction']}'. "
                f"Must be one of {valid_directions}."
            )

    def test_indicator_sub_indices_are_valid(self):
        """sub_index must be one of the four known sub-indices."""
        with open(cfg.INDICATOR_REGISTRY_PATH) as f:
            registry = json.load(f)
        valid_sub_indices = {
            "hazard", "exposure", "sensitivity", "adaptive_capacity"
        }
        for key, indicator in registry["indicators"].items():
            assert indicator["sub_index"] in valid_sub_indices, (
                f"Indicator '{key}' has invalid sub_index: '{indicator['sub_index']}'"
            )

    def test_indicator_registry_keys_match_weight_dicts(self):
        """Every key in weight dicts must appear in the registry."""
        with open(cfg.INDICATOR_REGISTRY_PATH) as f:
            registry = json.load(f)
        registry_keys = set(registry["indicators"].keys())

        all_weight_keys = (
            set(cfg.HAZARD_WEIGHTS)
            | set(cfg.EXPOSURE_WEIGHTS)
            | set(cfg.SENSITIVITY_WEIGHTS)
            | set(cfg.ADAPTIVE_CAPACITY_WEIGHTS)
        )
        for key in all_weight_keys:
            assert key in registry_keys, (
                f"Weight key '{key}' not found in indicator_registry.json. "
                f"Add it to the registry or remove it from config weights."
            )


# ── Intervention Library Tests ────────────────────────────────────────────────

class TestInterventionLibraryLoads:
    """Verify the intervention library loads and has correct structure."""

    def test_intervention_library_file_exists(self):
        assert os.path.isfile(cfg.INTERVENTION_LIBRARY_PATH), (
            f"intervention_library.json not found at: {cfg.INTERVENTION_LIBRARY_PATH}"
        )

    def test_intervention_library_loads(self):
        with open(cfg.INTERVENTION_LIBRARY_PATH) as f:
            library = json.load(f)
        assert "short_run" in library, "Library must have 'short_run' key"
        assert "long_run" in library, "Library must have 'long_run' key"
        assert len(library["short_run"]) == 5, (
            f"Expected 5 short-run interventions, found {len(library['short_run'])}"
        )
        assert len(library["long_run"]) == 5, (
            f"Expected 5 long-run interventions, found {len(library['long_run'])}"
        )

    def test_all_interventions_have_triangular_cost_distributions(self):
        """cost_per_unit_usd must be [min, mode, max], not a scalar."""
        with open(cfg.INTERVENTION_LIBRARY_PATH) as f:
            library = json.load(f)
        for scope in ["short_run", "long_run"]:
            for key, interv in library[scope].items():
                group_c = interv.get("group_C_cost", {})
                cost = group_c.get("cost_per_unit_usd")
                assert isinstance(cost, list) and len(cost) == 3, (
                    f"Intervention '{key}' cost_per_unit_usd must be "
                    f"[min, mode, max] but got: {cost}"
                )
                min_v, mode_v, max_v = cost
                assert min_v <= mode_v <= max_v, (
                    f"Intervention '{key}' cost triangle must have "
                    f"min <= mode <= max, got {cost}"
                )

    def test_all_interventions_have_required_groups(self):
        """Every intervention must have groups A through F."""
        required_groups = [
            "group_A_spatial_effects", "group_B_hvi_linkage",
            "group_C_cost", "group_D_cobenefits",
            "group_E_feasibility", "group_F_uncertainty"
        ]
        with open(cfg.INTERVENTION_LIBRARY_PATH) as f:
            library = json.load(f)
        for scope in ["short_run", "long_run"]:
            for key, interv in library[scope].items():
                for group in required_groups:
                    assert group in interv, (
                        f"Intervention '{key}' missing required group '{group}'"
                    )

    def test_intervention_scopes_match_keys(self):
        """scope field must match the containing section."""
        with open(cfg.INTERVENTION_LIBRARY_PATH) as f:
            library = json.load(f)
        for scope in ["short_run", "long_run"]:
            for key, interv in library[scope].items():
                assert interv["scope"] == scope, (
                    f"Intervention '{key}' has scope='{interv['scope']}' "
                    f"but is in section '{scope}'"
                )


# ── Beta Coefficients Tests ───────────────────────────────────────────────────

class TestBetaCoefficients:
    """Verify beta CSV loads and contains all Köppen-Geiger zones."""

    def test_beta_csv_file_exists(self):
        assert os.path.isfile(cfg.BETA_COEFFICIENTS_PATH), (
            f"beta_coefficients.csv not found at: {cfg.BETA_COEFFICIENTS_PATH}"
        )

    def test_beta_csv_loads(self):
        import pandas as pd
        df = pd.read_csv(cfg.BETA_COEFFICIENTS_PATH)
        assert set(df["climate_zone"]) == {"A", "B", "C", "D", "E"}, (
            f"Beta CSV must cover all 5 Köppen-Geiger zones (A-E). "
            f"Found: {set(df['climate_zone'])}"
        )

    def test_all_beta_values_are_positive(self):
        import pandas as pd
        df = pd.read_csv(cfg.BETA_COEFFICIENTS_PATH)
        assert all(df["beta_per_degC"] > 0), (
            "All beta values must be positive. "
            f"Found: {df[df['beta_per_degC'] <= 0]}"
        )

    def test_beta_values_are_plausible(self):
        """Beta values should be in the range [0.001, 0.02] per degree C."""
        import pandas as pd
        df = pd.read_csv(cfg.BETA_COEFFICIENTS_PATH)
        assert all(df["beta_per_degC"] < 0.02), (
            "Beta values > 0.02/°C seem unreasonably high for literature values."
        )
        assert all(df["beta_per_degC"] > 0.001), (
            "Beta values < 0.001/°C seem unreasonably low."
        )

    def test_beta_config_keys_match_csv(self):
        """BETA_BY_CLIMATE_ZONE in config must cover all zones in the CSV."""
        import pandas as pd
        df = pd.read_csv(cfg.BETA_COEFFICIENTS_PATH)
        csv_zones = set(df["climate_zone"])
        config_zones = set(cfg.BETA_BY_CLIMATE_ZONE.keys())
        assert csv_zones == config_zones, (
            f"Config zones {config_zones} do not match CSV zones {csv_zones}"
        )

    def test_cdr_lookup_file_exists(self):
        assert os.path.isfile(cfg.CDR_LOOKUP_PATH), (
            f"cdr_lookup.csv not found at: {cfg.CDR_LOOKUP_PATH}"
        )

    def test_cdr_lookup_loads(self):
        import pandas as pd
        df = pd.read_csv(cfg.CDR_LOOKUP_PATH, comment="#")
        assert "country_iso3" in df.columns, "CDR table must have 'country_iso3' column"
        assert "cdr_per_person_per_year" in df.columns, (
            "CDR table must have 'cdr_per_person_per_year' column"
        )

    def test_cdr_lookup_has_minimum_countries(self):
        import pandas as pd
        df = pd.read_csv(cfg.CDR_LOOKUP_PATH, comment="#")
        assert len(df) >= 20, (
            f"CDR lookup must have at least 20 countries, found {len(df)}"
        )

    def test_cdr_values_are_plausible(self):
        """CDR should be between 0 and 0.05 deaths/person/year."""
        import pandas as pd
        df = pd.read_csv(cfg.CDR_LOOKUP_PATH, comment="#")
        assert all(df["cdr_per_person_per_year"] > 0), (
            "All CDR values must be positive (deaths/person/year)"
        )
        assert all(df["cdr_per_person_per_year"] < 0.05), (
            "CDR values > 0.05 deaths/person/year seem unreasonably high. "
            "Expected range: 0.003–0.015 for most countries."
        )


# ── Configuration Threshold Tests ─────────────────────────────────────────────

class TestThresholds:
    """Basic sanity checks on configuration constant values."""

    def test_ac_floor_is_positive(self):
        assert cfg.AC_FLOOR > 0, (
            f"AC_FLOOR must be > 0 to prevent division by zero. Got: {cfg.AC_FLOOR}"
        )

    def test_ac_floor_is_small(self):
        assert cfg.AC_FLOOR < 0.1, (
            f"AC_FLOOR = {cfg.AC_FLOOR} is unusually large. "
            "Expected a small value like 0.01 to avoid distorting AC scores."
        )

    def test_cell_size_is_100(self):
        assert cfg.CELL_SIZE_M == 100, (
            f"CELL_SIZE_M must be 100 for WorldPop compatibility. Got: {cfg.CELL_SIZE_M}"
        )

    def test_gee_scale_matches_cell_size(self):
        assert cfg.GEE_SCALE == cfg.CELL_SIZE_M, (
            f"GEE_SCALE ({cfg.GEE_SCALE}) must equal CELL_SIZE_M ({cfg.CELL_SIZE_M})"
        )

    def test_lambda_range_contains_default(self):
        lo, hi = cfg.LAMBDA_HVI_RANGE
        assert lo <= cfg.LAMBDA_HVI_DEFAULT <= hi, (
            f"LAMBDA_HVI_DEFAULT ({cfg.LAMBDA_HVI_DEFAULT}) must be within "
            f"LAMBDA_HVI_RANGE ({cfg.LAMBDA_HVI_RANGE})"
        )

    def test_moore_weights_are_ordered(self):
        assert cfg.MOORE_WEIGHT_PRIMARY >= cfg.MOORE_WEIGHT_FACE, (
            "Primary cell weight must be >= face-adjacent weight"
        )
        assert cfg.MOORE_WEIGHT_FACE >= cfg.MOORE_WEIGHT_DIAGONAL, (
            "Face-adjacent weight must be >= diagonal weight"
        )

    def test_moore_weight_primary_is_one(self):
        assert cfg.MOORE_WEIGHT_PRIMARY == 1.0, (
            f"Primary Moore weight must be 1.0, got {cfg.MOORE_WEIGHT_PRIMARY}"
        )

    def test_nsga3_pop_sizes_are_positive(self):
        assert cfg.NSGA3_POP_SIZE_SR > 0
        assert cfg.NSGA3_POP_SIZE_LR > 0

    def test_nsga3_n_gen_are_positive(self):
        assert cfg.NSGA3_N_GEN_SR > 0
        assert cfg.NSGA3_N_GEN_LR > 0

    def test_long_run_has_more_generations_than_short_run(self):
        assert cfg.NSGA3_N_GEN_LR >= cfg.NSGA3_N_GEN_SR, (
            "Long-run optimization should use at least as many generations as short-run"
        )

    def test_epsilon_lengths_match_objectives(self):
        assert len(cfg.EPSILON_SR) == 3, (
            f"EPSILON_SR must have 3 values (one per short-run objective), "
            f"got {len(cfg.EPSILON_SR)}"
        )
        assert len(cfg.EPSILON_LR) == 5, (
            f"EPSILON_LR must have 5 values (one per long-run objective), "
            f"got {len(cfg.EPSILON_LR)}"
        )

    def test_random_seeds_is_nonempty_list(self):
        assert isinstance(cfg.RANDOM_SEEDS, list)
        assert len(cfg.RANDOM_SEEDS) >= 1, "Must have at least one random seed"

    def test_mmt_percentile_is_plausible(self):
        assert 50 <= cfg.MMT_PERCENTILE <= 95, (
            f"MMT_PERCENTILE = {cfg.MMT_PERCENTILE}. "
            "Expected range: 50–95 (typical literature: 70–85)"
        )

    def test_n_hottest_months_is_valid(self):
        assert 1 <= cfg.N_HOTTEST_MONTHS <= 6, (
            f"N_HOTTEST_MONTHS = {cfg.N_HOTTEST_MONTHS}. Expected 1–6."
        )

    def test_weighting_default_in_methods(self):
        assert cfg.WEIGHTING_DEFAULT in cfg.WEIGHTING_METHODS, (
            f"WEIGHTING_DEFAULT '{cfg.WEIGHTING_DEFAULT}' not in "
            f"WEIGHTING_METHODS {cfg.WEIGHTING_METHODS}"
        )

    def test_carbon_price_is_positive(self):
        assert cfg.CARBON_PRICE_USD_PER_TCO2 > 0, (
            f"Carbon price must be > 0, got: {cfg.CARBON_PRICE_USD_PER_TCO2}"
        )

    def test_stormwater_value_is_positive(self):
        assert cfg.STORMWATER_VALUE_USD_PER_M3 > 0


# ── File Paths Tests ──────────────────────────────────────────────────────────

class TestFilePaths:
    """Verify all configured data file paths exist and are readable."""

    def test_package_dir_exists(self):
        assert os.path.isdir(cfg.PACKAGE_DIR), (
            f"PACKAGE_DIR does not exist: {cfg.PACKAGE_DIR}"
        )

    def test_data_dir_exists(self):
        assert os.path.isdir(cfg.DATA_DIR), (
            f"DATA_DIR does not exist: {cfg.DATA_DIR}"
        )

    def test_all_data_files_exist(self):
        paths = {
            "INDICATOR_REGISTRY_PATH": cfg.INDICATOR_REGISTRY_PATH,
            "INTERVENTION_LIBRARY_PATH": cfg.INTERVENTION_LIBRARY_PATH,
            "BETA_COEFFICIENTS_PATH": cfg.BETA_COEFFICIENTS_PATH,
            "CDR_LOOKUP_PATH": cfg.CDR_LOOKUP_PATH,
        }
        for name, path in paths.items():
            assert os.path.isfile(path), (
                f"Data file {name} not found at: {path}"
            )

    def test_output_crs_is_valid_string(self):
        assert isinstance(cfg.OUTPUT_CRS, str)
        assert "EPSG" in cfg.OUTPUT_CRS.upper(), (
            f"OUTPUT_CRS should be an EPSG code string, got: {cfg.OUTPUT_CRS}"
        )
