# Project RxHARM
### Prescriptive Optimization of Heat Associated Risk Mitigation

> **Paper DOI:** `https://doi.org/PLACEHOLDER` *(link will be updated on publication)*

RxHARM is an open-source Python package that computes a globally applicable,
annually updatable **Heat Vulnerability Index (HVI)** and **Heat Risk Index (HRI)**
at **100-meter spatial resolution** from freely available satellite data, then uses
**NSGA-III multi-objective optimization** to prescribe urban heat interventions.

The tool is designed to run in Google Colab notebooks backed by a clean Python
package structure. No commercial data sources or API keys beyond Google Earth
Engine (free tier) are required.

---

## Installation

### Option A — Development mode (clone repo)
```bash
git clone https://github.com/YOUR_USERNAME/rxharm.git
cd rxharm
pip install -e .
```

### Option B — Install directly from GitHub (Colab)
```python
!pip install -q git+https://github.com/YOUR_USERNAME/rxharm.git
```

---

## Quick Start

```python
import rxharm

aoi  = rxharm.AOIHandler("Ahmedabad, India", year=2023)
rxharm.run(aoi)
```

*(Full pipeline implemented across Steps II–VI. See notebooks for guided usage.)*

---

## Notebooks

| Notebook | Purpose | What it produces |
|----------|---------|-----------------|
| `01_HVI_HRI.ipynb` | Compute the Heat Vulnerability Index and Heat Risk Index for any city | GeoTIFF rasters of HVI, HRI, and all 14 sub-indicators |
| `02_Short_Run.ipynb` | Emergency / operational mode — prescribe interventions for an incoming heatwave using GFS forecast data | Ranked intervention map valid for next 72 hours |
| `03_Long_Run.ipynb` | Strategic planning mode — optimize interventions under CMIP6 SSP2-4.5 and SSP5-8.5 climate scenarios | Pareto-optimal intervention portfolios with uncertainty bands |

Open any notebook in Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/rxharm/blob/main/notebooks/01_HVI_HRI.ipynb)

---

## Repository Structure

```
rxharm/
│
├── README.md
├── LICENSE                  MIT
├── requirements.txt         Pinned dependencies
├── setup.py
│
├── rxharm/
│   ├── __init__.py          Package entry point, version, run()
│   ├── config.py            ★ All constants and parameters live here
│   │
│   ├── data/
│   │   ├── indicator_registry.json   Metadata for all 14 satellite indicators
│   │   ├── intervention_library.json 10 interventions with triangular distributions
│   │   ├── beta_coefficients.csv     Köppen-Geiger mortality response coefficients
│   │   └── cdr_lookup.csv            Crude death rates for 30+ countries
│   │
│   ├── aoi/           Area-of-Interest parsing and spatial decomposition
│   ├── fetch/         GEE data fetching (hazard, exposure, sensitivity, AC)
│   ├── index/         HVI and HRI computation (normalizer, weighter, composer)
│   ├── seasonal/      Hottest-period detection from LST climatology
│   ├── risk/          ERA5 context and GFS short-run forecast ingestion
│   ├── scenarios/     CMIP6 long-run scenario downscaling
│   ├── interventions/ Library loader and feasibility masking
│   ├── optimize/      NSGA-III problem definition and runner
│   ├── uncertainty/   Monte Carlo, Bayesian calibration, Morris screening
│   ├── spatial/       Spatial prescription output writer
│   └── viz/           Maps, charts, and export helpers
│
├── notebooks/
│   ├── 01_HVI_HRI.ipynb
│   ├── 02_Short_Run.ipynb
│   └── 03_Long_Run.ipynb
│
└── tests/
    ├── test_config.py        ← Runs immediately after Step I (fully implemented)
    ├── test_hvi.py
    ├── test_objectives.py
    ├── test_interventions.py
    └── test_decomposer.py
```

---

## Running the Tests

```bash
# Run all Step-I tests (should pass immediately after installation)
pytest tests/test_config.py -v

# Run full test suite (later steps required)
pytest tests/ -v
```

---

## Verifying Installation

After `pip install -e .`, run these four commands in order. All must succeed:

```bash
# 1. Install in development mode
pip install -e .

# 2. Verify import works
python -c "import rxharm; print(rxharm.__version__)"

# 3. Run the config tests
pytest tests/test_config.py -v

# 4. Verify all modules are importable (stubs only, no GEE needed)
python -c "
from rxharm.aoi.handler import AOIHandler
from rxharm.aoi.decomposer import ZoneDecomposer
from rxharm.fetch.hazard import HazardFetcher
from rxharm.index.hvi import HVIEngine
from rxharm.optimize.objectives import ObjectiveFunctions
from rxharm.optimize.problem import HeatInterventionProblem
print('All module imports successful')
"
```

---

## Adding New Indicators

RxHARM is designed to be extensible. To register a new indicator:

1. Add an entry to `rxharm/data/indicator_registry.json` following the existing schema.
2. Add the indicator key and weight to the appropriate `*_WEIGHTS` dictionary in `rxharm/config.py`.
3. Implement the fetcher method in the relevant `rxharm/fetch/*.py` module.
4. Re-run `pytest tests/test_config.py -v` to verify weight sums still equal 1.0.

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request
so the proposed change can be discussed. Code must follow the style rules in
`CONTRIBUTING.md` (to be added) — most importantly: no magic numbers outside
`config.py`, type hints on all functions, and docstrings on every class and method.

---

## Citation

If you use RxHARM in published research, please cite:

```bibtex
@software{rxharm2024,
  author    = {PLACEHOLDER},
  title     = {{Project RxHARM}: Prescriptive Optimization of Heat Associated Risk Mitigation},
  year      = {2024},
  url       = {https://github.com/YOUR_USERNAME/rxharm},
  note      = {Version 0.1.0}
}
```

---

## License

MIT © 2024 — see `LICENSE` for details.
