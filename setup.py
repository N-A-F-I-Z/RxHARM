"""
setup.py
========
Project RxHARM — Prescriptive Optimization of Heat Associated Risk Mitigation

Allows:
    pip install -e .                                              (development)
    pip install git+https://github.com/YOUR_USERNAME/rxharm.git  (end-users)

The package_data section ensures that JSON and CSV files inside rxharm/data/
are included in the installed package and accessible at runtime via
importlib.resources or pkg_resources.
"""

from setuptools import setup, find_packages
import os

# ── Read long description from README ─────────────────────────────────────────
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

# ── Read pinned requirements ───────────────────────────────────────────────────
with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as fh:
    # Strip comment lines and blank lines
    install_requires = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    # ── Identity ───────────────────────────────────────────────────────────────
    name="rxharm",
    version="0.1.0",
    description=(
        "Prescriptive Optimization of Heat Associated Risk Mitigation — "
        "globally applicable Heat Vulnerability and Risk Index at 100m resolution "
        "with NSGA-III multi-objective intervention optimization."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/rxharm",
    author="Project RxHARM Contributors",
    author_email="placeholder@example.com",
    license="MIT",

    # ── Package discovery ──────────────────────────────────────────────────────
    packages=find_packages(exclude=["tests*", "notebooks*"]),

    # ── Static data files bundled with the package ─────────────────────────────
    # REASON: JSON registries and CSV lookup tables must be accessible at runtime
    # regardless of how/where the package is installed. package_data ensures they
    # are copied into the installed egg/wheel alongside the Python modules.
    package_data={
        "rxharm": [
            "data/*.json",   # indicator_registry.json, intervention_library.json
            "data/*.csv",    # beta_coefficients.csv, cdr_lookup.csv
        ]
    },
    include_package_data=True,

    # ── Dependencies ───────────────────────────────────────────────────────────
    install_requires=install_requires,
    python_requires=">=3.9",

    # ── PyPI classifiers ───────────────────────────────────────────────────────
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],

    # ── Entry points (CLI placeholder for future use) ─────────────────────────
    entry_points={
        "console_scripts": [
            # rxharm-run will be wired up in Step VI
            # "rxharm-run=rxharm.cli:main",
        ]
    },
)
