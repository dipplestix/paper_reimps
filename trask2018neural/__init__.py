"""Reproduction code for the Trask et al. (2018) neural extrapolation failure experiment."""

from .identity_failure import IdentityMLP, run_experiment, ExperimentConfig

__all__ = ["IdentityMLP", "run_experiment", "ExperimentConfig"]
