"""
Stress-test scenarios for robustness evaluation (three tiers).

Tier 1 — Signal degradation:
  baseline, gaussian_noise, dropout, low_depth

Tier 2 — Reference / sample mismatch:
  batch_shift, reference_reduction, rare_cell

Tier 3 — Out-of-distribution (distribution shift):
  missing_cell_type, cross_donor, cross_dataset

Each scenario perturbs clean pseudo-bulk data in a controlled manner
so we can measure how uncertainty estimates respond.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Tier 1: Signal degradation ────────────────────────────────────────────

def apply_dropout(bulk: pd.DataFrame, dropout_rate: float = 0.3, seed: int = 42) -> pd.DataFrame:
    """Randomly zero out a fraction of entries."""
    rng = np.random.default_rng(seed)
    mask = rng.random(bulk.shape) > dropout_rate
    result = bulk.copy()
    result.values[~mask] = 0.0
    logger.info("Dropout rate=%.2f: %.1f%% entries zeroed", dropout_rate, 100 * (1 - mask.mean()))
    return result


def apply_gaussian_noise(bulk: pd.DataFrame, noise_std: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """Add Gaussian noise N(0, noise_std * gene_std) to each gene."""
    rng = np.random.default_rng(seed)
    gene_stds = bulk.values.std(axis=0, keepdims=True)
    noise = rng.normal(0, noise_std, size=bulk.shape) * gene_stds
    result = bulk.copy()
    result.values[:] = np.clip(bulk.values + noise, 0, None)
    logger.info("Gaussian noise std=%.2f", noise_std)
    return result


def apply_low_depth(bulk: pd.DataFrame, depth_fraction: float = 0.25, seed: int = 42) -> pd.DataFrame:
    """Binomial down-sampling to simulate low sequencing depth."""
    rng = np.random.default_rng(seed)
    result = bulk.copy()
    counts = result.values.astype(int)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            if counts[i, j] > 0:
                counts[i, j] = rng.binomial(counts[i, j], depth_fraction)
    result.values[:] = counts.astype(float)
    logger.info("Low depth fraction=%.2f", depth_fraction)
    return result


# ── Tier 2: Reference / sample mismatch ────────────────────────────────────

def apply_batch_shift(
    bulk: pd.DataFrame, shift_size: float = 1.0, batch_fraction: float = 0.5, seed: int = 42
) -> pd.DataFrame:
    """Add a mean shift to a random subset of samples (batch effect)."""
    rng = np.random.default_rng(seed)
    n = bulk.shape[0]
    n_shift = max(1, int(n * batch_fraction))
    batch_idx = rng.choice(n, size=n_shift, replace=False)
    result = bulk.copy()
    result.values[batch_idx] = np.clip(result.values[batch_idx] + shift_size, 0, None)
    logger.info("Batch shift size=%.2f on %d/%d samples", shift_size, n_shift, n)
    return result


def reduce_reference_cell_types(
    cell_types: List[str], n_remove: int = 1, seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Remove N cell types from the reference (they stay in pseudo-bulk).

    Returns (reduced_cell_types, removed_cell_types).
    """
    rng = np.random.default_rng(seed)
    n = min(n_remove, len(cell_types) - 2)
    removed = rng.choice(cell_types, size=n, replace=False).tolist()
    kept = [ct for ct in cell_types if ct not in removed]
    logger.info("Reference reduction: removed %s, kept %d types", removed, len(kept))
    return kept, removed


def add_rare_cell_type(
    true_proportions: pd.DataFrame,
    rare_cell_type: str,
    rare_fraction: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Add a synthetic rare cell type at very low proportion.

    The rare type's proportion is sampled from Beta(1, 1/rare_fraction)
    and the existing proportions are scaled down.
    """
    rng = np.random.default_rng(seed)
    result = true_proportions.copy()
    n = len(result)
    rare_props = rng.beta(1, 1 / max(rare_fraction, 1e-6), size=n)
    # Ensure minimum 0.001
    rare_props = np.clip(rare_props, 0.001, 0.1)
    scale = 1 - rare_props
    for col in result.columns:
        result[col] = result[col].values * scale
    result[rare_cell_type] = rare_props
    logger.info("Rare cell type '%s' added at mean fraction %.4f", rare_cell_type, rare_props.mean())
    return result


# ── Tier 3: Out-of-distribution ────────────────────────────────────────────

def apply_missing_cell_type(
    bulk: pd.DataFrame,
    true_proportions: pd.DataFrame,
    n_missing_types: int = 1,
    seed: int = 42,
) -> Dict:
    """Mark N cell types as 'missing' from the reference.

    These cell types are present in the pseudo-bulk but absent from the
    reference signature — a realistic OOD scenario.
    """
    rng = np.random.default_rng(seed)
    cell_types = true_proportions.columns.tolist()
    n = min(n_missing_types, len(cell_types) - 1)
    missing = rng.choice(cell_types, size=n, replace=False).tolist()
    updated = true_proportions.copy()
    # The missing types' proportions are merged into "other"
    updated["other_unknown"] = updated[missing].sum(axis=1)
    updated[missing] = 0.0
    logger.info("Missing cell type scenario: %s treated as unknown", missing)
    return {"bulk": bulk, "true_proportions": updated, "missing_types": missing}


def split_by_donor(
    adata_full: "AnnData",
    donor_column: str,
    reference_donors: List[str],
    test_donors: List[str],
) -> Tuple["AnnData", "AnnData"]:
    """Split AnnData by donor for cross-donor testing.

    Returns (reference_adata, test_adata) from disjoint donor sets.
    """
    adata_ref = adata_full[adata_full.obs[donor_column].isin(reference_donors)].copy()
    adata_test = adata_full[adata_full.obs[donor_column].isin(test_donors)].copy()
    logger.info("Cross-donor split: ref=%d cells (%d donors), test=%d cells (%d donors)",
                adata_ref.shape[0], len(reference_donors), adata_test.shape[0], len(test_donors))
    return adata_ref, adata_test


# ── Batch runners ──────────────────────────────────────────────────────────

TIER1_SCENARIOS = [
    "baseline",
    "gaussian_noise_low", "gaussian_noise_med", "gaussian_noise_high",
    "dropout_low", "dropout_med", "dropout_high",
    "low_depth_25", "low_depth_10",
]

TIER2_SCENARIOS = [
    "batch_shift_small", "batch_shift_large",
    "reference_reduction_1", "reference_reduction_2",
    "rare_cell_low", "rare_cell_ultralow",
]

TIER3_SCENARIOS = [
    "missing_cell_type_1", "missing_cell_type_2",
    "cross_donor", "cross_dataset",
]


def run_stress_tier(
    tier_name: str,
    scenario_names: List[str],
    scenario_params: Dict[str, Dict],
    bulk: pd.DataFrame,
    true_proportions: pd.DataFrame,
    deconvolution_fn: Callable,
    seed: int = 42,
) -> pd.DataFrame:
    """Run a batch of stress scenarios and return a summary DataFrame.

    Parameters
    ----------
    tier_name : str
        e.g. "tier1"
    scenario_names : list of str
    scenario_params : dict of str → dict
        Parameters for each scenario.
    bulk : pd.DataFrame
    true_proportions : pd.DataFrame
    deconvolution_fn : callable
        (perturbed_bulk) -> predicted_proportions DataFrame
    seed : int

    Returns
    -------
    pd.DataFrame
        Summary with columns: tier, scenario, overall_MAE, overall_RMSE,
        overall_Pearson_r.
    """
    from ..evaluation.metrics import compute_accuracy_metrics

    summaries = []
    for i, name in enumerate(scenario_names):
        sc_seed = seed + hash(name) % 10000
        params = scenario_params.get(name, {})
        logger.info("[%s] Scenario: %s (params=%s)", tier_name, name, params)

        if name == "baseline":
            perturbed = bulk.copy()
            y_true = true_proportions.copy()
        elif name.startswith("dropout"):
            perturbed = apply_dropout(bulk, dropout_rate=params.get("dropout_rate", 0.3), seed=sc_seed)
            y_true = true_proportions.copy()
        elif name.startswith("gaussian_noise"):
            perturbed = apply_gaussian_noise(bulk, noise_std=params.get("noise_std", 0.5), seed=sc_seed)
            y_true = true_proportions.copy()
        elif name.startswith("low_depth"):
            perturbed = apply_low_depth(bulk, depth_fraction=params.get("depth_fraction", 0.25), seed=sc_seed)
            y_true = true_proportions.copy()
        elif name.startswith("batch_shift"):
            perturbed = apply_batch_shift(bulk, shift_size=params.get("shift_size", 1.0), seed=sc_seed)
            y_true = true_proportions.copy()
        elif name.startswith("reference_reduction"):
            # This scenario modifies signature, handled at call site
            logger.info("  (reference_reduction handled by caller — skipping in-tier)")
            continue
        elif name.startswith("rare_cell"):
            y_true = add_rare_cell_type(
                true_proportions,
                rare_cell_type="Rare",
                rare_fraction=params.get("rare_fraction", 0.01),
                seed=sc_seed,
            )
            perturbed = bulk.copy()
        elif name.startswith("missing_cell_type"):
            result = apply_missing_cell_type(
                bulk, true_proportions,
                n_missing_types=params.get("n_missing_types", 1),
                seed=sc_seed,
            )
            perturbed = result["bulk"]
            y_true = result["true_proportions"]
        elif name in ("cross_donor", "cross_dataset"):
            logger.info("  (%s handled by caller — skipping in-tier)", name)
            continue
        else:
            logger.warning("Unknown scenario '%s', skipping", name)
            continue

        y_pred = deconvolution_fn(perturbed)
        acc = compute_accuracy_metrics(y_true, y_pred)
        summaries.append({
            "tier": tier_name,
            "scenario": name,
            "overall_MAE": acc["overall"]["MAE"].values[0],
            "overall_RMSE": acc["overall"]["RMSE"].values[0],
            "overall_Pearson_r": acc["overall"]["Pearson_r"].values[0],
        })

    return pd.DataFrame(summaries)
