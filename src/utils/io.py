"""
Shared utilities: logging, I/O, configuration, QC summary, and
reproducibility helpers.

All scripts use these utilities for consistent logging, file naming,
random seeding, and QC reporting.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


# ── Configuration ──────────────────────────────────────────────────────────

def load_config(config_path: str = None) -> dict:
    """Load YAML configuration from the given path.

    If no path is given, looks for ``config/config.yaml`` relative to
    the project root (two levels up from this file).
    """
    if config_path is None:
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_project_root() -> Path:
    """Return the absolute path to the project root."""
    return Path(__file__).resolve().parents[2]


def is_debug_mode(cfg: dict = None) -> bool:
    """Check whether debug mode is active."""
    if cfg is None:
        cfg = load_config()
    return cfg.get("project", {}).get("debug", False)


def debug_param(cfg: dict, section: str, key: str, default: Any) -> Any:
    """Return the debug override value if debug mode is on, else default."""
    if is_debug_mode(cfg):
        debug_val = (
            cfg.get(section, {})
            .get("debug", {})
            .get(key)
        )
        if debug_val is not None:
            return debug_val
    return default


# ── Logging ────────────────────────────────────────────────────────────────

def setup_logger(
    name: str = "calibdeconv",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    verbose: bool = False,
) -> logging.Logger:
    """Create a logger that writes to both console and a timestamped file.

    Parameters
    ----------
    name : str
        Logger name.
    log_dir : str or None
        Directory for log files. Defaults to ``<project_root>/logs``.
    level : int
        Logging level (e.g. ``logging.INFO``).
    verbose : bool
        If True, set console handler to DEBUG level.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)  # capture everything; handlers filter

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else level)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # File handler
    if log_dir is None:
        log_dir = str(get_project_root() / "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{name}_{timestamp}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d %(message)s"
    )
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    logger.info("Logging to %s", log_file)
    return logger


# ── Reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> np.random.Generator:
    """Set global random seeds and return a new Generator."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    return rng


def record_seed(seed: int, output_path: str) -> None:
    """Write the seed value into a text file for audit trail."""
    with open(output_path, "w") as fh:
        fh.write(f"seed={seed}\n")


# ── File I/O ───────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_df(df: "pd.DataFrame", path: str, index: bool = False, **kwargs) -> None:
    """Save a DataFrame to CSV with consistent defaults."""
    ensure_dir(str(Path(path).parent))
    df.to_csv(path, index=index, **kwargs)


def load_df(path: str, **kwargs) -> "pd.DataFrame":
    """Load a DataFrame from CSV."""
    return pd.read_csv(path, **kwargs)


# ── QC Summary ─────────────────────────────────────────────────────────────

def run_qc_summary(
    df: pd.DataFrame,
    label: str = "dataframe",
    expect_proportions: bool = False,
    proportion_tolerance: float = 0.01,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run a standard QC check on a DataFrame and return a summary dict.

    Checks performed:
    - n_samples (rows), n_genes/n_cell_types (cols)
    - any NA, any negative, any infinite values
    - proportion sum check (if expect_proportions=True)

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.
    label : str
        Human-readable label for the QC report.
    expect_proportions : bool
        If True, check that each row sums to approximately 1.0.
    proportion_tolerance : float
        Allowed deviation from 1.0 for proportion sum check.
    logger : logging.Logger or None

    Returns
    -------
    dict
        QC results.
    """
    qc = {
        "label": label,
        "n_samples": df.shape[0],
        "n_columns": df.shape[1],
        "any_na": bool(df.isna().any().any()),
        "any_negative": bool((df.select_dtypes(include=[np.number]) < 0).any().any()),
        "any_infinite": bool(
            np.isinf(df.select_dtypes(include=[np.number])).any().any()
        ),
        "dtype_summary": str(df.dtypes.value_counts().to_dict()),
    }

    if expect_proportions:
        row_sums = df.select_dtypes(include=[np.number]).sum(axis=1)
        qc["proportion_sum_min"] = float(row_sums.min())
        qc["proportion_sum_max"] = float(row_sums.max())
        qc["proportion_sum_mean"] = float(row_sums.mean())
        qc["proportion_sum_out_of_bounds"] = int(
            (np.abs(row_sums - 1.0) > proportion_tolerance).sum()
        )

    if logger is not None:
        _log_qc(qc, logger)

    return qc


def run_qc_on_path(
    path: str,
    label: str = None,
    expect_proportions: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Load a CSV and run QC on it, also reporting the file path."""
    if label is None:
        label = Path(path).stem
    df = load_df(path, index_col=0) if _has_index_col(path) else load_df(path)
    qc = run_qc_summary(
        df, label=label, expect_proportions=expect_proportions, logger=logger
    )
    qc["file_path"] = str(Path(path).resolve())
    qc["file_size_kb"] = round(Path(path).stat().st_size / 1024, 1)
    return qc


def qc_report_all(
    results: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> str:
    """Print a formatted QC report for multiple results in a table."""
    if logger:
        logger.info("=" * 60)
        logger.info("QC SUMMARY REPORT")
        logger.info("=" * 60)
    lines = []
    for r in results:
        msg = (
            f"[{r.get('label', '?')}] "
            f"samples={r.get('n_samples', '?')} "
            f"cols={r.get('n_columns', '?')} "
            f"NA={r.get('any_na', '?')} "
            f"neg={r.get('any_negative', '?')} "
            f"inf={r.get('any_infinite', '?')}"
        )
        if "proportion_sum_mean" in r:
            msg += f" prop_sum={r['proportion_sum_mean']:.4f} (OOB={r.get('proportion_sum_out_of_bounds', '?')})"
        if "file_path" in r:
            msg += f"\n      path={r['file_path']}"
        if logger:
            logger.info(msg)
        lines.append(msg)
    return "\n".join(lines)


def _has_index_col(path: str) -> bool:
    """Heuristic: check if first column looks like an index."""
    try:
        with open(path, "r") as fh:
            header = fh.readline().strip()
        return header.startswith(",") or header.split(",")[0] in ("", "sample_id", "gene")
    except Exception:
        return False


def _log_qc(qc: dict, logger: logging.Logger) -> None:
    """Log a single QC result."""
    status = "PASS" if not (qc["any_na"] or qc["any_negative"] or qc["any_infinite"]) else "FAIL"
    logger.info(
        "QC [%s] %s: samples=%d cols=%d NA=%s neg=%s inf=%s",
        status, qc["label"],
        qc["n_samples"], qc["n_columns"],
        qc["any_na"], qc["any_negative"], qc["any_infinite"],
    )
    if "proportion_sum_mean" in qc:
        logger.info(
            "  Proportion check: mean=%.4f min=%.4f max=%.4f OOB=%d",
            qc["proportion_sum_mean"], qc["proportion_sum_min"],
            qc["proportion_sum_max"], qc["proportion_sum_out_of_bounds"],
        )
    if qc["any_na"] or qc["any_negative"] or qc["any_infinite"]:
        logger.warning("  QC FAILED for %s — review data before proceeding!", qc["label"])
