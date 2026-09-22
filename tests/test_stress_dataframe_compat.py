#!/usr/bin/env python
"""
Regression test for the stress perturbation functions.

Motivation: until v1.2.2, all four functions in src/evaluation/stress.py
mutated the frame through `DataFrame.values[...] = ...`. Under pandas 3.0
Copy-on-Write `.values` returns a read-only array, so every one of them raised
`ValueError: assignment destination is read-only`, and neither Tier 1
(Figure 4) nor Tier 2 (Figure 5) could run on a current install.

This test is intentionally dependency-free: it runs under plain `python`
in both the pandas 2.x and pandas 3.x environments, and is also collectable by
pytest if it is ever added.

    python tests/test_stress_dataframe_compat.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.stress import (  # noqa: E402
    apply_batch_shift, apply_dropout, apply_gaussian_noise, apply_low_depth,
)

N_SAMPLES, N_GENES = 12, 20


def _bulk(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.integers(1, 1000, (N_SAMPLES, N_GENES)).astype(float),
        index=[f"s{i}" for i in range(N_SAMPLES)],
        columns=[f"g{j}" for j in range(N_GENES)])


CASES = [
    ("apply_dropout", apply_dropout, dict(dropout_rate=0.3)),
    ("apply_gaussian_noise", apply_gaussian_noise, dict(noise_std=0.5)),
    ("apply_low_depth", apply_low_depth, dict(depth_fraction=0.25)),
    ("apply_batch_shift", apply_batch_shift, dict(shift_size=0.5)),
]


def test_runs_without_touching_readonly_values():
    """Each function returns a frame; none writes through a read-only view."""
    for name, fn, kw in CASES:
        bulk = _bulk()
        out = fn(bulk, seed=42, **kw)
        assert isinstance(out, pd.DataFrame), name


def test_input_frame_is_not_mutated():
    """Perturbation must not modify the caller's frame in place."""
    for name, fn, kw in CASES:
        bulk = _bulk()
        before = bulk.to_numpy(copy=True)
        fn(bulk, seed=42, **kw)
        assert np.array_equal(bulk.to_numpy(), before), f"{name} mutated its input"


def test_output_structure_matches_input():
    """Index, columns, shape and dtype are preserved."""
    for name, fn, kw in CASES:
        bulk = _bulk()
        out = fn(bulk, seed=42, **kw)
        assert list(out.index) == list(bulk.index), name
        assert list(out.columns) == list(bulk.columns), name
        assert out.shape == bulk.shape, name
        assert out.to_numpy().dtype == bulk.to_numpy().dtype, name


def test_deterministic_for_a_fixed_seed():
    """Same seed, same output — the scenario seeds pinned in v1.2.1 rely on this."""
    for name, fn, kw in CASES:
        a = fn(_bulk(), seed=42, **kw)
        b = fn(_bulk(), seed=42, **kw)
        assert np.array_equal(a.to_numpy(), b.to_numpy()), name


def test_perturbations_have_their_intended_effect():
    """Guards against a 'fix' that silently turns a perturbation into a no-op."""
    bulk = _bulk()

    zeroed = (apply_dropout(bulk, dropout_rate=0.3, seed=42).to_numpy() == 0).mean()
    assert 0.15 < zeroed < 0.45, f"dropout zeroed {zeroed:.3f} of entries"

    noisy = apply_gaussian_noise(bulk, noise_std=0.5, seed=42).to_numpy()
    assert not np.array_equal(noisy, bulk.to_numpy())
    assert (noisy >= 0).all(), "gaussian noise must stay clipped at zero"

    thinned = apply_low_depth(bulk, depth_fraction=0.25, seed=42).to_numpy()
    assert thinned.sum() < bulk.to_numpy().sum(), "low depth must remove signal"

    shifted = apply_batch_shift(bulk, shift_size=0.5, seed=42).to_numpy()
    changed_rows = (shifted != bulk.to_numpy()).any(axis=1).sum()
    assert changed_rows == N_SAMPLES // 2, (
        f"batch shift touched {changed_rows} rows, expected {N_SAMPLES // 2}")


def test_low_depth_raw_count_branch():
    """The raw-count branch (used by Tier 1) returns CPM-scaled rows."""
    bulk = _bulk()
    raw = _bulk(seed=1)
    out = apply_low_depth(bulk, depth_fraction=0.25, seed=42, raw_counts=raw)
    assert out.shape == bulk.shape
    assert np.allclose(out.to_numpy().sum(axis=1), 1e6, rtol=1e-6)


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"pandas {pd.__version__} | numpy {np.__version__}")
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
