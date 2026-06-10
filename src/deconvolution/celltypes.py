"""
Cell-type collapse map and pipeline input-file resolution.

Shared across the formal pipeline (scripts 03/04/05/06) so the PRIMARY line
(marker genes + 5 collapsed cell types) and the diagnostic lines
(HVG / 7 types) are resolved consistently and without code duplication.

PRIMARY analysis line:
  gene_set      = "markers"   -> selected_genes_markers.txt (reference-pool DE)
  cell_type_set = "5type"     -> T_cell = CD4_T+CD8_T+other_T; B; NK; Monocyte; DC
  scale         = CPM
"""

from pathlib import Path

# The closely-related T subsets that collapse under NNLS (collinearity).
T_SUBSETS = ["CD4_T", "CD8_T", "other_T"]
PRIMARY_5TYPES = ["T_cell", "B", "NK", "Monocyte", "DC"]


def collapse_labels_5type(series):
    """Map 7-type labels to 5 types (T subsets -> T_cell)."""
    return series.map(lambda x: "T_cell" if x in T_SUBSETS else x)


def add_collapsed_column(adata, src_col="cell_type", dst_col="cell_type5"):
    """Add a 5-type collapsed cell-type column to ``adata.obs`` in place.

    Returns the destination column name so callers can use it as the
    cell-type column for signature building.
    """
    adata.obs[dst_col] = collapse_labels_5type(adata.obs[src_col]).astype("category")
    return dst_col


def resolve_cell_type_column(adata, cell_type_set, base_col="cell_type"):
    """Return the obs column to use for signatures given the cell-type set.

    For ``5type`` a collapsed column is created on ``adata`` if missing.
    """
    if cell_type_set == "5type":
        return add_collapsed_column(adata, src_col=base_col, dst_col="cell_type5")
    return base_col


def resolve_inputs(gene_set, cell_type_set, pb_dir):
    """Resolve pseudo-bulk / panel / true-proportion file names.

    Parameters
    ----------
    gene_set : {"all", "hvg3000", "markers"}
        Determines the X (expression) matrices and the gene panel.
    cell_type_set : {"7type", "5type"}
        Determines the y (true-proportion) files. Orthogonal to gene_set:
        the gene panel subsets columns (genes), the cell-type set subsets
        the proportion columns (cell types).
    pb_dir : Path
        Directory containing the processed CSVs.

    Returns
    -------
    dict with keys:
        cal_file, test_file        : pseudo-bulk CPM matrices (Path)
        panel_file                 : gene-panel txt or None (Path | None)
        true_cal, true_test        : true-proportion CSVs (Path)
        sig_out_name               : default signature filename (str)
        default_out_subdir         : suggested results/ subdir (str)
    """
    pb_dir = Path(pb_dir)

    # ── X matrices + gene panel (from gene_set) ──
    if gene_set == "markers":
        cal_file = pb_dir / "pseudobulk_matrix_cal_markers_cpm.csv"
        test_file = pb_dir / "pseudobulk_matrix_test_markers_cpm.csv"
        panel_file = pb_dir / "selected_genes_markers.txt"
        sig_out_name = "signature_matrix_markers.csv"
    elif gene_set == "hvg3000":
        cal_file = pb_dir / "pseudobulk_matrix_cal_hvg3000_cpm.csv"
        test_file = pb_dir / "pseudobulk_matrix_test_hvg3000_cpm.csv"
        panel_file = pb_dir / "selected_genes_hvg3000.txt"
        sig_out_name = "signature_matrix_hvg3000.csv"
    elif gene_set == "all":
        cal_file = pb_dir / "pseudobulk_matrix_cal_cpm.csv"
        test_file = pb_dir / "pseudobulk_matrix_test_cpm.csv"
        panel_file = None
        sig_out_name = "signature_matrix.csv"
    else:
        raise ValueError(f"Unknown gene_set: {gene_set}")

    # ── y files (from cell_type_set) ──
    if cell_type_set == "5type":
        true_cal = pb_dir / "true_proportions_cal_5type.csv"
        true_test = pb_dir / "true_proportions_test_5type.csv"
    elif cell_type_set == "7type":
        true_cal = pb_dir / "true_proportions_cal.csv"
        true_test = pb_dir / "true_proportions_test.csv"
    else:
        raise ValueError(f"Unknown cell_type_set: {cell_type_set}")

    # ── default output subdir (PRIMARY vs diagnostic) ──
    if gene_set == "markers" and cell_type_set == "5type":
        default_out_subdir = "nnls_marker_5types"
    elif gene_set == "markers" and cell_type_set == "7type":
        default_out_subdir = "nnls_marker_7types"
    elif gene_set == "hvg3000":
        default_out_subdir = "nnls_hvg3000_7types"
    else:
        default_out_subdir = "nnls"

    return {
        "cal_file": cal_file,
        "test_file": test_file,
        "panel_file": panel_file,
        "true_cal": true_cal,
        "true_test": true_test,
        "sig_out_name": sig_out_name,
        "default_out_subdir": default_out_subdir,
    }
