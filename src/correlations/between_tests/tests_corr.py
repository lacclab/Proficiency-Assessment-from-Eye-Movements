"""Correlations between the proficiency tests in each dataset's metadata.

    python -m src.correlations.between_tests.tests_corr
"""
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr

from src.constants import DATA_PATH

logger.add(DATA_PATH / 'logs' / 'tests_correlations.log', rotation='10 MB', retention='7 days')

# (dataset, column 1, column 2, label 1, label 2, subset label, only rows with a TOEFL score)
TEST_PAIRS = [
    ('OneStopL2', 'lextale_score', 'michtest_score', 'LexTALE', 'Michigan', 'All L2 participants', False),
    ('OneStopL2', 'converted_toefl_score', 'lextale_score', 'TOEFL', 'LexTALE', 'With TOEFL score', True),
    ('OneStopL2', 'converted_toefl_score', 'michtest_score', 'TOEFL', 'Michigan', 'With TOEFL score', True),
    ('OneStopL2', 'comprehension_score-regular_trials', 'lextale_score',
     'Comprehension (regular)', 'LexTALE', 'All L2 participants', False),
    ('MecoL2', 'proficiency_agg', 'lextale_score', 'Proficiency Aggregate', 'LexTALE', 'All L2 participants', False),
]


def load_metadata(dataset: str) -> pd.DataFrame:
    """The dataset's metadata.csv, or an empty frame if it is missing."""
    metadata_path = DATA_PATH / dataset / 'metadata' / 'metadata.csv'
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return pd.DataFrame()

    return pd.read_csv(metadata_path)


def calculate_correlation(x: pd.Series, y: pd.Series) -> dict:
    """Pearson r, p-value and n over the rows where both scores are present
    (not NaN and not negative, i.e. not the -1 missing-score sentinel)."""
    mask = (x.notna()) & (y.notna()) & (x >= 0) & (y >= 0)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return {'r': np.nan, 'p_value': np.nan, 'n': len(x_clean)}

    r, p_value = pearsonr(x_clean, y_clean)
    return {'r': r, 'p_value': p_value, 'n': len(x_clean)}


def compute_test_correlations() -> list[dict]:
    """One result per TEST_PAIRS entry whose dataset's metadata could be loaded."""
    metadata = {}
    results = []
    for dataset, col1, col2, label1, label2, subset, toefl_only in TEST_PAIRS:
        if dataset not in metadata:
            metadata[dataset] = load_metadata(dataset)
        df = metadata[dataset]
        if df.empty:
            continue
        if toefl_only:
            # Participants without a TOEFL score carry the -1 sentinel.
            df = df[df['converted_toefl_score'] > 0]
        results.append({
            'dataset': dataset,
            'measure1': label1,
            'measure2': label2,
            'subset': subset,
            **calculate_correlation(df[col1], df[col2]),
        })
    return results


def format_correlation_result(result: dict) -> str:
    """One correlation result as a display line."""
    measure1 = result.get('measure1', 'Unknown')
    measure2 = result.get('measure2', 'Unknown')
    r = result.get('r', np.nan)
    p_value = result.get('p_value', np.nan)
    n = result.get('n', 0)

    r_str = f"{r:.3f}" if not np.isnan(r) else "N/A"
    p_str = f"{p_value:.4f}" if not np.isnan(p_value) else "N/A"

    return f"  {measure1:25s} ↔ {measure2:25s} | r = {r_str:8s} | p = {p_str:10s} | n = {n:3d}"


def print_correlations():
    """Print every correlation, grouped by dataset."""
    all_results = compute_test_correlations()

    if not all_results:
        print("No correlations could be calculated.")
        return

    grouped = {}
    for result in all_results:
        grouped.setdefault(result['dataset'], []).append(result)

    print("\n" + "=" * 100)
    print("Language Proficiency Test Correlations")
    print("=" * 100)

    for dataset in sorted(grouped.keys()):
        print(f"\n{dataset}")
        print("-" * 100)

        for result in grouped[dataset]:
            print(f"\nSubset: {result.get('subset', '')}")
            print(format_correlation_result(result))

    print("\n" + "=" * 100 + "\n")


if __name__ == '__main__':
    print_correlations()
