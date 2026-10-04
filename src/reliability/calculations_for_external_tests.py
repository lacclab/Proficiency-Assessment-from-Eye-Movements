"""Reliability of the Michigan (MTELP) test from its item-level responses, the
external reference for the eye-movement reliability numbers: per part and for
the 100-item whole test, Cronbach's alpha, split-half and the six ICCs, with
the estimators of cronbach_alpha.py. Writes results/michigan_*.csv and
results/tables/michigan_*.tex.

The item responses are OneStop L2 data, which is not publicly available.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import Fields
from src.reliability.cronbach_alpha import (
    cronbachs_alpha_single,
    icc_variants_single,
    split_half_reliability_items,
)

PARTS_DICT = {
    "MPT_listening_score": (1, 20),
    "MPT_grammar_score": (21, 50),
    "MPT_vocabulary_score": (51, 80),
    "MPT_reading_score": (81, 100)
}


# The eye-movement reliability numbers these get compared against are computed
# within a single preview condition, so Michigan is reported the same way rather
# than pooled across both cohorts.
PREVIEWS = {"All": None, "Gathering": False, "Hunting": True}

# Pooled last in the tables: the two reading regimes are what the eye-movement
# tables report, and All is the aggregate over them.
PREVIEW_TABLE_ORDER = ["Gathering", "Hunting", "All"]

PART_DISPLAY = {
    "MPT_listening_score": "Listening",
    "MPT_grammar_score": "Grammar",
    "MPT_vocabulary_score": "Vocabulary",
    "MPT_reading_score": "Reading",
    "whole_test": "Whole Test",
}


def get_michigan_binary(mich_df_path, l2_df_path, preview=None):
    """
    Michigan item responses, restricted to readers present in the eye-movement data.

    preview selects the reading condition by question_preview: False = Gathering,
    True = Hunting, None = both pooled.
    """
    mich_df_org = pd.read_csv(mich_df_path)
    l2_df = pd.read_csv(l2_df_path)
    if preview is not None:
        l2_df = l2_df[l2_df[Fields.HAS_PREVIEW] == preview]
    l2_df["participant_id_clean"] = l2_df["participant_id"].apply(lambda x: int(x.split('_')[-1]))
    mich_df_org_filtered = mich_df_org[mich_df_org[Fields.SUBJECT_ID].apply(lambda x: int(x) in list(l2_df["participant_id_clean"].unique()) if x.isnumeric() else False)]

    return mich_df_org_filtered


def get_michigan_parts_binary(michigan_df):
    """Extract each test part as a separate dataframe with question responses as items."""
    michigan_parts_binary = {}
    for part, (first, last) in PARTS_DICT.items():
        part_cols = [f"{i}_response" for i in range(first, last + 1)]
        michigan_parts_binary[part] = michigan_df[part_cols + [Fields.SUBJECT_ID]]
    return michigan_parts_binary


def reshape_part_for_reliability(part_df):
    """Wide (one `<n>_response` column per question) to the long frame the
    reliability estimators take: participant_id, item, correctness."""
    question_cols = [col for col in part_df.columns if col.endswith('_response')]

    # Reshape to long format
    long_df = part_df.melt(
        id_vars=[Fields.SUBJECT_ID],
        value_vars=question_cols,
        var_name='question',
        value_name='correctness'
    )

    # Create item column from question number
    long_df['item'] = long_df['question'].str.replace('_response', '')
    long_df = long_df[[Fields.SUBJECT_ID, 'item', 'correctness']]

    return long_df


def calculate_michigan_part_reliability(part_df, n_iterations=100):
    """Cronbach's alpha and split-half of one test part (wide responses)."""
    long_df = reshape_part_for_reliability(part_df)
    long_df = long_df.dropna(subset=['correctness'])

    if len(long_df) < 2:
        return {'cronbachs_alpha': np.nan, 'split_half': np.nan}

    cronbach = cronbachs_alpha_single(long_df, value_col='correctness', item_col='item')

    split_half = split_half_reliability_items(
        long_df,
        value_col='correctness',
        participant_col=Fields.SUBJECT_ID,
        item_col='item',
        n_iterations=n_iterations
    )

    return {
        'cronbachs_alpha': cronbach,
        'split_half': split_half
    }


def calculate_all_michigan_parts_reliability(michigan_df, n_iterations=100):
    """One row per test part: part, cronbachs_alpha, split_half, n_participants."""
    parts_binary = get_michigan_parts_binary(michigan_df)
    n_participants = michigan_df[Fields.SUBJECT_ID].nunique()

    results = []
    for part_name, part_df in parts_binary.items():
        reliability = calculate_michigan_part_reliability(part_df, n_iterations)
        results.append({
            'part': part_name,
            'cronbachs_alpha': reliability['cronbachs_alpha'],
            'split_half': reliability['split_half'],
            'n_participants': n_participants,
        })

    return pd.DataFrame(results)


def save_michigan_reliability_results(results_df, output_path):
    """Save reliability results to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Saved reliability results to {output_path}")


def whole_test_reliability(michigan_df, n_iterations=100):
    """
    Reliability of the 100-item total score, by three estimators.

    whole_test_alpha treats all 100 items as one scale. stratified_alpha is the
    same quantity estimated per part and recombined, which is the standard
    correction when a test is built from heterogeneous strata (here: listening,
    grammar, vocabulary, reading). The two agreeing is the evidence that treating
    the test as one scale is not distorting the estimate; the two diverging would
    mean the parts are too heterogeneous to pool, and stratified is the one to
    report.

    Both describe the total score, so neither is comparable to the per-part
    average - a mean of four subtest reliabilities estimates no single score.
    """
    def alpha(x):
        n_items = x.shape[1]
        total_var = x.sum(axis=1).var(ddof=1)
        if total_var <= 0 or n_items < 2:
            return np.nan
        return n_items / (n_items - 1) * (1 - x.var(axis=0, ddof=1).sum() / total_var)

    all_items = michigan_df[[f"{i}_response" for i in range(1, 101)]].astype(float).values
    total_var = all_items.sum(axis=1).var(ddof=1)

    # Stratified alpha: 1 - sum of each part's *unreliable* variance / total variance
    unreliable_var = 0.0
    for first, last in PARTS_DICT.values():
        part = michigan_df[[f"{i}_response" for i in range(first, last + 1)]].astype(float).values
        unreliable_var += part.sum(axis=1).var(ddof=1) * (1 - alpha(part))

    # Split-half over all 100 items, so the whole-test column has the same pair of
    # statistics as the per-part average it sits next to.
    whole_long = reshape_part_for_reliability(
        michigan_df[[f"{i}_response" for i in range(1, 101)] + [Fields.SUBJECT_ID]]
    ).dropna(subset=['correctness'])

    # The plain split-half draws 50 items uniformly, so a half gets 10 listening
    # items only on average (SD ~2) and the two halves differ in composition. The
    # stratified version takes half of each part, matching the halves on content -
    # the split-half analogue of stratified alpha.
    part_of_item = {str(i): part
                    for part, (first, last) in PARTS_DICT.items()
                    for i in range(first, last + 1)}

    return {
        "whole_test_alpha": alpha(all_items),
        "whole_test_split_half": split_half_reliability_items(
            whole_long, value_col='correctness', participant_col=Fields.SUBJECT_ID,
            item_col='item', n_iterations=n_iterations,
        ),
        "whole_test_split_half_stratified": split_half_reliability_items(
            whole_long, value_col='correctness', participant_col=Fields.SUBJECT_ID,
            item_col='item', n_iterations=n_iterations, strata=part_of_item,
        ),
        "stratified_alpha": 1 - unreliable_var / total_var if total_var > 0 else np.nan,
    }


def calculate_michigan_icc_variants(michigan_df):
    """
    The six Shrout & Fleiss ICCs per test part, plus the 100-item whole test.

    Same estimator the eye-movement side uses, so the two are directly
    comparable. The complete-case version is exact here: every reader answered
    every item, so nothing is dropped and the all-readers variant would agree.

    icc3_k equals Cronbach's alpha by construction and is the cross-check
    against the alpha columns in the per-part table.
    """
    scopes = {**PARTS_DICT, "whole_test": (1, 100)}

    rows = []
    for part_name, (first, last) in scopes.items():
        part_cols = [f"{i}_response" for i in range(first, last + 1)]
        long_df = reshape_part_for_reliability(michigan_df[part_cols + [Fields.SUBJECT_ID]])
        long_df = long_df.dropna(subset=['correctness'])

        icc = icc_variants_single(
            long_df, value_col='correctness', item_col='item',
            participant_col=Fields.SUBJECT_ID,
        )
        rows.append({'part': part_name, **icc})

    return pd.DataFrame(rows)


def summarize_over_parts(results_df, whole_test_by_preview):
    """
    One row per preview: the per-part average alongside the two total-score estimates.

    The averages are unweighted - the parts differ in length (20/30/30/20 items),
    so these are the mean of the four reported reliabilities, not the reliability
    of the 100-item test. The latter is what whole_test_alpha / stratified_alpha
    give, and they run ~0.14 higher purely because the test is five times longer
    than any one part.
    """
    summary = (
        results_df.groupby('preview', sort=False)
        .agg(avg_part_alpha=('cronbachs_alpha', 'mean'),
             avg_part_split_half=('split_half', 'mean'),
             n_participants=('n_participants', 'first'),
             n_parts=('part', 'nunique'))
        .reset_index()
    )
    for col in ("whole_test_alpha", "whole_test_split_half",
                "whole_test_split_half_stratified", "stratified_alpha"):
        summary[col] = summary['preview'].map(lambda p: whole_test_by_preview[p][col])

    return summary[['preview', 'n_participants', 'n_parts',
                    'avg_part_alpha', 'avg_part_split_half',
                    'whole_test_alpha', 'whole_test_split_half',
                    'whole_test_split_half_stratified', 'stratified_alpha']]


def _fmt(value, decimals=2):
    """Table cell: a dash for anything that could not be estimated."""
    return "-" if pd.isna(value) else f"{value:.{decimals}f}"


def _write_tex(lines, output_path):
    # No bolding pass: these are reliability tables, and the best-per-column
    # mark would read as a contest between the test parts and regimes, which
    # is not what alpha and split-half report.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n")
    print(f"Saved table to {output_path}")


def write_per_part_table(results_df, output_path):
    """Cronbach's alpha and split-half per test part, one column pair per regime."""
    lines = [
        r"\begin{table}[ht!]", r"\centering", r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{l|cc|cc|cc}", r"\toprule",
        r" \multicolumn{1}{c}{} & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{\textbf{{{p}}}}}" for p in PREVIEW_TABLE_ORDER) + r" \\",
        r"\cmidrule{2-7}",
        r"\textbf{Part} & " + " & ".join([r"$\alpha$ & Split-Half $r$"] * 3) + r" \\",
        r"\midrule",
    ]

    n_by_preview = results_df.groupby('preview')['n_participants'].first()
    for part, display in PART_DISPLAY.items():
        if part == "whole_test":
            continue
        cells = []
        for preview in PREVIEW_TABLE_ORDER:
            row = results_df[(results_df['preview'] == preview) & (results_df['part'] == part)]
            cells += [_fmt(row['cronbachs_alpha'].iloc[0]), _fmt(row['split_half'].iloc[0])]
        lines.append(f"{display} & " + " & ".join(cells) + r" \\")

    sample = ", ".join(rf"{p}: $n$={n_by_preview[p]}" for p in PREVIEW_TABLE_ORDER)
    lines += [
        r"\end{tabular}", "}",
        r"\caption{\textbf{Internal consistency of the Michigan (MTELP) test parts}, "
        r"computed over the item-level responses of the readers present in the eye-movement data "
        rf"({sample}). Gathering and Hunting are the two reading regimes; All pools them. "
        r"Parts contain 20/30/30/20 items, so these values are not comparable to the "
        r"100-item whole-test reliability in Table~\ref{tab:michigan-reliability-summary}.}",
        r"\label{tab:michigan-reliability-parts}",
        r"\end{table}",
    ]
    _write_tex(lines, output_path)


def write_summary_table(summary_df, output_path):
    """Per-part average against the two whole-test estimates, one row per regime."""
    lines = [
        r"\begin{table}[ht!]", r"\centering", r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{lc|cc|cccc}", r"\toprule",
        r" \multicolumn{2}{c}{} & \multicolumn{2}{c}{Avg. over parts} & "
        r"\multicolumn{4}{c}{Whole test (100 items)} \\",
        r"\cmidrule{3-8}",
        r"\textbf{Regime} & $n$ & $\alpha$ & Split-Half $r$ & $\alpha$ & Split-Half $r$ "
        r"& Strat.\ Split-Half $r$ & Stratified $\alpha$ \\",
        r"\midrule",
    ]

    indexed = summary_df.set_index('preview')
    for preview in PREVIEW_TABLE_ORDER:
        row = indexed.loc[preview]
        lines.append(
            f"{preview} & {int(row['n_participants'])} & "
            f"{_fmt(row['avg_part_alpha'])} & {_fmt(row['avg_part_split_half'])} & "
            f"{_fmt(row['whole_test_alpha'])} & {_fmt(row['whole_test_split_half'])} & "
            f"{_fmt(row['whole_test_split_half_stratified'])} & "
            f"{_fmt(row['stratified_alpha'])} \\\\"
        )

    lines += [
        r"\end{tabular}", "}",
        r"\caption{\textbf{Michigan (MTELP) reliability at the part and whole-test level.} "
        r"The averages are unweighted means over the four parts and estimate no single score; "
        r"the whole-test columns estimate the reliability of the 100-item total, which is the "
        r"score used as a prediction target. Stratified $\alpha$ recombines the four part "
        r"estimates weighted by their variances, the standard correction when a test is built "
        r"from heterogeneous strata; its agreement with plain $\alpha$ indicates that pooling "
        r"the parts into one scale does not distort the estimate. Split-Half draws the two "
        r"halves uniformly over the 100 items, so they differ in part composition; "
        r"Strat.\ Split-Half draws half of each part instead, matching the halves on content. "
        r"The whole-test values run "
        r"higher than the part average because the test is five times longer than any one part.}",
        r"\label{tab:michigan-reliability-summary}",
        r"\end{table}",
    ]
    _write_tex(lines, output_path)


def write_icc_table(icc_df, output_path):
    """All six ICC variants, blocked by regime, mirroring the EyeScore ICC table."""
    lines = [
        r"\begin{table}[ht!]", r"\centering", r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{l|cc|cc|cc}", r"\toprule",
        r" \multicolumn{1}{c}{} & \multicolumn{2}{c}{ICC(1)} & \multicolumn{2}{c}{ICC(2)} "
        r"& \multicolumn{2}{c}{ICC(3)} \\",
        r"\cmidrule{2-7}",
        r"\textbf{Part} & " + " & ".join(["Single & Avg."] * 3) + r" \\",
    ]

    for block_index, preview in enumerate(PREVIEW_TABLE_ORDER):
        block = icc_df[icc_df['preview'] == preview].set_index('part')
        n_participants = int(block['n_participants'].iloc[0])
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{7}}{{l}}{{\textbf{{{preview}}} \quad "
            rf"{{\small ($n$={n_participants})}}}} \\"
        )
        for part, display in PART_DISPLAY.items():
            row = block.loc[part]
            cells = [_fmt(row[v]) for v in
                     ("icc1_1", "icc1_k", "icc2_1", "icc2_k", "icc3_1", "icc3_k")]
            if part == "whole_test":
                lines.append(r"\cmidrule{1-7}")
            lines.append(f"{display} ($k$={int(row['n_items'])}) & " + " & ".join(cells) + r" \\")

    lines += [
        r"\end{tabular}", "}",
        r"\caption{\textbf{Intraclass correlations for the Michigan (MTELP) test}, with readers "
        r"as targets and items as raters, computed with the same estimator as the EyeScore ICCs "
        r"in Table~\ref{tab:icc-variants-eyescore}. ICC(1) is the one-way model, in which item "
        r"effects cannot be separated from error; ICC(2) is two-way random with absolute "
        r"agreement, counting systematic item differences as error; ICC(3) is two-way mixed, "
        r"treating the item set as fixed (consistency). Single is the reliability of one item, "
        r"Avg.\ that of the mean over all $k$ items; ICC(3,$k$) equals Cronbach's $\alpha$ by "
        r"construction. Every reader answered every item, so no listwise deletion applies and "
        r"$n$ is the full sample. The Single columns are far below their EyeScore counterparts "
        r"because one Michigan item is a single binary response, whereas one EyeScore item is an "
        r"entire passage or article.}",
        r"\label{tab:icc-variants-michigan}",
        r"\end{table}",
    ]
    _write_tex(lines, output_path)


MICHIGAN_RESPONSES_UNAVAILABLE = (
    "The Michigan reliability needs the OneStop L2 Michigan item responses, which are "
    "not publicly available."
)


def main(mich_csv_path=None, l2_csv_path=None):
    """Compute and write the Michigan reliability results from the item
    responses (one 0/1 column per item) and the OneStop L2 features file, whose
    readers and preview conditions they are restricted to."""
    if mich_csv_path is None or l2_csv_path is None:
        raise SystemExit(MICHIGAN_RESPONSES_UNAVAILABLE)
    output_path = "src/reliability/results/michigan_reliability.csv"
    summary_output_path = "src/reliability/results/michigan_reliability_summary.csv"
    icc_output_path = "src/reliability/results/michigan_icc.csv"
    tables_dir = "src/reliability/results/tables"
    n_iterations = 20

    print(f"Loading Michigan test data from {mich_csv_path}...")

    all_results = []
    all_iccs = []
    whole_test_by_preview = {}
    for preview_name, preview_value in PREVIEWS.items():
        mich_df = get_michigan_binary(mich_csv_path, l2_csv_path, preview=preview_value)
        print(f"\n{preview_name}: {mich_df[Fields.SUBJECT_ID].nunique()} participants "
              f"(n_iterations={n_iterations})")

        results_df = calculate_all_michigan_parts_reliability(mich_df, n_iterations=n_iterations)
        results_df.insert(0, 'preview', preview_name)
        all_results.append(results_df)
        whole_test_by_preview[preview_name] = whole_test_reliability(
            mich_df, n_iterations=n_iterations)

        icc_df = calculate_michigan_icc_variants(mich_df)
        icc_df.insert(0, 'preview', preview_name)
        all_iccs.append(icc_df)

        print("=" * 70)
        for _, row in results_df.iterrows():
            print(f"{row['part']:30s} | Cronbach's α: {row['cronbachs_alpha']:.4f} | Split-Half: {row['split_half']:.4f}")
        print("-" * 70)
        print(f"{'Average':30s} | Cronbach's α: {results_df['cronbachs_alpha'].mean():.4f} "
              f"| Split-Half: {results_df['split_half'].mean():.4f}")
        print("=" * 70)

    results_df = pd.concat(all_results, ignore_index=True)
    summary_df = summarize_over_parts(results_df, whole_test_by_preview)
    icc_df = pd.concat(all_iccs, ignore_index=True)

    print("\nSummary (per-part average vs whole-test estimates):")
    print("=" * 100)
    for _, row in summary_df.iterrows():
        print(f"{row['preview']:12s} (N={row['n_participants']:3d}) | avg part α: {row['avg_part_alpha']:.4f} "
              f"| avg part split-half: {row['avg_part_split_half']:.4f} "
              f"| whole-test α: {row['whole_test_alpha']:.4f} "
              f"| whole-test split-half: {row['whole_test_split_half']:.4f} "
              f"(stratified: {row['whole_test_split_half_stratified']:.4f}) "
              f"| stratified α: {row['stratified_alpha']:.4f}")
    print("=" * 100)

    save_michigan_reliability_results(results_df, output_path)
    save_michigan_reliability_results(summary_df, summary_output_path)
    save_michigan_reliability_results(icc_df, icc_output_path)

    write_per_part_table(results_df, f"{tables_dir}/michigan_reliability_parts.tex")
    write_summary_table(summary_df, f"{tables_dir}/michigan_reliability_summary.tex")
    write_icc_table(icc_df, f"{tables_dir}/michigan_icc.tex")


if __name__ == "__main__":
    main()
