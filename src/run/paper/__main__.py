"""Stage 3: render every table and figure the paper reports into paper/.

paper/ is exactly this package's output: each section below writes a fixed set
of published artifacts to their paths under paper/, and nothing else in the
repository writes there.

    eyescore     eyescore/{main,hunting,comprehension}.tex
    predictions  prediction/{main,hunting}.tex,
                 model_comparison/{combined,meco,onestop}/lgbm_{holdout,kfold}.tex,
                 model_comparison/{meco,onestop}/tabpfn.tex
    lang_bias    lang_bias/<method>/{coef,pearson}/,
                 lang_bias/<method>/{composite,michigan}_debiasing/,
                 lang_bias/<method>/tables/eyescore_distance.tex
    debiased     lang_bias/<method>/tables/combined_debiased_lextale_{diff,rawcorr}.tex
    figures      lang_bias/<method>/figures/*.pdf
    reliability  reliability/{main,fixed,hunting,original_vs_per_item,evaluation_delta}.tex
    data         data/language_dist_{onestop,meco}.tex, data/{corpus_stats,feature_counts}.csv

where <method> is each debias method the paper reports (two_step,
distance_mreg, interaction). Every section but data reads the output of an
earlier stage -- the prediction and EyeScore evaluation CSVs, the EyeScore
results trees, the reliability results -- so run it after src.run.predictions,
src.run.eyescore and the src.reliability scripts; data reads data/ directly.
With the MECO results alone, each table renders its MECO part (OneStop cells
show "-") and the tables on OneStop alone are skipped with a warning. Every run
also rewrites README.md, the index of what is under the output root.

    python -m src.run.paper                          # everything
    python -m src.run.paper --only eyescore,predictions
    python -m src.run.paper --out /tmp/paper_check   # render elsewhere, e.g. to diff against paper/
"""
import argparse
from pathlib import Path

from loguru import logger

from src.run.paper import data, debiased, eyescore, figures, lang_bias, predictions, reliability
from src.run.paper.readme import write_readme
from src.utils.cli import parse_csv

SECTIONS = {
    "eyescore": eyescore.render,
    "predictions": predictions.render,
    "lang_bias": lang_bias.render,
    "debiased": debiased.render,
    "figures": figures.render,
    "reliability": reliability.render,
    "data": data.render,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.run.paper", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", type=str,
                        help=f"Comma-separated sections to render (default: all): {','.join(SECTIONS)}")
    parser.add_argument("--out", type=Path, default=Path("paper"),
                        help="Root to render into (default: paper/)")
    args = parser.parse_args()

    sections = parse_csv(args.only) or list(SECTIONS)
    unknown = [s for s in sections if s not in SECTIONS]
    if unknown:
        parser.error(f"unknown section(s) {unknown}; choices: {list(SECTIONS)}")
    for name in sections:
        written = SECTIONS[name](args.out)
        logger.info("{}: {} file(s) written under {}", name, len(written), args.out)
    logger.info("index: {}", write_readme(args.out))


if __name__ == "__main__":
    main()
