from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.discrete.conditional_models as cm
import textstat
from bert_score import score as bert_score

EO_PREDICTORS = ["is_evaluator_llm", "bert_f1", "word_count", "fkgl", "ttr"]


def _ttr(text: str) -> float:
    """Compute Type-Token Ratio (lexical diversity)."""
    tokens = text.lower().split()
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def compute_summary_metrics(
    summaries_df: pd.DataFrame,
    cache_path: Path | None = None,
    model_type: str = "roberta-base",
    force: bool = False,
) -> pd.DataFrame:
    """
    Step 3: Compute readability, structural, and BERTScore metrics for each summary.
    Expects columns: 'resume_id', 'author', 'summary_text', 'resume_text'.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force:
            if cache_path.suffix == ".parquet":
                return pd.read_parquet(cache_path)
            return pd.read_csv(cache_path, dtype={"resume_id": str})

    df = summaries_df.copy()
    summaries = df["summary_text"].tolist()
    references = df["resume_text"].tolist()

    # Readability & structural metrics
    df["word_count"] = [len(s.split()) for s in summaries]
    df["fkgl"] = [textstat.flesch_kincaid_grade(s) for s in summaries]
    df["reading_ease"] = [textstat.flesch_reading_ease(s) for s in summaries]
    df["ttr"] = [_ttr(s) for s in summaries]

    # Semantic fidelity (BERTScore)
    P, R, F1 = bert_score(
        cands=summaries,
        refs=references,
        lang="en",
        model_type=model_type,
        rescale_with_baseline=True,
        verbose=True,
    )
    df["bert_precision"] = P.numpy()
    df["bert_recall"] = R.numpy()
    df["bert_f1"] = F1.numpy()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.suffix == ".parquet":
            df.to_parquet(cache_path, index=False)
        else:
            df.to_csv(cache_path, index=False)

    return df


def fit_conditional_logit(
    sub_df: pd.DataFrame,
    predictors: list[str] = EO_PREDICTORS,
) -> dict:
    """
    Step 5 (core): Fit ConditionalLogit on a 2-row-per-pair DataFrame and
    convert beta_1 into Equal Opportunity Self-Preference Bias (%).
    """
    model = cm.ConditionalLogit(
        endog=sub_df["preferred"],
        exog=sub_df[predictors],
        groups=sub_df["pair_id"],
    )
    res = model.fit(disp=False)

    b1 = float(res.params["is_evaluator_llm"])
    pval = float(res.pvalues["is_evaluator_llm"])
    ci_low, ci_high = [float(v) for v in res.conf_int().loc["is_evaluator_llm"]]

    to_eo = lambda b: (np.exp(b) / (1.0 + np.exp(b)) - 1.0 / (1.0 + np.exp(b))) * 100.0
    return {
        "n_pairs": int(sub_df["pair_id"].nunique()),
        "beta_1": b1,
        "p_value": pval,
        "eo_bias_pct": to_eo(b1),
        "ci_low_pct": to_eo(ci_low),
        "ci_high_pct": to_eo(ci_high),
    }


def compute_eo_table(
    paired_df: pd.DataFrame,
    predictors: list[str] = EO_PREDICTORS,
) -> pd.DataFrame:
    """
    Step 5 (wrapper): Run ConditionalLogit across every (model, opponent) matchup.
    """
    records = []
    for (evaluator, opponent), sub in paired_df.groupby(["model", "opponent"]):
        stats = fit_conditional_logit(sub, predictors=predictors)
        records.append({"model": evaluator, "opponent": opponent, **stats})
    return pd.DataFrame(records)