"""
Test script implementing `LLM Bias Metrics.docx`:
  - Step A: Compute BERTScore (P, R, F1) + Readability/Structural metrics (word_count, FKGL, FRE, TTR)
  - Step B: Reshape pairwise evaluations into 2-row-per-pair format (`paired_df`) and fit
            `statsmodels.discrete.conditional_models.ConditionalLogit` (Xu et al. Eq. 3)
  - Step C: Compute Hardt et al. discrete Equal Opportunity (TPR) gap

Designed to plug directly into the DataFrames from `notebooks/reimplementation.ipynb`:
  - `test_resumes` (resume_id, resume_text, human_summary)
  - `gen` (resume_id, model, text)
  - `ev` (resume_id, model, opponent, version_A, version_B, picked, picked_own)
"""

import re
import numpy as np
import pandas as pd
import statsmodels.discrete.conditional_models as cm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Helpers (fallback if `textstat` or `bert_score` are not installed locally)
# ---------------------------------------------------------------------------

def _count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", word)
    word = re.sub(r"^y", "", word)
    vowels = re.findall(r"[aeiouy]{1,2}", word)
    return max(1, len(vowels))


def _fallback_readability(text: str) -> tuple[float, float]:
    words = re.findall(r"\b\w+\b", text)
    n_words = max(len(words), 1)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    n_sents = max(len(sentences), 1)
    n_sylls = sum(_count_syllables(w) for w in words)
    wps = n_words / n_sents
    spw = n_sylls / n_words
    fre = 206.835 - 1.015 * wps - 84.6 * spw
    fkgl = 0.39 * wps + 11.8 * spw - 15.59
    return fre, fkgl


# ---------------------------------------------------------------------------
# Step A: Calculate BERTScore and Readability Metrics
# ---------------------------------------------------------------------------

def compute_text_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes BERTScore and readability metrics for summaries.
    Assumes df has columns: 'summary_text' and 'resume_experience_body'.
    """
    df = df.copy()
    summaries = df["summary_text"].tolist()
    references = df["resume_experience_body"].tolist()

    # 1. Readability & Structural Metrics
    df["word_count"] = [len(text.split()) for text in summaries]

    try:
        import textstat
        df["fkgl"] = [textstat.flesch_kincaid_grade(text) for text in summaries]
        df["reading_ease"] = [textstat.flesch_reading_ease(text) for text in summaries]
    except ImportError:
        print("[Info] `textstat` not installed; using built-in Flesch/FKGL formulas.")
        readability = [_fallback_readability(text) for text in summaries]
        df["reading_ease"] = [r[0] for r in readability]
        df["fkgl"] = [r[1] for r in readability]

    # Lexical Diversity: Type-Token Ratio (TTR)
    def calculate_ttr(text: str) -> float:
        tokens = text.lower().split()
        return len(set(tokens)) / len(tokens) if tokens else 0.0

    df["ttr"] = [calculate_ttr(text) for text in summaries]

    # 2. Semantic Fidelity: BERTScore (roberta-base, rescaled)
    try:
        from bert_score import score
        P, R, F1 = score(
            cands=summaries,
            refs=references,
            lang="en",
            model_type="roberta-base",
            rescale_with_baseline=True,
            verbose=False,
        )
        df["bert_precision"] = P.numpy()
        df["bert_recall"] = R.numpy()
        df["bert_f1"] = F1.numpy()
    except ImportError:
        print("[Info] `bert_score` not installed; using TF-IDF cosine similarity as a proxy for demo.")
        vec = TfidfVectorizer(stop_words="english")
        tfidf = vec.fit_transform(summaries + references)
        n = len(summaries)
        sims = np.array([cosine_similarity(tfidf[i], tfidf[n + i])[0, 0] for i in range(n)])
        df["bert_precision"] = sims
        df["bert_recall"] = sims
        df["bert_f1"] = sims

    return df


# ---------------------------------------------------------------------------
# Helper: Convert `ev` (1 row per pair) -> `paired_df` (2 rows per pair)
# ---------------------------------------------------------------------------

def build_paired_df(ev_df: pd.DataFrame, summary_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands each pairwise evaluation row in `ev_df` into two rows:
      - Row 1: Candidate A (version_A)
      - Row 2: Candidate B (version_B)
    linked by `pair_id`, with quality features merged from `summary_metrics_df`.
    """
    rows = []
    for pair_id, r in enumerate(ev_df.itertuples(), start=1):
        for candidate in (r.version_A, r.version_B):
            rows.append({
                "pair_id": pair_id,
                "evaluator": r.model,
                "opponent": r.opponent,
                "resume_id": r.resume_id,
                "author": candidate,
                "preferred": int(r.picked == candidate),
                "is_evaluator_llm": int(candidate == r.model),
            })

    paired = pd.DataFrame(rows)
    feature_cols = ["resume_id", "author", "word_count", "fkgl", "reading_ease", "ttr", "bert_f1"]
    return paired.merge(summary_metrics_df[feature_cols], on=["resume_id", "author"], how="left")


# ---------------------------------------------------------------------------
# Step B: Fit Conditional Logistic Regression (Xu et al. Section 3.2.2)
# ---------------------------------------------------------------------------

def calculate_equal_opportunity_bias(paired_df: pd.DataFrame):
    """
    Estimates beta_1 conditioned on resume pair to measure Equal Opportunity bias.
    """
    endog = paired_df["preferred"]
    exog = paired_df[["is_evaluator_llm", "bert_f1", "word_count", "fkgl", "ttr"]]
    groups = paired_df["pair_id"]

    # Fit Conditional Logit (absorbs pair-level fixed effects R_i)
    model = cm.ConditionalLogit(endog=endog, exog=exog, groups=groups)
    results = model.fit(disp=False)

    beta_1 = results.params["is_evaluator_llm"]
    p_val = results.pvalues["is_evaluator_llm"]

    # Equal Opportunity Self-Preference Bias (Xu et al. Eq. 3)
    p_llm = np.exp(beta_1) / (1 + np.exp(beta_1))
    p_alt = 1.0 / (1 + np.exp(beta_1))
    eo_bias = (p_llm - p_alt) * 100  # percentage

    print("\n=== Step B: Equal Opportunity Analysis (Conditional Logit) ===")
    print(results.summary().tables[1])
    print(f"\nEvaluator LLM Coefficient (beta_1): {beta_1:.4f} (p = {p_val:.4e})")
    print(f"Equal Opportunity Self-Preference Bias: {eo_bias:.2f}%")
    return eo_bias, results


# ---------------------------------------------------------------------------
# Step C: Hardt et al. True Positive Rate (TPR) Disparity
# ---------------------------------------------------------------------------

def calculate_hardt_equal_opportunity_gap(df_pairs: pd.DataFrame):
    """
    Computes TPR parity under Hardt et al. (2016) Definition 2.2.
    Quality Y=1 is defined by high semantic fidelity and standard readability.
    """
    df_pairs = df_pairs.copy()
    med_bert = df_pairs["bert_f1"].median()
    df_pairs["is_high_quality"] = (
        (df_pairs["bert_f1"] >= med_bert) & (df_pairs["fkgl"].between(8, 14))
    ).astype(int)

    qualified = df_pairs[df_pairs["is_high_quality"] == 1]
    tpr_llm = qualified[qualified["is_evaluator_llm"] == 1]["preferred"].mean()
    tpr_alt = qualified[qualified["is_evaluator_llm"] == 0]["preferred"].mean()
    eo_gap = tpr_llm - tpr_alt

    print("\n=== Step C: Hardt et al. Equal Opportunity (TPR) ===")
    print(f"Qualified summaries (Y=1): {len(qualified)} / {len(df_pairs)}")
    print(f"TPR for Evaluator-LLM Resumes (Y=1): {tpr_llm:.3f}")
    print(f"TPR for Human/Alt Resumes     (Y=1): {tpr_alt:.3f}")
    print(f"Equal Opportunity Gap (Delta EO):    {eo_gap:.3f} ({eo_gap*100:.1f}%)")
    return eo_gap


# ---------------------------------------------------------------------------
# Demo on a Small Sample Modeled After `reimplementation.ipynb`
# ---------------------------------------------------------------------------

def make_sample_data(n_resumes: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    evaluator = "claude-haiku-5.5"
    opponent = "human"
    roles = [
        ("Accountant", ["GAAP accounting", "reconciliations", "Excel", "ERP", "tax compliance", "audits"]),
        ("Software Engineer", ["Python", "SQL", "REST APIs", "cloud systems", "Agile Scrum", "testing"]),
        ("HR Manager", ["recruiting", "onboarding", "employee relations", "benefits", "HRIS", "compliance"]),
    ]

    summary_rows, eval_rows = [], []
    for i in range(n_resumes):
        rid = str(200000 + i)
        role, skills = roles[i % len(roles)]
        years = int(rng.integers(4, 16))
        body = (
            f"Experience\n{role} ({years} years). Managed projects in {', '.join(skills)}. "
            f"Led teams, improved processes, and reported to leadership. Skills: {', '.join(skills)}."
        )

        # Human summary
        h_skills = ", ".join(rng.choice(skills, size=int(rng.integers(2, 5)), replace=False))
        h_sum = (
            f"Dedicated {role} with {years} years of experience in {h_skills}. "
            f"Strong communicator focused on team success and reliable results."
        )

        # Evaluator LLM summary
        m_skills = ", ".join(rng.choice(skills, size=int(rng.integers(3, 6)), replace=False))
        m_sum = (
            f"Results-driven {role} with {years}+ years of experience spanning {m_skills}. "
            f"Proven record of streamlining workflows and delivering measurable business growth."
        )

        summary_rows.append({"resume_id": rid, "author": opponent, "summary_text": h_sum, "resume_experience_body": body})
        summary_rows.append({"resume_id": rid, "author": evaluator, "summary_text": m_sum, "resume_experience_body": body})

        # Randomized A/B order + simulated evaluator choice (with true self-bias)
        pair = [opponent, evaluator]
        rng.shuffle(pair)
        picked_own = bool(rng.random() < 0.85)
        eval_rows.append({
            "resume_id": rid,
            "model": evaluator,
            "opponent": opponent,
            "version_A": pair[0],
            "version_B": pair[1],
            "picked": evaluator if picked_own else opponent,
            "picked_own": picked_own,
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(eval_rows)


if __name__ == "__main__":
    summaries_df, ev_df = make_sample_data(n_resumes=60, seed=0)

    print("=== Step A: Summary Quality Metrics Preview ===")
    metrics_df = compute_text_quality_features(summaries_df)
    print(metrics_df[["resume_id", "author", "word_count", "fkgl", "reading_ease", "ttr", "bert_f1"]].head(6).round(3).to_string(index=False))

    paired_df = build_paired_df(ev_df, metrics_df)
    print("\n=== Reshaped `paired_df` (2 rows per pair) Preview ===")
    print(paired_df[["pair_id", "resume_id", "author", "preferred", "is_evaluator_llm", "bert_f1", "word_count", "fkgl", "ttr"]].head(6).round(3).to_string(index=False))

    raw_parity = (2 * ev_df["picked_own"].mean() - 1) * 100
    print(f"\nRaw Statistical Parity Bias (unadjusted): {raw_parity:.2f}%")

    calculate_equal_opportunity_bias(paired_df)
    calculate_hardt_equal_opportunity_gap(paired_df)
