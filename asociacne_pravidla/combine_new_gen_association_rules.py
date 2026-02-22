# dokaznik_end_to_end_assoc_rules_v2.py
# ------------------------------------------------------------
# End-to-end pipeline (single script):
# 1) Read Dokazník_merged_association_rules.xlsx
# 2) Drop date/time-like columns (by name patterns)
# 3) Number questions (Q1..Qn)
# 4) Build transaction (one-hot) format for Apriori/FP-Growth
# 5) Run Apriori + association rules
# 6) Keep rules with single-item consequent (X -> single category)
# 7) For each QUESTION (Q#), select TOP 10 rules by (lift, confidence, support)
# 8) Export one Excel workbook:
#    - question_map
#    - frequent_itemsets
#    - rules_all (filtered by confidence + lift)
#    - top10_all_questions
#    - one sheet per question ALWAYS (Q001, Q002, ...); if no rules -> note
#
# Install:
#   pip install pandas openpyxl mlxtend
# ------------------------------------------------------------

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# -------------------- CONFIG --------------------
INPUT_PATH = "datasets/Dokazník_merged_association_rules.xlsx"

OUTPUT_EXCEL = "outputs/dokaznik_assoc_output_v2.xlsx"
OUTPUT_ONEHOT_XLSX = "outputs/dokaznik_onehot_v2.xlsx"  # optional inspection

# Recommended defaults for ~117 rows
MIN_SUPPORT = 0.10        # ~12 respondents if N≈117
MAX_LEN = 3              # keep patterns interpretable
MIN_CONFIDENCE = 0.70
MIN_LIFT = 1.30

# Missing values handling
KEEP_MISSING_AS_CATEGORY = False  # True -> treat NaN as explicit "NA" category

# Drop columns that look like date/time by name (extend if needed)
DROP_COLNAME_PATTERNS = [
    r"\bdátum\b", r"\bdatum\b",
    r"\bčas\b", r"\bcas\b",
    r"\btime\b", r"\bdate\b",
]
# ------------------------------------------------


def should_drop_col(colname: str) -> bool:
    low = str(colname).strip().lower()
    return any(re.search(pat, low) for pat in DROP_COLNAME_PATTERNS)


def clean_value(v) -> str | None:
    """Convert a cell value to a stable category string."""
    if pd.isna(v):
        return "NA" if KEEP_MISSING_AS_CATEGORY else None
    s = str(v).strip()
    if s == "":
        return "NA" if KEEP_MISSING_AS_CATEGORY else None
    # Normalize Excel numeric strings like "1.0" -> "1"
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def build_question_map(df: pd.DataFrame) -> Dict[str, str]:
    """Assign Q1..Qn to each column (in order)."""
    return {f"Q{i}": col for i, col in enumerate(df.columns, start=1)}


def build_transactions(df: pd.DataFrame, qmap: Dict[str, str]) -> List[List[str]]:
    """
    Each row becomes a list of items.
    Item format: Qn||question_text||value
    """
    col_to_q = {col: q for q, col in qmap.items()}
    transactions: List[List[str]] = []

    for _, row in df.iterrows():
        items: List[str] = []
        for col in df.columns:
            qnum = col_to_q[col]
            val = clean_value(row[col])
            if val is None:
                continue
            items.append(f"{qnum}||{col}||{val}")
        transactions.append(items)

    return transactions


def onehot_encode(transactions: List[List[str]]) -> pd.DataFrame:
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_)  # bool matrix


def set_to_str(fs) -> str:
    return ", ".join(sorted(list(fs)))


def parse_qnum(item: str) -> str:
    parts = item.split("||", 2)
    return parts[0] if parts else "Q?"


def parse_question(item: str) -> str:
    parts = item.split("||", 2)
    return parts[1] if len(parts) >= 2 else ""


def parse_value(item: str) -> str:
    parts = item.split("||", 2)
    return parts[2] if len(parts) >= 3 else ""


def main() -> None:
    in_path = Path(INPUT_PATH)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    # 1) Load
    raw = pd.read_excel(INPUT_PATH)

    # 2) Drop date/time-like columns
    keep_cols = [c for c in raw.columns if not should_drop_col(c)]
    df = raw[keep_cols].copy()

    # 3) Question numbering
    qmap = build_question_map(df)
    qmap_df = pd.DataFrame(
        [{"question_number": q, "question_text": t} for q, t in qmap.items()]
    )
    qnum_to_text = dict(qmap)  # Q# -> text

    # 4) Transactions + one-hot
    transactions = build_transactions(df, qmap)
    onehot = onehot_encode(transactions)

    # Save one-hot for inspection (optional)
    with pd.ExcelWriter(OUTPUT_ONEHOT_XLSX, engine="openpyxl") as w:
        onehot.astype(int).to_excel(w, index=False, sheet_name="onehot")

    # 5) Frequent itemsets
    itemsets = apriori(
        onehot,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_LEN
    ).sort_values("support", ascending=False)

    # 6) Rules (confidence threshold)
    rules = association_rules(
        itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    ).copy()

    # Add helpful columns
    rules["antecedents_len"] = rules["antecedents"].apply(len)
    rules["consequents_len"] = rules["consequents"].apply(len)
    rules["antecedents_str"] = rules["antecedents"].apply(set_to_str)
    rules["consequents_str"] = rules["consequents"].apply(set_to_str)

    # Filter by lift
    rules = rules[rules["lift"] >= MIN_LIFT].copy()

    # 7) Build TOP10 per question for single-item consequent (X -> one category)
    single = rules[rules["consequents_len"] == 1].copy()

    if len(single) > 0:
        single["consequent_item"] = single["consequents"].apply(lambda fs: next(iter(fs)))
        single["consequent_qnum"] = single["consequent_item"].apply(parse_qnum)
        single["consequent_question"] = single["consequent_item"].apply(parse_question)
        single["consequent_value"] = single["consequent_item"].apply(parse_value)

        # Sort for "best"
        single = single.sort_values(
            by=["lift", "confidence", "support"],
            ascending=[False, False, False]
        )

        # Take top 10 per question number
        top10_all = (
            single.groupby("consequent_qnum", group_keys=False)
            .head(10)
            .copy()
        )
    else:
        top10_all = pd.DataFrame(columns=[
            "consequent_qnum", "consequent_question", "consequent_value",
            "antecedents_str", "support", "confidence", "lift"
        ])

    # Common columns for output
    top10_cols = [
        "consequent_qnum", "consequent_question", "consequent_value",
        "antecedents_str", "support", "confidence", "lift"
    ]

    # Prepare group lookup for per-question sheets
    top10_by_q = {q: g for q, g in top10_all.groupby("consequent_qnum")} if len(top10_all) else {}

    # 8) Export everything to one workbook
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        # question map
        qmap_df.to_excel(writer, index=False, sheet_name="question_map")

        # frequent itemsets (readable)
        itemsets_out = itemsets.copy()
        itemsets_out["itemsets_str"] = itemsets_out["itemsets"].apply(set_to_str)
        itemsets_out = itemsets_out[["support", "itemsets_str"]]
        itemsets_out.to_excel(writer, index=False, sheet_name="frequent_itemsets")

        # all rules (readable)
        rules_out_cols = [
            "antecedents_str", "consequents_str",
            "support", "confidence", "lift",
            "leverage", "conviction",
            "antecedents_len", "consequents_len"
        ]
        rules[rules_out_cols].to_excel(writer, index=False, sheet_name="rules_all")

        # top10 overview
        top10_all[top10_cols].to_excel(writer, index=False, sheet_name="top10_all_questions")

        # ALWAYS create a sheet per question (Q001, Q002, ...)
        for qnum, qtext in qnum_to_text.items():
            q_index = int(qnum[1:])  # "Q12" -> 12
            sheet_name = f"Q{q_index:03d}"  # Q012 (always consistent)

            if qnum in top10_by_q:
                grp_out = top10_by_q[qnum][top10_cols].copy()
                grp_out.to_excel(writer, index=False, sheet_name=sheet_name)
            else:
                empty_df = pd.DataFrame([{
                    "question_number": qnum,
                    "question_text": qtext,
                    "note": (
                        "No X -> (single consequent item) rules found for this question "
                        "after filtering. Try lowering MIN_LIFT / MIN_SUPPORT / MIN_CONFIDENCE."
                    )
                }])
                empty_df.to_excel(writer, index=False, sheet_name=sheet_name)

    print("DONE")
    print(f"Saved one-hot: {OUTPUT_ONEHOT_XLSX}")
    print(f"Saved output: {OUTPUT_EXCEL}")
    print(f"Questions: {len(qmap)}")
    print(f"Rules after filters (confidence + lift): {len(rules)}")
    print(f"Single-consequent rules: {len(single)}")
    print(f"Top10 rows total: {len(top10_all)}")


if __name__ == "__main__":
    main()