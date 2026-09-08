# dokaznik_assoc_targeted_v3.py
# ------------------------------------------------------------
# Targeted association rules for selected questions + ALL their values.
#
# Input:  Dokazník_merged_association_rules.xlsx
# Output: dokaznik_targeted_assoc_output.xlsx
#
# What it does:
# 1) Load original dataset
# 2) Drop date/time-like columns (by name patterns)
# 3) Number questions (Q1..Qn)
# 4) Build transactions and one-hot encode
# 5) Run Apriori + association rules
# 6) Keep rules where consequent is exactly ONE item (X -> single category)
# 7) Keep only consequents that belong to selected TARGET questions
# 8) For EACH target question AND EACH target value (category), output TOP 10 rules by lift
# 9) Export:
#    - question_map
#    - frequent_itemsets
#    - rules_all_filtered (confidence + lift)
#    - targeted_top10_all (all targets stacked)
#    - one sheet per target question
#    - one sheet per target question-value (e.g., SleepWork_1)
#
# Install:
#   pip install pandas openpyxl mlxtend
# ------------------------------------------------------------

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# -------------------- CONFIG --------------------
INPUT_PATH = "datasets/Dokazník_merged_association_rules.xlsx"
OUTPUT_EXCEL = "outputs/dokaznik_targeted_assoc_output.xlsx"
OUTPUT_ONEHOT_XLSX = "outputs/dokaznik_onehot_targeted.xlsx"  # optional

# Parameters (good starting point for ~117 rows)
MIN_SUPPORT = 0.8        # ~12 respondents if N≈117
MAX_LEN = 3              # interpretability
MIN_CONFIDENCE = 0.60
MIN_LIFT = 1.2

KEEP_MISSING_AS_CATEGORY = False  # True -> treat NaN as explicit "NA"

DROP_COLNAME_PATTERNS = [
    r"\bdátum\b", r"\bdatum\b",
    r"\bčas\b", r"\bcas\b",
    r"\btime\b", r"\bdate\b",
]

# ---- TARGET QUESTIONS + ALLOWED VALUES ----
# Write the question text EXACTLY as in the Excel header (or a distinctive substring).
# The script matches by substring (case-insensitive) and then filters values.

TARGETS = [
    {
        "key": "HealthFeel",
        "question_match": "Pocit po zdravotnej stránke",
        "values": {"1", "2", "3"},
    },
    {
        "key": "SleepWork",
        "question_match": "Spánok cez pracovný deň",
        "values": {"0", "1", "2"},
    },
    {
        "key": "SleepWeekend",
        "question_match": "Spánok cez víkendový deň",
        "values": {"0", "1", "2"},
    },
    {
        "key": "MotivationPA",
        "question_match": "Som motivovaný vykonávať fyzickú aktivitu",
        "values": {"1", "2", "3", "4", "5"},
    },
    {
        "key": "EnoughPA",
        "question_match": "Dostatok fyzickej aktivity",
        "values": {"0", "1"},
    },
]

TOP_K = 10  # top rules per (question,value)
# ------------------------------------------------


def should_drop_col(colname: str) -> bool:
    low = str(colname).strip().lower()
    return any(re.search(pat, low) for pat in DROP_COLNAME_PATTERNS)


def clean_value(v) -> Optional[str]:
    if pd.isna(v):
        return "NA" if KEEP_MISSING_AS_CATEGORY else None
    s = str(v).strip()
    if s == "":
        return "NA" if KEEP_MISSING_AS_CATEGORY else None
    # Normalize "1.0" -> "1"
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def build_question_map(df: pd.DataFrame) -> Dict[str, str]:
    return {f"Q{i}": col for i, col in enumerate(df.columns, start=1)}


def build_transactions(df: pd.DataFrame, qmap: Dict[str, str]) -> List[List[str]]:
    col_to_q = {col: q for q, col in qmap.items()}
    transactions: List[List[str]] = []

    for _, row in df.iterrows():
        items: List[str] = []
        for col in df.columns:
            val = clean_value(row[col])
            if val is None:
                continue
            qnum = col_to_q[col]
            # Item format: Qn||question_text||value
            items.append(f"{qnum}||{col}||{val}")
        transactions.append(items)

    return transactions


def onehot_encode(transactions: List[List[str]]) -> pd.DataFrame:
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_)  # bool


def set_to_str(fs) -> str:
    return ", ".join(sorted(list(fs)))


def parse_item(item: str) -> Tuple[str, str, str]:
    # returns (qnum, question_text, value)
    parts = item.split("||", 2)
    qnum = parts[0] if len(parts) >= 1 else "Q?"
    qtxt = parts[1] if len(parts) >= 2 else ""
    val = parts[2] if len(parts) >= 3 else ""
    return qnum, qtxt, val


def safe_sheet_name(name: str, maxlen: int = 31) -> str:
    # Excel sheet names cannot contain: : \ / ? * [ ]
    bad = r'[:\\/?*\[\]]'
    name = re.sub(bad, "_", name)
    return name[:maxlen]


def match_target(question_text: str) -> Optional[Dict]:
    """Return target dict if question_text matches a target (substring, case-insensitive)."""
    qt_low = question_text.lower()
    for t in TARGETS:
        if t["question_match"].lower() in qt_low:
            return t
    return None


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

    # 4) Transactions + onehot
    transactions = build_transactions(df, qmap)
    onehot = onehot_encode(transactions)

    # Optional: save onehot for inspection
    with pd.ExcelWriter(OUTPUT_ONEHOT_XLSX, engine="openpyxl") as w:
        onehot.astype(int).to_excel(w, index=False, sheet_name="onehot")

    # 5) Frequent itemsets
    itemsets = apriori(
        onehot,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_LEN
    ).sort_values("support", ascending=False)

    # 6) Rules
    rules = association_rules(
        itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    ).copy()

    # Add readable columns + lengths
    rules["antecedents_len"] = rules["antecedents"].apply(len)
    rules["consequents_len"] = rules["consequents"].apply(len)
    rules["antecedents_str"] = rules["antecedents"].apply(set_to_str)
    rules["consequents_str"] = rules["consequents"].apply(set_to_str)

    # Filter by lift
    rules = rules[rules["lift"] >= MIN_LIFT].copy()

    # Only X -> single category
    single = rules[rules["consequents_len"] == 1].copy()

    # If none, still output workbook with notes
    if len(single) == 0:
        with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
            qmap_df.to_excel(writer, index=False, sheet_name="question_map")
            pd.DataFrame([{
                "note": "No single-consequent rules found after filtering. "
                        "Try lowering MIN_SUPPORT / MIN_CONFIDENCE / MIN_LIFT."
            }]).to_excel(writer, index=False, sheet_name="targeted_top10_all")
        print("DONE (no single-consequent rules).")
        return

    # Extract consequent item info
    single["consequent_item"] = single["consequents"].apply(lambda fs: next(iter(fs)))
    single[["consequent_qnum", "consequent_question", "consequent_value"]] = single["consequent_item"].apply(
        lambda x: pd.Series(parse_item(x))
    )

    # Determine which are targets + values
    single["target_key"] = single["consequent_question"].apply(
        lambda qt: (match_target(qt) or {}).get("key")
    )
    single["is_target"] = single["target_key"].notna()

    targeted = single[single["is_target"]].copy()

    # Filter to allowed values per target
    def value_allowed(row) -> bool:
        t = match_target(row["consequent_question"])
        if not t:
            return False
        return str(row["consequent_value"]) in t["values"]

    targeted = targeted[targeted.apply(value_allowed, axis=1)].copy()

    # Sort for "best"
    targeted = targeted.sort_values(
        by=["lift", "confidence", "support"],
        ascending=[False, False, False]
    )

    # Take TOP 10 per (target_key, value)
    targeted["target_value_key"] = targeted["target_key"] + "_" + targeted["consequent_value"].astype(str)

    top10 = (
        targeted.groupby(["target_key", "consequent_value"], group_keys=False)
        .head(TOP_K)
        .copy()
    )

    # Output columns
    out_cols = [
        "target_key",
        "consequent_question",
        "consequent_value",
        "antecedents_str",
        "support",
        "confidence",
        "lift",
        "antecedents_len",
    ]
    top10_out = top10[out_cols].copy()

    # Convenience: also create per-question aggregated TOP10 (across all values)
    # (optional; can be helpful in report)
    per_question_top10 = (
        targeted.groupby(["target_key"], group_keys=False)
        .head(TOP_K)
        .copy()
    )
    per_question_out = per_question_top10[out_cols].copy()

    # 7) Export workbook
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        qmap_df.to_excel(writer, index=False, sheet_name="question_map")

        # frequent itemsets (readable)
        itemsets_out = itemsets.copy()
        itemsets_out["itemsets_str"] = itemsets_out["itemsets"].apply(set_to_str)
        itemsets_out = itemsets_out[["support", "itemsets_str"]]
        itemsets_out.to_excel(writer, index=False, sheet_name="frequent_itemsets")

        # all rules after filters
        rules_out_cols = [
            "antecedents_str", "consequents_str",
            "support", "confidence", "lift",
            "leverage", "conviction",
            "antecedents_len", "consequents_len"
        ]
        rules[rules_out_cols].to_excel(writer, index=False, sheet_name="rules_all_filtered")

        # stacked top10 for all targets/values
        top10_out.to_excel(writer, index=False, sheet_name="targeted_top10_all")

        # optional: top10 per target question (across any value)
        per_question_out.to_excel(writer, index=False, sheet_name="targeted_top10_per_question")

        # One sheet per target question
        for t in TARGETS:
            key = t["key"]
            sheet = safe_sheet_name(f"{key}_TOP10", 31)
            sub = per_question_out[per_question_out["target_key"] == key].copy()
            if len(sub) == 0:
                pd.DataFrame([{
                    "target_key": key,
                    "question_match": t["question_match"],
                    "note": "No rules found for this target question after filtering."
                }]).to_excel(writer, index=False, sheet_name=sheet)
            else:
                sub.to_excel(writer, index=False, sheet_name=sheet)

        # One sheet per target question-value (e.g., SleepWork_1)
        for t in TARGETS:
            key = t["key"]
            for val in sorted(t["values"], key=lambda x: int(x) if x.isdigit() else x):
                sheet = safe_sheet_name(f"{key}_{val}", 31)
                sub = top10_out[(top10_out["target_key"] == key) & (top10_out["consequent_value"].astype(str) == str(val))].copy()
                if len(sub) == 0:
                    pd.DataFrame([{
                        "target_key": key,
                        "value": val,
                        "question_match": t["question_match"],
                        "note": "No rules found for this target value after filtering."
                    }]).to_excel(writer, index=False, sheet_name=sheet)
                else:
                    sub.to_excel(writer, index=False, sheet_name=sheet)

    print("DONE")
    print(f"Saved one-hot: {OUTPUT_ONEHOT_XLSX}")
    print(f"Saved output: {OUTPUT_EXCEL}")
    print(f"Rules after filters: {len(rules)}")
    print(f"Single-consequent rules: {len(single)}")
    print(f"Targeted single-consequent rules: {len(targeted)}")
    print(f"Top10 rows (per target,value): {len(top10_out)}")


if __name__ == "__main__":
    main()