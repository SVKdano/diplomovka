# dokaznik_assoc_targeted_v4.py
# ------------------------------------------------------------
# Targeted association rules for selected questions + ALL their values,
# with robust matching (diacritics-insensitive) + diagnostics explaining
# why a target (e.g., HealthFeel) may have zero results.
#
# Input:  Dokazník_merged_association_rules.xlsx
# Output: dokaznik_targeted_assoc_output_v4.xlsx
#
# Pipeline:
# 1) Load dataset
# 2) Drop date/time-like columns (by name patterns)
# 3) Number questions (Q1..Qn)
# 4) Build transactions + one-hot encode
# 5) Run Apriori + association rules
# 6) Keep rules where consequent is exactly ONE item (X -> single category)
# 7) Keep only consequents that belong to selected TARGET questions AND allowed values
# 8) For EACH target question-value, output TOP 10 rules by (lift, confidence, support)
# 9) Always produce sheets (even if empty) + add a diagnostics sheet.
#
# Install:
#   pip install pandas openpyxl mlxtend
# ------------------------------------------------------------

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# -------------------- CONFIG --------------------
INPUT_PATH = "datasets/Dokazník_merged_association_rules_copy.xlsx"
OUTPUT_EXCEL = "outputs/dokaznik_targeted_assoc_output_v4.xlsx"
OUTPUT_ONEHOT_XLSX = "outputs/dokaznik_onehot_targeted_v4.xlsx"  # optional inspection

# Parameters (good starting point for ~117 rows).
# If you get "no rules" for some targets, lower MIN_LIFT first.
MIN_SUPPORT = 0.08        
MAX_LEN = 3              
MIN_CONFIDENCE = 0.65
MIN_LIFT = 1.15

KEEP_MISSING_AS_CATEGORY = False  # True -> treat NaN as explicit "NA"

DROP_COLNAME_PATTERNS = [
    r"\bdátum\b", r"\bdatum\b",
    r"\bčas\b", r"\bcas\b",
    r"\btime\b", r"\bdate\b",
]

TOP_K = 10  # top rules per (question,value)

# ---- TARGET QUESTIONS + ALLOWED VALUES ----
# Use short, distinctive phrases; matching is diacritics-insensitive.
TARGETS = [
    {
        "key": "HealthFeel",
        "question_match": "citite po zdravotnej stranke",  # robust (shorter than full header)
        "values": {"0", "1"},
    },
    {
        "key": "SleepWork",
        "question_match": "spanok cez pracovny den",
        "values": {"0", "1", "2"},
    },
    {
        "key": "SleepWeekend",
        "question_match": "spanok cez vikendovy den",
        "values": {"0", "1", "2"},
    },
    {
        "key": "MotivationPA",
        "question_match": "som motivovany vykonavat fyzicku aktivitu",
        "values": {"1", "2", "3", "4", "5"},
    },
    {
        "key": "EnoughPA",
        "question_match": "dostatok fyzickej aktivity",
        "values": {"0", "1"},
    },
]
# ------------------------------------------------


def normalize_text(s: str) -> str:
    """Lowercase + remove diacritics + collapse spaces."""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def should_drop_col(colname: str) -> bool:
    low = normalize_text(colname)
    return any(re.search(pat, low) for pat in DROP_COLNAME_PATTERNS)


def clean_value(v) -> Optional[str]:
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
    """Return target dict if question_text matches a target (substring, normalized)."""
    qt = normalize_text(question_text)
    for t in TARGETS:
        if normalize_text(t["question_match"]) in qt:
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

    # Diagnostics base
    diag_rows = []
    diag_rows.append({
        "step": "after_filters",
        "MIN_SUPPORT": MIN_SUPPORT,
        "MIN_CONFIDENCE": MIN_CONFIDENCE,
        "MIN_LIFT": MIN_LIFT,
        "MAX_LEN": MAX_LEN,
        "rules_count": len(rules),
        "single_consequent_rules_count": len(single),
    })

    # If none, still output workbook
    if len(single) == 0:
        with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
            qmap_df.to_excel(writer, index=False, sheet_name="question_map")
            pd.DataFrame([{
                "note": "No single-consequent rules found after filtering. "
                        "Try lowering MIN_LIFT / MIN_SUPPORT / MIN_CONFIDENCE."
            }]).to_excel(writer, index=False, sheet_name="targeted_top10_all")
            pd.DataFrame(diag_rows).to_excel(writer, index=False, sheet_name="diagnostics")
        print("DONE (no single-consequent rules).")
        return

    # Extract consequent item info
    single["consequent_item"] = single["consequents"].apply(lambda fs: next(iter(fs)))
    single[["consequent_qnum", "consequent_question", "consequent_value"]] = single["consequent_item"].apply(
        lambda x: pd.Series(parse_item(x))
    )
    single["consequent_question_norm"] = single["consequent_question"].apply(normalize_text)

    # Mark targets
    single["target_key"] = single["consequent_question"].apply(
        lambda qt: (match_target(qt) or {}).get("key")
    )
    targeted_any = single[single["target_key"].notna()].copy()

    # Filter allowed values per target
    def allowed(row) -> bool:
        t = match_target(row["consequent_question"])
        if not t:
            return False
        return str(row["consequent_value"]) in t["values"]

    targeted = targeted_any[targeted_any.apply(allowed, axis=1)].copy()

    # Add per-target diagnostics:
    for t in TARGETS:
        key = t["key"]
        # matched by question text
        matched = targeted_any[targeted_any["target_key"] == key].copy()
        # matched + allowed values
        matched_allowed = targeted[targeted["target_key"] == key].copy()

        diag_rows.append({
            "step": "target_match_counts",
            "target_key": key,
            "question_match": t["question_match"],
            "allowed_values": ",".join(sorted(list(t["values"]))),
            "matched_rules_count": len(matched),
            "matched_allowed_values_count": len(matched_allowed),
            "max_lift_in_matched": float(matched["lift"].max()) if len(matched) else None,
            "max_conf_in_matched": float(matched["confidence"].max()) if len(matched) else None,
            "max_support_in_matched": float(matched["support"].max()) if len(matched) else None,
        })

    # Sort for "best"
    targeted = targeted.sort_values(
        by=["lift", "confidence", "support"],
        ascending=[False, False, False]
    )

    # Take TOP 10 per (target_key, value)
    top10 = (
        targeted.groupby(["target_key", "consequent_value"], group_keys=False)
        .head(TOP_K)
        .copy()
    )

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

    # For convenience: top10 per target question across all values
    per_q_top10 = (
        targeted.groupby(["target_key"], group_keys=False)
        .head(TOP_K)
        .copy()
    )
    per_q_out = per_q_top10[out_cols].copy()

    # Export workbook
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        qmap_df.to_excel(writer, index=False, sheet_name="question_map")

        # frequent itemsets readable
        itemsets_out = itemsets.copy()
        itemsets_out["itemsets_str"] = itemsets_out["itemsets"].apply(set_to_str)
        itemsets_out = itemsets_out[["support", "itemsets_str"]]
        itemsets_out.to_excel(writer, index=False, sheet_name="frequent_itemsets")

        # rules after filters
        rules_out_cols = [
            "antecedents_str", "consequents_str",
            "support", "confidence", "lift",
            "leverage", "conviction",
            "antecedents_len", "consequents_len"
        ]
        rules[rules_out_cols].to_excel(writer, index=False, sheet_name="rules_all_filtered")

        # stacked top10 for all target values
        top10_out.to_excel(writer, index=False, sheet_name="targeted_top10_all")

        # diagnostics sheet
        pd.DataFrame(diag_rows).to_excel(writer, index=False, sheet_name="diagnostics")

        # One sheet per target question (always)
        for t in TARGETS:
            key = t["key"]
            sheet = safe_sheet_name(f"{key}_TOP10", 31)
            sub = per_q_out[per_q_out["target_key"] == key].copy()
            if len(sub) == 0:
                pd.DataFrame([{
                    "target_key": key,
                    "question_match": t["question_match"],
                    "note": (
                        "No rules found for this target question after filtering. "
                        "Check 'diagnostics' sheet; try lowering MIN_LIFT or MIN_SUPPORT."
                    )
                }]).to_excel(writer, index=False, sheet_name=sheet)
            else:
                sub.to_excel(writer, index=False, sheet_name=sheet)

        # One sheet per target question-value (always)
        for t in TARGETS:
            key = t["key"]
            for val in sorted(t["values"], key=lambda x: int(x) if x.isdigit() else x):
                sheet = safe_sheet_name(f"{key}_{val}", 31)
                sub = top10_out[
                    (top10_out["target_key"] == key) &
                    (top10_out["consequent_value"].astype(str) == str(val))
                ].copy()
                if len(sub) == 0:
                    pd.DataFrame([{
                        "target_key": key,
                        "value": val,
                        "question_match": t["question_match"],
                        "note": (
                            "No rules found for this target value after filtering. "
                            "If diagnostics shows 'matched_rules_count' > 0, then filters/values caused it."
                        )
                    }]).to_excel(writer, index=False, sheet_name=sheet)
                else:
                    sub.to_excel(writer, index=False, sheet_name=sheet)

    print("DONE")
    print(f"Saved one-hot: {OUTPUT_ONEHOT_XLSX}")
    print(f"Saved output: {OUTPUT_EXCEL}")
    print(f"Rules after filters: {len(rules)}")
    print(f"Single-consequent rules: {len(single)}")
    print(f"Targeted rules (matched+allowed): {len(targeted)}")
    print(f"Top10 rows total: {len(top10_out)}")
    print("See 'diagnostics' sheet for why a target might have 0 rules.")


if __name__ == "__main__":
    main()