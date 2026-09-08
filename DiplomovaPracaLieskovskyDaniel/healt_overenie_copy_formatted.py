# healt_overenie_v6_targets_auto_values.py
# ------------------------------------------------------------
# Install:
#   pip install pandas openpyxl mlxtend
# ------------------------------------------------------------

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

INPUT_PATH = "datasets/Dokazník_merged_association_rules_copy_copy.xlsx"
OUTPUT_EXCEL = "outputs/dokaznik_targeted_assoc_output_final.xlsx"
OUTPUT_ONEHOT_XLSX = "outputs/dokaznik_onehot_targeted_final.xlsx"  # optional inspection

MIN_SUPPORT = 0.05       
MAX_LEN = 3
MIN_CONFIDENCE = 0.65

MIN_LIFT_DEFAULT = 1.15

MIN_LIFT_OVERRIDES = {
    ("HealthFeel", "1"): 1.05,  
    ("HealthFeel", "0"): 1.50, 
}

KEEP_MISSING_AS_CATEGORY = False

DROP_COLNAME_PATTERNS = [
    r"\bdatum\b", r"\bdátum\b",
    r"\bcas\b", r"\bčas\b",
    r"\btime\b", r"\bdate\b",
]

TOP_K = 10

TARGETS = [
    {
        "key": "HealthFeel",
        "question_match": "citite po zdravotnej stranke",
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
        "values": {"1", "2", "3"},
    },
    {
        "key": "EnoughPA",
        "question_match": "dostatok fyzickej aktivity",
        "values": {"0", "1"},
    },
]
# ------------------------------------------------


def normalize_text(s: str) -> str:
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
    tx: List[List[str]] = []
    for _, row in df.iterrows():
        items: List[str] = []
        for col in df.columns:
            val = clean_value(row[col])
            if val is None:
                continue
            qnum = col_to_q[col]
            items.append(f"{qnum}||{col}||{val}")
        tx.append(items)
    return tx


def onehot_encode(transactions: List[List[str]]) -> pd.DataFrame:
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_)  # bool


def format_item(item: str) -> str:
    """Convert internal item format 'Qx||Question||value' to 'Qx: Question -> value'."""
    parts = item.split("||", 2)
    if len(parts) == 3:
        qnum, qtext, val = parts
        return f"{qnum}: {qtext} -> {val}"
    return item


def set_to_str(fs) -> str:
    # Use ' AND ' between items to match the requested readable format
    formatted = [format_item(item) for item in fs]
    return " AND ".join(sorted(formatted))


def parse_item(item: str) -> Tuple[str, str, str]:
    parts = item.split("||", 2)
    qnum = parts[0] if len(parts) >= 1 else "Q?"
    qtxt = parts[1] if len(parts) >= 2 else ""
    val = parts[2] if len(parts) >= 3 else ""
    return qnum, qtxt, val


def safe_sheet_name(name: str, maxlen: int = 31) -> str:
    bad = r'[:\\/?*\[\]]'
    name = re.sub(bad, "_", name)
    return name[:maxlen]


def match_target(question_text: str) -> Optional[Dict[str, Any]]:
    qt = normalize_text(question_text)
    for t in TARGETS:
        if normalize_text(t["question_match"]) in qt:
            return t
    return None


def required_lift(target_key: str, consequent_value: str) -> float:
    return MIN_LIFT_OVERRIDES.get((target_key, str(consequent_value)), MIN_LIFT_DEFAULT)


def sort_values_nicely(vals: List[str]) -> List[str]:
    # sort numeric strings numerically, others lexicographically
    def keyfn(x: str):
        return (0, int(x)) if re.fullmatch(r"-?\d+", x) else (1, x)
    return sorted(vals, key=keyfn)


def main() -> None:
    in_path = Path(INPUT_PATH)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    raw = pd.read_excel(INPUT_PATH)

    keep_cols = [c for c in raw.columns if not should_drop_col(c)]
    df = raw[keep_cols].copy()

    qmap = build_question_map(df)
    qmap_df = pd.DataFrame(
        [{"question_number": q, "question_text": t} for q, t in qmap.items()]
    )

    transactions = build_transactions(df, qmap)
    onehot = onehot_encode(transactions)

    with pd.ExcelWriter(OUTPUT_ONEHOT_XLSX, engine="openpyxl") as w:
        onehot.astype(int).to_excel(w, index=False, sheet_name="onehot")

    itemsets = apriori(
        onehot,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_LEN
    ).sort_values("support", ascending=False)

    rules = association_rules(
        itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    ).copy()

    rules["antecedents_len"] = rules["antecedents"].apply(len)
    rules["consequents_len"] = rules["consequents"].apply(len)
    rules["antecedents_str"] = rules["antecedents"].apply(set_to_str)
    rules["consequents_str"] = rules["consequents"].apply(set_to_str)

    single = rules[rules["consequents_len"] == 1].copy()

    diag_rows = [{
        "step": "base_counts",
        "MIN_SUPPORT": MIN_SUPPORT,
        "MIN_CONFIDENCE": MIN_CONFIDENCE,
        "MAX_LEN": MAX_LEN,
        "rules_after_conf": len(rules),
        "single_consequent_rules": len(single),
        "MIN_LIFT_DEFAULT": MIN_LIFT_DEFAULT,
        "MIN_LIFT_OVERRIDES": str(MIN_LIFT_OVERRIDES),
    }]

    if len(single) == 0:
        with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
            qmap_df.to_excel(writer, index=False, sheet_name="question_map")
            pd.DataFrame([{
                "note": "No single-consequent rules after confidence filter. "
                        "Try lowering MIN_SUPPORT / MIN_CONFIDENCE."
            }]).to_excel(writer, index=False, sheet_name="targeted_top10_all")
            pd.DataFrame(diag_rows).to_excel(writer, index=False, sheet_name="diagnostics")
        print("DONE (no single-consequent rules).")
        return

    single["consequent_item"] = single["consequents"].apply(lambda fs: next(iter(fs)))
    single[["consequent_qnum", "consequent_question", "consequent_value"]] = single["consequent_item"].apply(
        lambda x: pd.Series(parse_item(x))
    )

    single["target_key"] = single["consequent_question"].apply(
        lambda qt: (match_target(qt) or {}).get("key")
    )
    targeted_any = single[single["target_key"].notna()].copy()

    allowed_map: Dict[str, Optional[set]] = {t["key"]: (set(t["values"]) if t["values"] is not None else None) for t in TARGETS}

    auto_values_map: Dict[str, List[str]] = {}
    for t in TARGETS:
        key = t["key"]
        if allowed_map[key] is None:
            vals = targeted_any[targeted_any["target_key"] == key]["consequent_value"].astype(str).unique().tolist()
            auto_values_map[key] = sort_values_nicely(vals)
        else:
            auto_values_map[key] = sort_values_nicely(list(allowed_map[key]))

    def allowed_value(row) -> bool:
        key = row["target_key"]
        val = str(row["consequent_value"])

        if allowed_map[key] is None:
            return True
        return val in allowed_map[key]

    targeted_any = targeted_any[targeted_any.apply(allowed_value, axis=1)].copy()

    targeted_any["min_lift_required"] = targeted_any.apply(
        lambda r: required_lift(r["target_key"], str(r["consequent_value"])),
        axis=1
    )
    targeted = targeted_any[targeted_any["lift"] >= targeted_any["min_lift_required"]].copy()

    value_counts_rows = []
    for t in TARGETS:
        key = t["key"]
        before_lift = targeted_any[targeted_any["target_key"] == key].copy()
        after_lift = targeted[targeted["target_key"] == key].copy()

        diag_rows.append({
            "step": "target_counts",
            "target_key": key,
            "values_mode": "AUTO" if allowed_map[key] is None else "FIXED",
            "values_used": ",".join(auto_values_map[key]),
            "count_before_lift": int(len(before_lift)),
            "count_after_lift": int(len(after_lift)),
            "max_lift_before": float(before_lift["lift"].max()) if len(before_lift) else None,
            "max_lift_after": float(after_lift["lift"].max()) if len(after_lift) else None,
        })

        if len(before_lift):
            vc = before_lift["consequent_value"].astype(str).value_counts().to_dict()
            for v in auto_values_map[key]:
                value_counts_rows.append({
                    "target_key": key,
                    "value": v,
                    "rules_count_before_lift": int(vc.get(v, 0)),
                    "min_lift_required": required_lift(key, v),
                    "rules_count_after_lift": int(len(after_lift[after_lift["consequent_value"].astype(str) == v])),
                })
        else:
            for v in auto_values_map[key]:
                value_counts_rows.append({
                    "target_key": key,
                    "value": v,
                    "rules_count_before_lift": 0,
                    "min_lift_required": required_lift(key, v),
                    "rules_count_after_lift": 0,
                })

    targeted = targeted.sort_values(
        by=["lift", "confidence", "support"],
        ascending=[False, False, False]
    )

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
        "min_lift_required",
        "antecedents_len",
    ]
    top10_out = top10[out_cols].copy()

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        qmap_df.to_excel(writer, index=False, sheet_name="question_map")

        itemsets_out = itemsets.copy()
        itemsets_out["itemsets_str"] = itemsets_out["itemsets"].apply(set_to_str)
        itemsets_out = itemsets_out[["support", "itemsets_str"]]
        itemsets_out.to_excel(writer, index=False, sheet_name="frequent_itemsets")

        rules_out_cols = [
            "antecedents_str", "consequents_str",
            "support", "confidence", "lift",
            "leverage", "conviction",
            "antecedents_len", "consequents_len"
        ]
        rules[rules_out_cols].to_excel(writer, index=False, sheet_name="rules_all_conf")

        top10_out.to_excel(writer, index=False, sheet_name="targeted_top10_all")

        pd.DataFrame(diag_rows).to_excel(writer, index=False, sheet_name="diagnostics")
        pd.DataFrame(value_counts_rows).to_excel(writer, index=False, sheet_name="target_value_counts")

        for t in TARGETS:
            key = t["key"]
            for val in auto_values_map[key]:
                sheet = safe_sheet_name(f"{key}_{val}", 31)
                sub = top10_out[
                    (top10_out["target_key"] == key) &
                    (top10_out["consequent_value"].astype(str) == str(val))
                ].copy()
                if len(sub) == 0:
                    pd.DataFrame([{
                        "target_key": key,
                        "value": val,
                        "values_mode": "AUTO" if allowed_map[key] is None else "FIXED",
                        "min_lift_required": required_lift(key, str(val)),
                        "note": (
                            "No rules for this (target,value) after thresholds. "
                            "See 'target_value_counts' and 'diagnostics'."
                        )
                    }]).to_excel(writer, index=False, sheet_name=sheet)
                else:
                    sub.to_excel(writer, index=False, sheet_name=sheet)

    print("DONE")
    print(f"Output: {OUTPUT_EXCEL}")
    print(f"One-hot: {OUTPUT_ONEHOT_XLSX}")
    print(f"Rules after confidence: {len(rules)}")
    print(f"Single-consequent: {len(single)}")
    print(f"Targeted before lift: {len(targeted_any)}")
    print(f"Targeted after lift: {len(targeted)}")
    print(f"Top10 rows: {len(top10_out)}")
    print("See 'diagnostics' + 'target_value_counts' for why some values may still be empty.")


if __name__ == "__main__":
    main()