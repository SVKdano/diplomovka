# healt_overenie_formatted.py
# ------------------------------------------------------------
# Association rules with formatted readable output:
# Q8||Question||0  ->  Q8: Question -> 0
# ------------------------------------------------------------

from __future__ import annotations
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# -------------------- CONFIG --------------------
INPUT_PATH = "datasets/Dokazník_merged_association_rules_copy_copy.xlsx"
OUTPUT_EXCEL = "outputs/dokaznik_assoc_output_formatted_FINAL.xlsx"

MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.60
MAX_LEN = 3
MIN_LIFT = 1.15
# ------------------------------------------------


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s.lower()).strip()


def clean_value(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s == "":
        return None
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def build_question_map(df: pd.DataFrame):
    return {f"Q{i}": col for i, col in enumerate(df.columns, start=1)}


def build_transactions(df: pd.DataFrame, qmap):
    col_to_q = {col: q for q, col in qmap.items()}
    transactions = []

    for _, row in df.iterrows():
        items = []
        for col in df.columns:
            val = clean_value(row[col])
            if val is None:
                continue
            qnum = col_to_q[col]
            items.append(f"{qnum}||{col}||{val}")
        transactions.append(items)

    return transactions


def onehot_encode(transactions):
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_)


# --------- NEW: Pretty formatting ---------

def format_item(item: str) -> str:
    parts = item.split("||", 2)
    if len(parts) == 3:
        qnum, qtext, val = parts
        return f"{qnum}: {qtext} -> {val}"
    return item


def set_to_str(fs) -> str:
    formatted = [format_item(item) for item in fs]
    return ", ".join(sorted(formatted))


def parse_item(item: str) -> Tuple[str, str, str]:
    parts = item.split("||", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return "", "", ""


# ------------------------------------------------

def main():
    df = pd.read_excel(INPUT_PATH)

    qmap = build_question_map(df)
    transactions = build_transactions(df, qmap)
    onehot = onehot_encode(transactions)

    itemsets = apriori(
        onehot,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_LEN
    )

    rules = association_rules(
        itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )

    rules = rules[rules["lift"] >= MIN_LIFT].copy()

    rules["antecedents_str"] = rules["antecedents"].apply(set_to_str)
    rules["consequents_str"] = rules["consequents"].apply(set_to_str)

    rules["antecedents_len"] = rules["antecedents"].apply(len)
    rules["consequents_len"] = rules["consequents"].apply(len)

    output_cols = [
        "antecedents_str",
        "consequents_str",
        "support",
        "confidence",
        "lift",
        "antecedents_len",
        "consequents_len"
    ]

    rules_out = rules[output_cols].sort_values(
        by=["lift", "confidence", "support"],
        ascending=[False, False, False]
    )

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        rules_out.to_excel(writer, index=False, sheet_name="association_rules")

    print("DONE")
    print(f"Rules generated: {len(rules_out)}")
    print(f"Output saved to: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()