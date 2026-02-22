# association_rules_run.py
# -----------------------------------------
# Run Apriori + association rules on the prepared one-hot Excel dataset.
#
# Input:  Dokaznik_pre_association_rules_transformed.xlsx
# Output: dokaznik_rules.xlsx (frequent itemsets + rules)
#
# Install deps:
#   pip install pandas openpyxl mlxtend
# -----------------------------------------

from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


INPUT_PATH = "datasets/Dokaznik_pre_association_rules_transformed.xlsx"
OUTPUT_PATH = "outputs/dokaznik_rules_asoc_new_gen.xlsx"

# --- Hyperparameters (tune these) ---
MIN_SUPPORT = 0.10        # e.g. 0.05–0.20 depending on how many rules you want
MIN_CONFIDENCE = 0.7     # common starting point
MIN_LIFT = 1.5           # filter to keep more interesting rules
MAX_LEN = 3               # max size of itemsets (helps keep runtime reasonable)


def load_onehot_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Ensure numeric 0/1, convert booleans if present
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # Sometimes Excel loads 0/1 as floats; normalize to int 0/1
    df = df.fillna(0)
    df = df.apply(lambda s: (s > 0).astype(int))

    # mlxtend expects True/False or 0/1; either works.
    return df


def main() -> None:
    X = load_onehot_xlsx(INPUT_PATH)
    print(f"Loaded dataset: {X.shape[0]} rows, {X.shape[1]} items")

    # --- Frequent itemsets ---
    itemsets = apriori(
        X,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_LEN
    ).sort_values("support", ascending=False)

    print(f"Frequent itemsets found: {len(itemsets)}")

    # --- Rules ---
    rules = association_rules(
        itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )

    # Add useful helper columns
    rules["antecedents_len"] = rules["antecedents"].apply(len)
    rules["consequents_len"] = rules["consequents"].apply(len)

    # Filter by lift (optional, but recommended)
    rules = rules[rules["lift"] >= MIN_LIFT].copy()

    # Make antecedents/consequents human-readable strings
    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))

    # Sort by most interesting
    rules = rules.sort_values(
        by=["lift", "confidence", "support"],
        ascending=[False, False, False]
    )

    print(f"Rules after filtering: {len(rules)}")

    # --- Save outputs ---
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        itemsets.to_excel(writer, sheet_name="frequent_itemsets", index=False)

        # Keep a nice set of columns for the report
        cols = [
            "antecedents_str", "consequents_str",
            "support", "confidence", "lift",
            "leverage", "conviction",
            "antecedents_len", "consequents_len"
        ]
        rules[cols].to_excel(writer, sheet_name="rules", index=False)

    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()