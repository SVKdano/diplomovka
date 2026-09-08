# ===============================================
# ASOCIAČNÉ PRAVIDLÁ – CIEĽ: SPÁNOK (pracovné dni)
# ===============================================

import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# -----------------------------
# 1️⃣ Nastavenie súboru
# -----------------------------
INPUT_PATH = "datasets/dokaznik_merged.csv"

# Cieľová premenná (konzekvent)
SLEEP_COL = "Koľko hodín v priemere spíte cez pracovný deň?"

# Prahy
MIN_SUPPORT = 0.15
MIN_CONFIDENCE = 0.70
MAX_LEN = 3

# -----------------------------
# 2️⃣ Načítanie dát
# -----------------------------
df = pd.read_csv(INPUT_PATH, sep=";", engine="python")

# Voliteľne vyhoď časové stĺpce
for col in ["Dátum", "Čas"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Konverzia čísel s desatinnou čiarkou
for col in df.columns:
    if df[col].dtype == "object":
        s = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().mean() > 0.6:
            df[col] = numeric

# -----------------------------
# 3️⃣ Prevod riadkov na transakcie
# -----------------------------
transactions = []

for _, row in df.iterrows():
    items = []
    for col in df.columns:
        val = row[col]
        if pd.isna(val):
            continue

        # čísla s malým počtom unikátov ber ako kategórie
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= 10:
                if float(val).is_integer():
                    val = int(val)
                items.append(f"{col}={val}")
            else:
                # veľa unikátnych hodnôt → intervaly (kvartily)
                bins = pd.qcut(df[col], q=4, duplicates="drop")
                items.append(f"{col}∈{bins.loc[row.name]}")
        else:
            items.append(f"{col}={val}")

    transactions.append(list(set(items)))

# -----------------------------
# 4️⃣ One-hot encoding
# -----------------------------
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
onehot = pd.DataFrame(te_array, columns=te.columns_)

# -----------------------------
# 5️⃣ Frequent itemsets
# -----------------------------
itemsets = apriori(
    onehot,
    min_support=MIN_SUPPORT,
    use_colnames=True,
    max_len=MAX_LEN
)

# -----------------------------
# 6️⃣ Association rules
# -----------------------------
rules = association_rules(
    itemsets,
    metric="confidence",
    min_threshold=MIN_CONFIDENCE
)

# Odstráň slabé závislosti
rules = rules[rules["lift"] > 1.0].copy()

# -----------------------------
# 7️⃣ Filtrovanie – len pravidlá kde konzekvent = spánok
# -----------------------------
rules_sleep = rules[
    rules["consequents"].apply(
        lambda x: any(item.startswith(SLEEP_COL) for item in x)
    )
].copy()

# Formátovanie výstupu
rules_sleep["antecedents"] = rules_sleep["antecedents"].apply(lambda s: " & ".join(sorted(list(s))))
rules_sleep["consequents"] = rules_sleep["consequents"].apply(lambda s: " & ".join(sorted(list(s))))

rules_sleep = rules_sleep.sort_values(
    ["lift", "confidence", "support"],
    ascending=[False, False, False]
)

# -----------------------------
# 8️⃣ Uloženie
# -----------------------------
itemsets.to_csv("frequent_itemsets_mlxtend.csv", index=False)
rules_sleep.to_csv("association_rules_SLEEP_ONLY.csv", index=False)

# -----------------------------
# 9️⃣ Výpis top pravidiel
# -----------------------------
print("Hotovo ✅")
print(f"Itemsets: {len(itemsets)}")
print(f"Sleep rules: {len(rules_sleep)}")

if len(rules_sleep) > 0:
    print("\nTop 10 pravidiel (→ spánok):\n")
    print(
        rules_sleep[
            ["antecedents", "consequents", "support", "confidence", "lift"]
        ].head(10).to_string(index=False)
    )
else:
    print("\n⚠️ Nenašli sa pravidlá pre spánok. Skús znížiť MIN_SUPPORT alebo MIN_CONFIDENCE.")
