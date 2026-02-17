# ===============================================
# ASOCIAČNÉ PRAVIDLÁ – CIEĽ: SPÁNOK (KATEGÓRIE)
# ===============================================

import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# -----------------------------
# 1️⃣ Nastavenie súboru
# -----------------------------
INPUT_PATH = "datasets/dokaznik_merged.csv"

# Pôvodný stĺpec so spánkom
SLEEP_COL = "Koľko hodín v priemere spíte cez pracovný deň?"

# Nový kategorizovaný stĺpec (konzekvent)
SLEEP_CAT_COL = "SLEEP_CAT"

# Kategórie (upraviť podľa potreby)
# <=6, 7-8, >=9
def sleep_to_cat(x):
    if pd.isna(x):
        return np.nan
    try:
        x = float(x)
    except Exception:
        return np.nan

    if x <= 6:
        return "LOW_<=6"
    elif x <= 8:
        return "NORMAL_7_8"
    else:
        return "HIGH_>=9"

# Prahy
MIN_SUPPORT = 0.10
MIN_CONFIDENCE = 0.60
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
# 3️⃣ Vytvor kategóriu spánku
# -----------------------------
df[SLEEP_CAT_COL] = df[SLEEP_COL].apply(sleep_to_cat)

# Ak chceš, môžeš pôvodný numerický spánok vyhodiť, aby nebol v antecedentoch
df = df.drop(columns=[SLEEP_COL])

# -----------------------------
# 4️⃣ Prevod riadkov na transakcie
#     - cieľový item bude napr. "SLEEP_CAT=LOW_<=6"
# -----------------------------
transactions = []

for _, row in df.iterrows():
    items = []
    for col in df.columns:
        val = row[col]
        if pd.isna(val):
            continue

        # kategórie (vrátane SLEEP_CAT)
        if pd.api.types.is_numeric_dtype(df[col]):
            # malé rozsahy ber ako kategórie
            if df[col].nunique() <= 10:
                if float(val).is_integer():
                    val = int(val)
                items.append(f"{col}={val}")
            else:
                # ak by tu ostali numerické kontinuálne stĺpce, zabaň ich kvartilmi
                bins = pd.qcut(df[col], q=4, duplicates="drop")
                items.append(f"{col}∈{bins.loc[row.name]}")
        else:
            items.append(f"{col}={val}")

    transactions.append(list(set(items)))

# -----------------------------
# 5️⃣ One-hot encoding
# -----------------------------
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
onehot = pd.DataFrame(te_array, columns=te.columns_)

# -----------------------------
# 6️⃣ Frequent itemsets
# -----------------------------
itemsets = apriori(
    onehot,
    min_support=MIN_SUPPORT,
    use_colnames=True,
    max_len=MAX_LEN
)

# -----------------------------
# 7️⃣ Association rules
# -----------------------------
rules = association_rules(
    itemsets,
    metric="confidence",
    min_threshold=MIN_CONFIDENCE
)

# (voliteľné) odfiltruj slabé
rules = rules[rules["lift"] > 1.0].copy()

# -----------------------------
# 8️⃣ Filtrovanie – len pravidlá kde konzekvent je SLEEP_CAT
# -----------------------------
rules_sleep = rules[
    rules["consequents"].apply(
        lambda x: any(item.startswith(SLEEP_CAT_COL + "=") for item in x)
    )
].copy()

# Formátovanie
rules_sleep["antecedents"] = rules_sleep["antecedents"].apply(lambda s: " & ".join(sorted(list(s))))
rules_sleep["consequents"] = rules_sleep["consequents"].apply(lambda s: " & ".join(sorted(list(s))))

rules_sleep = rules_sleep.sort_values(
    ["lift", "confidence", "support"],
    ascending=[False, False, False]
)

# -----------------------------
# 9️⃣ Uloženie
# -----------------------------
itemsets.to_csv("frequent_itemsets_mlxtend_sleepcat.csv", index=False)
rules_sleep.to_csv("association_rules_SLEEP_CAT_ONLY.csv", index=False)

# -----------------------------
# 🔟 Výpis
# -----------------------------
print("Hotovo ✅")
print(f"Itemsets: {len(itemsets)}")
print(f"Sleep-cat rules: {len(rules_sleep)}")

print("\nRozdelenie kategórií spánku (koľko respondentov v každej):")
print(df[SLEEP_CAT_COL].value_counts(dropna=False))

if len(rules_sleep) > 0:
    print("\nTop 10 pravidiel (→ kategória spánku):\n")
    print(
        rules_sleep[
            ["antecedents", "consequents", "support", "confidence", "lift"]
        ].head(10).to_string(index=False)
    )
else:
    print("\n⚠️ Nenašli sa pravidlá pre kategórie spánku. Skús znížiť MIN_SUPPORT alebo MIN_CONFIDENCE.")
