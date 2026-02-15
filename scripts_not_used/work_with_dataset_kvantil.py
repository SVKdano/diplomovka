import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Automatická detekcia ID / dátumových stĺpcov
# --------------------------------------------------
def auto_detect_columns_to_drop(df):
    cols_to_drop = []
    object_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in object_cols:
        if df[col].nunique() / len(df) > 0.95:
            cols_to_drop.append(col)
            print(f"[Auto-Drop] '{col}' -> identifikovaný ako ID/unikátny text")
    return cols_to_drop

# --------------------------------------------------
# 2. KVANTILOVÁ DISKRETIZÁCIA CIEĽA (VARIANT B)
# --------------------------------------------------
def quantile_discretize(series, q=3):
    """
    Rozdelí cieľovú premennú do q kvantilov (data-driven)
    """
    return pd.qcut(series, q=q, labels=False, duplicates='drop')

# --------------------------------------------------
# 3. Načítanie a predspracovanie dát
# --------------------------------------------------
def load_and_preprocess_data(file_path, target_col):
    print(f"Načítavam súbor: {file_path}")
    df = pd.read_csv(file_path, sep=';', decimal=',')

    # Odstránime riadky bez cieľa
    df = df.dropna(subset=[target_col])

    bad_cols = auto_detect_columns_to_drop(df)
    if bad_cols:
        df = df.drop(columns=bad_cols)

    df_numeric = df.select_dtypes(include=[np.number]).copy()

    # Kvantilová diskretizácia cieľa
    df_numeric["sleep_class"] = quantile_discretize(df_numeric[target_col], q=3)
    df_numeric = df_numeric.drop(columns=[target_col])

    return df_numeric

# --------------------------------------------------
# 4. Hľadanie najlepšieho klasifikátora
# --------------------------------------------------
def find_best_model(X_train, y_train):

    param_grid = {
        'max_depth': [3, 4, 5, 6, 8, None],
        'min_samples_leaf': [5, 10, 20],
        'min_samples_split': [5, 10, 20],
        'max_features': [None, 'sqrt'],
        'criterion': ['gini', 'entropy']
    }

    clf = DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced'
    )

    grid = GridSearchCV(
        clf,
        param_grid,
        cv=10,
        scoring='f1_macro',
        n_jobs=-1
    )

    print("\nSpúšťam GridSearch (CV=10, optimalizujem F1-macro)...")
    grid.fit(X_train, y_train)

    print(f"Najlepšie parametre: {grid.best_params_}")
    print(f"CV F1-macro: {grid.best_score_:.4f}")

    return grid.best_estimator_

# --------------------------------------------------
# 5. Hlavná funkcia
# --------------------------------------------------
def main():

    FILE_PATH = "Dokazník_merged_wo_dateTime_feelingToday_sleepAvg_sumBike_deletedDayOfExcercise_mergedActivityTime.csv"
    TARGET_COL = "Koľko hodín spíte v priemere"

    df = load_and_preprocess_data(FILE_PATH, TARGET_COL)

    X = df.drop(columns=["sleep_class"])
    y = df["sleep_class"]

    # Stratifikovaný split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Imputácia
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

    best_model = find_best_model(X_train, y_train)

    # Vyhodnotenie
    y_pred = best_model.predict(X_test)

    class_names = ["Málo spím", "Normálne", "Veľa spím"]

    print("\n--- Výsledok na testovacích dátach ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f} %")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predpovedané")
    plt.ylabel("Skutočné")
    plt.title("Matica zámien – kvantilová klasifikácia")
    plt.show()

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    importances.nlargest(10).sort_values().plot(kind='barh')
    plt.title("Top 10 faktorov ovplyvňujúcich dĺžku spánku")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # Vizualizácia stromu
    # --------------------------------------------------
    plt.figure(figsize=(26, 14))
    plot_tree(
        best_model,
        feature_names=X.columns,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=9
    )
    plt.title("Rozhodovací strom – kvantilové rozdelenie")
    plt.savefig("final_tree_variantB_quantile.png", bbox_inches="tight")
    plt.show()

    print("\nStrom bol uložený ako 'final_tree_variantB_quantile.png'")

if __name__ == "__main__":
    main()
