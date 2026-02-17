import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------
# 1) Automatická detekcia stĺpcov na drop (unikátny text/ID)
# --------------------------------------------------
def auto_detect_columns_to_drop(df: pd.DataFrame) -> list[str]:
    cols_to_drop = []
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        uniq_ratio = df[col].nunique(dropna=False) / max(len(df), 1)
        if uniq_ratio > 0.95:
            cols_to_drop.append(col)
            print(f"[Auto-Drop] '{col}' -> ID/unikátny text (uniq_ratio={uniq_ratio:.3f})")
    return cols_to_drop


# --------------------------------------------------
# 2) Načítanie a predspracovanie dát
# --------------------------------------------------
def load_and_preprocess_data(file_path: str, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    print(f"Načítavam súbor: {file_path}")
    df = pd.read_csv(file_path, sep=";", decimal=",")

    if target_col not in df.columns:
        raise ValueError(
            f"Cieľový stĺpec sa nenašiel.\n"
            f"Hľadaný: {target_col}\n"
            f"Dostupné stĺpce: {list(df.columns)}"
        )

    df = df.dropna(subset=[target_col]).copy()

    bad_cols = auto_detect_columns_to_drop(df)
    if bad_cols:
        df = df.drop(columns=bad_cols)

    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df.drop(columns=[target_col])

    valid_mask = y.between(1, 10)
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(int)

    X = X.select_dtypes(include=[np.number]).copy()

    nun = X.nunique(dropna=False)
    const_cols = nun[nun <= 1].index.tolist()
    if const_cols:
        print(f"[Auto-Drop] Konštantné stĺpce odstránené: {const_cols}")
        X = X.drop(columns=const_cols)

    print(f"Počet vzoriek: {len(X)} | Počet feature: {X.shape[1]}")
    print("Rozdelenie cieľovej premennej (po filtrovaní 1–10):")
    print(y.value_counts().sort_index())

    return X, y


# --------------------------------------------------
# 3) Hľadanie najlepšieho modelu
# --------------------------------------------------
def find_best_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    param_grid = {
        "max_depth": [None, 3, 4, 5, 6, 8, 10, 12],
        "min_samples_leaf": [1, 2, 5, 10],
        "min_samples_split": [2, 5, 10],
        "max_features": [None, "sqrt", "log2"],
        "criterion": ["gini", "entropy"],
    }

    clf = DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    )

    min_class = int(y_train.value_counts().min())
    cv_folds = int(max(2, min(10, min_class)))

    print("\nSpúšťam GridSearch...")
    print(f"Min počet vzoriek v triede (train): {min_class} -> používam cv={cv_folds}")
    print("Optimalizujem: f1_macro")

    grid = GridSearchCV(
        clf,
        param_grid,
        cv=cv_folds,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print(f"Najlepšie parametre: {grid.best_params_}")
    print(f"Najlepšie CV f1_macro: {grid.best_score_:.4f}")

    return grid.best_estimator_


# --------------------------------------------------
# 4) MAIN
# --------------------------------------------------
def main():
    FILE_PATH = "datasets/Dokazník_merged_adjusted.csv"
    TARGET_COL = "Ako sa dnes cítite po zdravotnej stránke od 1 po 10? (1 - zle, 10 - dobre)"

    X, y = load_and_preprocess_data(FILE_PATH, TARGET_COL)

    # --------------------------------------------------
    # Výpis korelácií s cieľovou premennou
    # --------------------------------------------------
    print("\n--- Korelácia jednotlivých feature s cieľovou premennou ---")

    tmp = X.copy()
    tmp["TARGET"] = y

    corr_matrix = tmp.corr(numeric_only=True)
    target_corr = corr_matrix["TARGET"].drop("TARGET")

    # zoradiť podľa absolútnej hodnoty
    target_corr_sorted = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)

    print(target_corr_sorted)

    # --------------------------------------------------
    # Train/Test split
    # --------------------------------------------------
    class_counts = y.value_counts()
    too_small = class_counts[class_counts < 2]

    if len(too_small) > 0:
        print("\nPozor: niektoré triedy majú < 2 vzorky -> stratify vypínam.")
        print("Triedy s malým počtom:", too_small.to_dict())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)

    best_model = find_best_model(X_train_imp, y_train)

    y_pred = best_model.predict(X_test_imp)

    labels = sorted(np.unique(y_test))
    target_names = [str(x) for x in labels]

    print("\n--- Výsledok na testovacích dátach ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f} %")
    print(f"MAE (priemerná odchýlka v bodoch): {mean_absolute_error(y_test, y_pred):.3f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    ))

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predpovedané")
    plt.ylabel("Skutočné")
    plt.title("Matica zámien (Confusion Matrix) - Feeling 1–10")
    plt.tight_layout()
    plt.savefig("outputs/matrix_feeling.png", bbox_inches="tight")
    plt.show()

    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n--- Feature importance ---")
    print(importances)

    tree_depth = best_model.get_depth()
    tree_leaves = best_model.get_n_leaves()
    print(f"\nSkutočná hĺbka stromu: {tree_depth}")
    print(f"Počet listov (leaves): {tree_leaves}")

    plot_depth = tree_depth if tree_depth <= 5 else 5
    print(f"Zobrazujem do hĺbky: {plot_depth}")

    plt.figure(figsize=(30, 14))
    plot_tree(
        best_model,
        feature_names=X.columns,
        class_names=[str(i) for i in sorted(np.unique(y_train))],
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=plot_depth
    )
    plt.title(f"Rozhodovací strom (zobrazené prvé {plot_depth} úrovne z {tree_depth})")
    out_name = "outputs/final_tree_feeling.png"
    plt.savefig(out_name, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"\nStrom uložený ako '{out_name}'")


if __name__ == "__main__":
    main()
