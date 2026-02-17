import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------
# 1) Automatická detekcia stĺpcov na drop
# --------------------------------------------------
def auto_detect_columns_to_drop(df: pd.DataFrame) -> list[str]:
    cols_to_drop = []
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        uniq_ratio = df[col].nunique(dropna=False) / max(len(df), 1)
        if uniq_ratio > 0.95:
            cols_to_drop.append(col)
            print(f"[Auto-Drop] '{col}' -> ID/unikátny text")
    return cols_to_drop


# --------------------------------------------------
# 2) Diskretizácia cieľa
# --------------------------------------------------
def categorize_feeling(score: int) -> int:
    if score <= 6:
        return 0
    if score <= 8:
        return 1
    return 2


# --------------------------------------------------
# 3) Načítanie dát
# --------------------------------------------------
def load_and_preprocess_data(file_path: str, target_col: str):
    print(f"Načítavam súbor: {file_path}")
    df = pd.read_csv(file_path, sep=";", decimal=",")

    if target_col not in df.columns:
        raise ValueError("Cieľový stĺpec sa nenašiel.")

    df = df.dropna(subset=[target_col]).copy()

    bad_cols = auto_detect_columns_to_drop(df)
    if bad_cols:
        df = df.drop(columns=bad_cols)

    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df.drop(columns=[target_col])

    valid_mask = y.between(1, 10)
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(int)

    y_raw = y.copy()
    y = y.apply(categorize_feeling).astype(int)

    X = X.select_dtypes(include=[np.number]).copy()

    nun = X.nunique(dropna=False)
    const_cols = nun[nun <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)

    print(f"Počet vzoriek: {len(X)} | Počet feature: {X.shape[1]}")
    print("Rozdelenie pôvodnej škály (1–10):")
    print(y_raw.value_counts().sort_index())
    print("Rozdelenie kategórií:")
    print(y.value_counts().sort_index())

    return X, y


# --------------------------------------------------
# 4) Korelácie (bez heatmapy)
# --------------------------------------------------
def report_correlations(X: pd.DataFrame, y: pd.Series, out_dir: str, top_k: int = 20):
    os.makedirs(out_dir, exist_ok=True)

    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)

    corr_to_target = X_imp.corrwith(y, method="spearman") \
                          .sort_values(key=lambda s: s.abs(), ascending=False)

    print("\n--- TOP korelácie feature ↔ cieľ (Spearman) ---")
    print(corr_to_target.head(top_k))

    corr_to_target.to_csv(os.path.join(out_dir, "correlation_to_target_spearman.csv"))


# --------------------------------------------------
# 5) Model so SMOTE
# --------------------------------------------------
def find_best_model_smote(X_train: pd.DataFrame, y_train: pd.Series):

    min_class = int(y_train.value_counts().min())
    cv_folds = int(max(2, min(10, min_class)))
    k_neighbors = max(1, min(5, min_class - 1))

    print("\n--- SMOTE nastavenie ---")
    print(f"Min trieda (train): {min_class}")
    print(f"cv={cv_folds}")
    print(f"k_neighbors={k_neighbors}")

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", SMOTE(random_state=42, k_neighbors=k_neighbors)),
        ("clf", DecisionTreeClassifier(random_state=42))
    ])

    param_grid = {
        "clf__max_depth": [None, 3, 4, 5, 6, 8, 10],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__min_samples_split": [2, 5, 10],
        "clf__criterion": ["gini", "entropy"],
    }

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv_folds,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Najlepšie parametre:", grid.best_params_)
    print("Najlepšie CV f1_macro:", grid.best_score_)

    return grid.best_estimator_


# --------------------------------------------------
# 6) MAIN
# --------------------------------------------------
def main():

    os.makedirs("outputs", exist_ok=True)

    FILE_PATH = "datasets/Dokazník_feeling_wo_datetime_feeling_today_cycling_only_minutes_total_activ.csv"
    TARGET_COL = "Ako sa dnes cítite po zdravotnej stránke od 1 po 10? (1 - zle, 10 - dobre)"

    CLASS_NAMES = {
        0: "Slabšie (1–6)",
        1: "Dobre (7–8)",
        2: "Výborne (9–10)",
    }

    X, y = load_and_preprocess_data(FILE_PATH, TARGET_COL)

    # korelácie
    report_correlations(X, y, "outputs", top_k=20)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nRozdelenie tried v train:")
    print(y_train.value_counts())

    # model
    best_model = find_best_model_smote(X_train, y_train)

    y_pred = best_model.predict(X_test)

    labels = sorted(CLASS_NAMES.keys())
    target_names = [CLASS_NAMES[x] for x in labels]

    print("\n--- Výsledky ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix:")
    print(cm)

    # Feature importance
    clf = best_model.named_steps["clf"]
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    importances.head(10).sort_values().plot(kind="barh")
    plt.title("Top 10 faktorov (SMOTE)")
    plt.tight_layout()
    plt.savefig("outputs/factors_smote.png", dpi=300)
    plt.show()

    # Strom
    tree_depth = clf.get_depth()
    print(f"\nHĺbka stromu: {tree_depth}")

    plot_depth = tree_depth if tree_depth <= 5 else 5

    plt.figure(figsize=(28, 14))
    plot_tree(
        clf,
        feature_names=X.columns,
        class_names=[CLASS_NAMES[i] for i in labels],
        filled=True,
        rounded=True,
        max_depth=plot_depth
    )
    plt.tight_layout()
    plt.savefig("outputs/tree_smote.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
