import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------
# NÁZVY ACTIVITY STĹPCOV
# --------------------------------------------------
LIGHT_ACTIVITY = "Koľko minút ste počas minulého týždňa venovali nenáročnej fyzickej aktivite? "
MODERATE_ACTIVITY = "Koľko minút ste počas minulého týždňa venovali mierne náročnej fyzickej aktivite? "
VIGOROUS_ACTIVITY = "Koľko minút ste počas minulého týždňa venovali náročnej fyzickej aktivite? (heavylifting, vysoké tempo pri behu alebo bicyklovaní)"


# --------------------------------------------------
# 1. Automatická detekcia ID / dátumových stĺpcov
# --------------------------------------------------
def auto_detect_columns_to_drop(df: pd.DataFrame) -> list:
    cols_to_drop = []

    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        unique_ratio = df[col].nunique(dropna=True) / len(df)
        if unique_ratio > 0.95:
            cols_to_drop.append(col)
            print(f"[Auto-Drop] '{col}' -> identifikovaný ako ID / unikátny text")

    return cols_to_drop


# --------------------------------------------------
# 2. Report premenných silno korelovaných s cieľom
# --------------------------------------------------
def report_highly_target_correlated_columns(
    df: pd.DataFrame,
    target_col: str,
    threshold: float = 0.75
) -> list:
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if target_col not in numeric_df.columns:
        print(f"[Correlation Report] Cieľová premenná '{target_col}' nie je numerická alebo sa v dátach nenašla.")
        return []

    corr_matrix = numeric_df.corr()
    corr_with_target = corr_matrix[target_col].drop(labels=[target_col]).abs()
    suspicious = corr_with_target[corr_with_target > threshold].sort_values(ascending=False)

    if suspicious.empty:
        print(f"[Correlation Report] Neboli nájdené numerické stĺpce s |korelácia| > {threshold} voči cieľu.")
        return []

    print(f"\n[Correlation Report] Premenné s |korelácia| > {threshold} voči cieľu '{target_col}':")
    for col, corr_value in suspicious.items():
        print(f" - {col}: {corr_value:.4f}")

    print("Pozn.: Ide len o report pre kontrolu možného data leakage. Stĺpce sa automaticky neodstraňujú.\n")
    return suspicious.index.tolist()


# --------------------------------------------------
# 3. Diskretizácia cieľovej premennej podľa targetu
# --------------------------------------------------
def discretize_activity(minutes: float, target_col: str) -> float:
    if pd.isna(minutes):
        return np.nan

    if target_col == LIGHT_ACTIVITY:
        return 0 if minutes < 300 else 1

    if target_col == MODERATE_ACTIVITY:
        return 0 if minutes < 150 else 1

    if target_col == VIGOROUS_ACTIVITY:
        return 0 if minutes < 75 else 1

    raise ValueError(f"Neznámy target_col: {target_col}")


# --------------------------------------------------
# 4. Texty tried podľa targetu
# --------------------------------------------------
def get_class_names(target_col: str) -> list:
    if target_col == LIGHT_ACTIVITY:
        return ["< 300 min", ">= 300 min"]
    if target_col == MODERATE_ACTIVITY:
        return ["< 150 min", ">= 150 min"]
    if target_col == VIGOROUS_ACTIVITY:
        return ["< 75 min", ">= 75 min"]
    return ["Trieda 0", "Trieda 1"]


# --------------------------------------------------
# 5. Krátky názov targetu pre výstupy
# --------------------------------------------------
def get_target_short_name(target_col: str) -> str:
    if target_col == LIGHT_ACTIVITY:
        return "light_activity"
    if target_col == MODERATE_ACTIVITY:
        return "moderate_activity"
    if target_col == VIGOROUS_ACTIVITY:
        return "vigorous_activity"
    return "target"


# --------------------------------------------------
# 6. Načítanie a predspracovanie dát
# --------------------------------------------------
def load_and_preprocess_data(file_path: str, target_col: str) -> pd.DataFrame:
    print(f"Načítavam súbor: {file_path}")
    print(f"Použitý target: {target_col}")

    df = pd.read_csv(file_path, sep=";", decimal=",")
    print(f"Pôvodný tvar dát: {df.shape}")

    # Odstránenie ostatných activity stĺpcov (prevencia leakage)
    activity_cols = [LIGHT_ACTIVITY, MODERATE_ACTIVITY, VIGOROUS_ACTIVITY]
    leakage_cols = [col for col in activity_cols if col != target_col and col in df.columns]

    if leakage_cols:
        df = df.drop(columns=leakage_cols)
        print(f"Automaticky odstránené activity stĺpce (leakage): {leakage_cols}")

    # Odstránenie riadkov bez cieľa
    df = df.dropna(subset=[target_col])
    print(f"Tvar dát po odstránení riadkov bez cieľa: {df.shape}")

    # Automatická detekcia nevhodných textových stĺpcov
    bad_cols = auto_detect_columns_to_drop(df)
    if bad_cols:
        df = df.drop(columns=bad_cols)
        print(f"Odstránené textové / unikátne stĺpce: {bad_cols}")
    else:
        print("Neboli nájdené žiadne textové stĺpce na automatické odstránenie.")

    # Kontrola leakage
    report_highly_target_correlated_columns(df, target_col=target_col, threshold=0.9)

    # Len numerické dáta
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    print(f"Tvar numerických dát: {df_numeric.shape}")

    # Diskretizácia cieľa
    df_numeric["activity_class"] = df_numeric[target_col].apply(lambda x: discretize_activity(x, target_col))
    df_numeric = df_numeric.dropna(subset=["activity_class"]).copy()
    df_numeric["activity_class"] = df_numeric["activity_class"].astype(int)

    # Drop pôvodného targetu
    df_numeric = df_numeric.drop(columns=[target_col])

    print(f"Tvar dát po vytvorení 'activity_class': {df_numeric.shape}")
    print("Rozdelenie tried v 'activity_class':")
    print(df_numeric["activity_class"].value_counts(dropna=False).sort_index())

    return df_numeric


# --------------------------------------------------
# 7. Bezpečný počet CV foldov
# --------------------------------------------------
def get_safe_cv_folds(y: pd.Series, max_cv: int = 5) -> int:
    min_class_count = y.value_counts().min()
    cv_folds = min(max_cv, min_class_count)

    if cv_folds < 2:
        raise ValueError(
            f"Najmenšia trieda obsahuje iba {min_class_count} vzorku/vzorky. "
            f"Nie je možné korektne spustiť aspoň 2-fold cross-validation."
        )

    return cv_folds


# --------------------------------------------------
# 8. Baseline model
# --------------------------------------------------
def evaluate_baseline(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    baseline_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", DummyClassifier(strategy="most_frequent"))
    ])

    baseline_pipeline.fit(X_train, y_train)
    baseline_pred = baseline_pipeline.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    print("\n--- Baseline (najčastejšia trieda) ---")
    print(f"Baseline Accuracy: {baseline_acc * 100:.2f} %")

    return baseline_acc, baseline_pred


# --------------------------------------------------
# 9. Hľadanie najlepšieho modelu so SMOTE
# --------------------------------------------------
def find_best_model(X_train: pd.DataFrame, y_train: pd.Series, target_col: str):
    """
    Nájde najlepší decision tree model pomocou GridSearchCV.
    SMOTE je aplikované správne iba vo vnútri CV foldov.

    Špeciálna oprava:
    - iba pre LIGHT_ACTIVITY sa použije:
        cv <= 3
        k_neighbors = 1
    """
    min_class_count = y_train.value_counts().min()

    if target_col == LIGHT_ACTIVITY:
        cv_folds = get_safe_cv_folds(y_train, max_cv=3)
        smote_k_neighbors = 1
        print("\n[LIGHT_ACTIVITY režim] Používam špeciálne nastavenie pre extrémny imbalance:")
        print(" - max cv = 3")
        print(" - SMOTE k_neighbors = 1")
    else:
        cv_folds = get_safe_cv_folds(y_train, max_cv=5)
        smote_k_neighbors = min(5, min_class_count - 1)

        if smote_k_neighbors < 1:
            raise ValueError(
                f"Najmenšia trieda v tréningových dátach má iba {min_class_count} vzorku/vzorky. "
                f"SMOTE nie je možné korektne použiť."
            )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", SMOTE(random_state=42, k_neighbors=smote_k_neighbors)),
        ("clf", DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ))
    ])

    param_grid = {
        "clf__max_depth": [2, 3, 4, 5],
        "clf__min_samples_leaf": [2, 5, 8, 10],
        "clf__min_samples_split": [5, 10, 15, 20],
        "clf__criterion": ["gini", "entropy"]
    }

    print(f"\nPoužívam GridSearchCV s cv={cv_folds}.")
    print(f"SMOTE bude v pipeline s k_neighbors={smote_k_neighbors}.")
    print("SMOTE sa teda aplikuje správne iba na tréningovú časť každého CV foldu.")

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
        error_score="raise"
    )

    print("Spúšťam GridSearchCV...")
    grid.fit(X_train, y_train)

    print(f"Najlepšie parametre: {grid.best_params_}")
    print(f"Najlepšie CV F1-macro: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


# --------------------------------------------------
# 10. Hlavná funkcia
# --------------------------------------------------
def main():
    FILE_PATH = "datasets/sleep/Dokazník_merged_wo_dateTime_feelingToday_sleepAvg_sumBike_deletedDayOfExcercise.csv"

    # --------------------------------------------------
    # PREPÍNAČ TARGETU:
    # LIGHT_ACTIVITY / MODERATE_ACTIVITY / VIGOROUS_ACTIVITY
    # --------------------------------------------------
    TARGET_COL = LIGHT_ACTIVITY

    target_short_name = get_target_short_name(TARGET_COL)
    class_names = get_class_names(TARGET_COL)

    df = load_and_preprocess_data(FILE_PATH, TARGET_COL)

    # Rozdelenie na vstupy a cieľ
    X = df.drop(columns=["activity_class"])
    y = df["activity_class"]

    # Train-test split so stratifikáciou
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    print("\nRozdelenie tried v y_train:")
    print(y_train.value_counts().sort_index())

    print("\nRozdelenie tried v y_test:")
    print(y_test.value_counts().sort_index())

    # Baseline
    baseline_acc, baseline_pred = evaluate_baseline(X_train, X_test, y_train, y_test)

    # Model so SMOTE
    best_model, best_params, best_cv_score = find_best_model(X_train, y_train, TARGET_COL)

    # Predikcia a vyhodnotenie
    y_pred = best_model.predict(X_test)
    model_acc = accuracy_score(y_test, y_pred)

    print("\n--- Výsledok na testovacích dátach ---")
    print(f"Accuracy: {model_acc * 100:.2f} %")
    print(f"Rozdiel oproti baseline: {(model_acc - baseline_acc) * 100:.2f} p. b.")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Model stromu z pipeline
    tree_model = best_model.named_steps["clf"]

    # Feature importance
    importances = pd.Series(tree_model.feature_importances_, index=X.columns)
    print("\nTop 10 feature importances:")
    print(importances.sort_values(ascending=False).head(10))

    # Textový výpis stromu
    print("\n--- TEXTOVÝ VÝPIS ROZHODOVACIEHO STROMU ---")
    tree_rules = export_text(
        tree_model,
        feature_names=list(X.columns),
        max_depth=4
    )
    print(tree_rules)

    # Matica zámen
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predpovedané")
    plt.ylabel("Skutočné")
    plt.title(f"Matica zámen - {target_short_name} (SMOTE)")
    plt.tight_layout()
    plt.savefig(f"outputs/matrix_{target_short_name}_smote.png", bbox_inches="tight")
    plt.show()

    # Feature importance graf
    plt.figure(figsize=(10, 6))
    importances.nlargest(10).sort_values().plot(kind="barh", color="skyblue")
    plt.title(f"Top 10 faktorov - {target_short_name} (SMOTE)")
    plt.xlabel("Dôležitosť (Gini importance)")
    plt.tight_layout()
    plt.savefig(f"outputs/factors_{target_short_name}_smote.png", bbox_inches="tight")
    plt.show()

    # Vizualizácia stromu
    plt.figure(figsize=(26, 14))
    plot_tree(
        tree_model,
        feature_names=X.columns,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9,
        max_depth=4
    )
    plt.title(f"Rozhodovací strom - {target_short_name} (SMOTE)")
    plt.tight_layout()
    plt.savefig(f"outputs/tree_{target_short_name}_smote.png", bbox_inches="tight")
    plt.show()

    print(f"\nHĺbka stromu: {tree_model.get_depth()}")
    print(f"Počet listov: {tree_model.get_n_leaves()}")

    print("\nNajlepšie parametre modelu:")
    print(best_params)

    print(f"\nNajlepšie CV F1-macro: {best_cv_score:.4f}")
    print("\nVýstupy boli uložené do priečinka 'outputs'.")


if __name__ == "__main__":
    main()