import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
    """
    Nájde textové stĺpce s veľmi vysokou unikátnosťou.
    Takéto stĺpce často reprezentujú identifikátory, mená alebo presné dátumy/časy.
    """
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
    """
    Vypíše numerické stĺpce silno korelované s cieľovou premennou.
    Slúži ako metodická kontrola možného data leakage.
    """
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
    """
    Diskretizácia podľa typu aktivity.

    LIGHT_ACTIVITY:
        0 = < 300 min / týždeň
        1 = >= 300 min / týždeň

    MODERATE_ACTIVITY:
        0 = < 150 min / týždeň
        1 = >= 150 min / týždeň

    VIGOROUS_ACTIVITY:
        0 = < 75 min / týždeň
        1 = >= 75 min / týždeň
    """
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
    """
    Načíta dáta, automaticky odstráni ostatné activity stĺpce,
    vytvorí cieľovú triedu a ponechá len numerické premenné.
    """
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

    # Drop pôvodného targetu
    df_numeric = df_numeric.drop(columns=[target_col])

    print(f"Tvar dát po vytvorení 'activity_class': {df_numeric.shape}")
    print("Rozdelenie tried v 'activity_class':")
    print(df_numeric["activity_class"].value_counts(dropna=False).sort_index())

    return df_numeric


# --------------------------------------------------
# 7. Hľadanie najlepšieho klasifikátora
# --------------------------------------------------
def find_best_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """
    Nájde najlepšie parametre rozhodovacieho stromu pomocou GridSearchCV.
    """
    param_grid = {
        "max_depth": [None, 4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 5, 10],
        "min_samples_split": [2, 5, 10],
        "max_features": [None, "sqrt", "log2"],
        "criterion": ["gini", "entropy"]
    }

    clf = DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    )

    min_class_count = y_train.value_counts().min()
    cv_folds = min(10, min_class_count)

    if cv_folds < 2:
        raise ValueError(
            f"Najmenšia trieda v tréningových dátach má iba {min_class_count} vzorku/vzorky. "
            f"To nestačí ani na 2-fold cross-validation."
        )

    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="f1_macro",
        n_jobs=-1
    )

    print(f"\nSpúšťam GridSearchCV (cv={cv_folds}, scoring='f1_macro')...")
    grid.fit(X_train, y_train)

    print(f"Najlepšie parametre: {grid.best_params_}")
    print(f"Najlepšie CV F1-macro: {grid.best_score_:.4f}")

    return grid.best_estimator_


# --------------------------------------------------
# 8. Hlavná funkcia
# --------------------------------------------------
def main():
    FILE_PATH = "datasets/sleep/Dokazník_merged_wo_dateTime_feelingToday_sleepAvg_sumBike_deletedDayOfExcercise.csv"

    # --------------------------------------------------
    # PREPÍNAČ TARGETU:
    # LIGHT_ACTIVITY / MODERATE_ACTIVITY / VIGOROUS_ACTIVITY
    # --------------------------------------------------
    TARGET_COL = VIGOROUS_ACTIVITY

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

    # Imputácia
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

    # Model
    best_model = find_best_model(X_train, y_train)

    # Predikcia a vyhodnotenie
    y_pred = best_model.predict(X_test)

    print("\n--- Výsledok na testovacích dátach ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f} %")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Feature importance
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    print("\nTop 10 feature importances:")
    print(importances.sort_values(ascending=False).head(10))

    # Textový výpis stromu
    print("\n--- TEXTOVÝ VÝPIS ROZHODOVACIEHO STROMU ---")
    tree_rules = export_text(
        best_model,
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
    plt.title(f"Matica zámen - {target_short_name}")
    plt.tight_layout()
    plt.savefig(f"outputs/matrix_{target_short_name}.png", bbox_inches="tight")
    plt.show()

    # Feature importance graf
    plt.figure(figsize=(10, 6))
    importances.nlargest(10).sort_values().plot(kind="barh", color="skyblue")
    plt.title(f"Top 10 faktorov - {target_short_name}")
    plt.xlabel("Dôležitosť (Gini importance)")
    plt.tight_layout()
    plt.savefig(f"outputs/factors_{target_short_name}.png", bbox_inches="tight")
    plt.show()

    # Vizualizácia stromu
    plt.figure(figsize=(26, 14))
    plot_tree(
        best_model,
        feature_names=X.columns,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9,
        max_depth=4
    )
    plt.title(f"Rozhodovací strom - {target_short_name}")
    plt.tight_layout()
    plt.savefig(f"outputs/tree_{target_short_name}.png", bbox_inches="tight")
    plt.show()

    print(f"\nHĺbka stromu: {best_model.get_depth()}")
    print(f"Počet listov: {best_model.get_n_leaves()}")

    print("\nVýstupy boli uložené do priečinka 'outputs'.")


if __name__ == "__main__":
    main()