import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------
# 1. Automatická detekcia ID / dátumových stĺpcov
# --------------------------------------------------
def auto_detect_columns_to_drop(df: pd.DataFrame) -> list:
    """
    Nájde textové stĺpce s veľmi vysokou unikátnosťou.
    Takéto stĺpce často reprezentujú identifikátory, mená alebo presné dátumy/časy,
    ktoré nebývajú vhodné na modelovanie.
    """
    cols_to_drop = []

    object_cols = df.select_dtypes(include=['object', 'string']).columns
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
    Tento krok slúži ako metodická kontrola možného data leakage.
    Stĺpce sa automaticky NEODSTRAŇUJÚ, iba sa reportujú.
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
# 3. Diskretizácia cieľovej premennej
# --------------------------------------------------
def discretize_sleep(hours: float) -> float:
    """
    Diskretizácia priemernej dĺžky spánku do 3 tried:
    0 = málo
    1 = normálne
    2 = veľa
    """
    if pd.isna(hours):
        return np.nan
    if hours <= 7:
        return 0   # málo
    elif hours <= 9:
        return 1   # normálne
    else:
        return 2   # veľa


# --------------------------------------------------
# 4. Načítanie a predspracovanie dát
# --------------------------------------------------
def load_and_preprocess_data(file_path: str, target_col: str) -> pd.DataFrame:
    """
    Načíta dáta, odstráni nepoužiteľné stĺpce, vytvorí cieľovú triedu
    a ponechá iba numerické premenné vhodné pre model.
    """
    print(f"Načítavam súbor: {file_path}")
    df = pd.read_csv(file_path, sep=';', decimal=',')

    print(f"Pôvodný tvar dát: {df.shape}")

    # Odstránenie riadkov bez cieľovej premennej
    df = df.dropna(subset=[target_col])
    print(f"Tvar dát po odstránení riadkov bez cieľa: {df.shape}")

    # Automatická detekcia nevhodných textových stĺpcov
    bad_cols = auto_detect_columns_to_drop(df)
    if bad_cols:
        df = df.drop(columns=bad_cols)
        print(f"Odstránené textové / unikátne stĺpce: {bad_cols}")
    else:
        print("Neboli nájdené žiadne textové stĺpce na automatické odstránenie.")

    # Metodická kontrola možného leakage
    report_highly_target_correlated_columns(df, target_col=target_col, threshold=0.9)

    # Ponecháme len numerické dáta
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    print(f"Tvar numerických dát: {df_numeric.shape}")

    # Diskretizácia cieľovej premennej
    df_numeric["sleep_class"] = df_numeric[target_col].apply(discretize_sleep)

    # Odstránenie prípadných NaN po diskretizácii
    df_numeric = df_numeric.dropna(subset=["sleep_class"]).copy()
    df_numeric["sleep_class"] = df_numeric["sleep_class"].astype(int)

    # Odstránenie pôvodnej cieľovej premennej -> prevencia leakage
    df_numeric = df_numeric.drop(columns=[target_col])

    print(f"Tvar dát po vytvorení 'sleep_class': {df_numeric.shape}")
    print("Rozdelenie tried v 'sleep_class':")
    print(df_numeric["sleep_class"].value_counts(dropna=False).sort_index())

    return df_numeric


# --------------------------------------------------
# 5. Výpočet počtu foldov pre CV
# --------------------------------------------------
def get_safe_cv_folds(y: pd.Series, max_cv: int = 5) -> int:
    """
    Nastaví bezpečný počet CV foldov podľa najmenšej triedy.
    """
    min_class_count = y.value_counts().min()
    cv_folds = min(max_cv, min_class_count)

    if cv_folds < 2:
        raise ValueError(
            f"Najmenšia trieda obsahuje iba {min_class_count} vzorku/vzorky. "
            f"Nie je možné korektne spustiť aspoň 2-fold cross-validation."
        )

    return cv_folds


# --------------------------------------------------
# 6. Hľadanie najlepšieho klasifikátora pomocou ROS
# --------------------------------------------------
def find_best_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Nájde najlepšie parametre jednoduchšieho rozhodovacieho stromu pomocou GridSearchCV.
    RandomOverSampler (ROS) je správne aplikovaný iba vo vnútri CV foldov cez pipeline.
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('sampler', RandomOverSampler(random_state=42)),
        ('clf', DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced'
        ))
    ])

    param_grid = {
        'clf__max_depth': [2, 3, 4],
        'clf__min_samples_leaf': [5, 8, 10],
        'clf__min_samples_split': [10, 15, 20],
        'clf__criterion': ['gini', 'entropy']
    }

    cv_folds = get_safe_cv_folds(y_train, max_cv=5)

    print(f"\nPoužívam GridSearchCV s cv={cv_folds}.")
    print("Používam RandomOverSampler (ROS) v pipeline.")
    print("ROS sa teda aplikuje správne iba na tréningovú časť každého CV foldu.")

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='f1_macro',
        n_jobs=-1,
        refit=True
    )

    print("Spúšťam GridSearchCV...")
    grid.fit(X_train, y_train)

    print(f"Najlepšie parametre: {grid.best_params_}")
    print(f"Najlepšie CV F1-macro: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


# --------------------------------------------------
# 7. Baseline model
# --------------------------------------------------
def evaluate_baseline(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    """
    Vyhodnotí triviálnu baseline: vždy predikuje najčastejšiu triedu.
    """
    baseline_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', DummyClassifier(strategy='most_frequent'))
    ])

    baseline_pipeline.fit(X_train, y_train)
    baseline_pred = baseline_pipeline.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    print("\n--- Baseline (najčastejšia trieda) ---")
    print(f"Baseline Accuracy: {baseline_acc * 100:.2f} %")

    return baseline_acc, baseline_pred


# --------------------------------------------------
# 8. Hlavná funkcia
# --------------------------------------------------
def main():
    # without merged sleep time
    # FILE_PATH = "Dokazník_merged_adjusted_v3.csv"
    # FILE_PATH = "datasets/sleep/dokaznik_merged_wo_date_and_time.csv"
    # FILE_PATH = "datasets/sleep/Dokazník_merged_wo_datetime_feeling_today.csv"
    # TARGET_COL = "Koľko hodín v priemere spíte cez pracovný deň?"

    # with merged time
    # FILE_PATH = "datasets/sleep/Dokazník_merged_wo_datetime_feeling_today_sleep_avg.csv"
    # FILE_PATH = "datasets/sleep/Dokazník_merged_wo_dateTime_feelingToday_sleepAvg_sumBike.csv"
    FILE_PATH = "datasets/sleep/Dokazník_merged_wo_dateTime_feelingToday_sleepAvg_sumBike_deletedDayOfExcercise.csv"
    # FILE_PATH = "datasets/sleep/Dokazník_merged_wo_dateTime_feelingToday_sleepAvg_sumBike_deletedDayOfExcercise_mergedActivityTime.csv"
    TARGET_COL = "Koľko hodín v priemere spíte?"

    # FILE_PATH = "datasets/Dokazník_merged_wo_dateTime_feelingToday_sleepAvg_sumBike_deletedDayOfExcercise_mergedActivityTime.csv"
    # TARGET_COL = "Koľko hodín spíte v priemere"

    df = load_and_preprocess_data(FILE_PATH, TARGET_COL)

    # Rozdelenie na vstupy a cieľ
    X = df.drop(columns=["sleep_class"])
    y = df["sleep_class"]

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

    # Hľadanie najlepšieho modelu
    best_model, best_params, best_cv_score = find_best_model(X_train, y_train)

    # Predikcia a vyhodnotenie
    y_pred = best_model.predict(X_test)
    class_names = ["Málo spím", "Normálne", "Veľa spím"]

    model_acc = accuracy_score(y_test, y_pred)

    print("\n--- Výsledok na testovacích dátach ---")
    print(f"Accuracy: {model_acc * 100:.2f} %")
    print(f"Rozdiel oproti baseline: {(model_acc - baseline_acc) * 100:.2f} p. b.")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # --- MATICA ZÁMIEN ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predpovedané')
    plt.ylabel('Skutočné')
    plt.title('Matica zámen (Confusion Matrix)')
    plt.tight_layout()
    plt.savefig("outputs/matrix_scoring.png", bbox_inches="tight")
    plt.show()

    # --- FEATURE IMPORTANCE ---
    tree_model = best_model.named_steps['clf']

    plt.figure(figsize=(10, 6))
    importances = pd.Series(tree_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    importances.head(10).sort_values().plot(kind='barh', color='skyblue')
    plt.title("Top 10 faktorov ovplyvňujúcich dĺžku spánku")
    plt.xlabel("Dôležitosť (Gini importance)")
    plt.tight_layout()
    plt.savefig("outputs/factors_scoring.png", bbox_inches="tight")
    plt.show()

    print("\nTop 10 feature importances:")
    print(importances.head(10))

    # --- VIZUALIZÁCIA STROMU ---
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
    plt.title("Rozhodovací strom (vizualizácia prvých úrovní)")
    plt.tight_layout()
    plt.savefig("outputs/final_tree_classification_scoring.png", bbox_inches="tight")
    plt.show()

    print(f"\nHĺbka najlepšieho stromu: {tree_model.get_depth()}")
    print(f"Počet listov: {tree_model.get_n_leaves()}")

    print("\nNajlepšie parametre modelu:")
    print(best_params)

    print(f"\nNajlepšie CV F1-macro: {best_cv_score:.4f}")

    print("\nVýstupy boli uložené do priečinka 'outputs'.")


if __name__ == "__main__":
    main()