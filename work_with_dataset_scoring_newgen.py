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
    # (Ak by boli potrebné aj kategorizované netextové atribúty, bolo by treba encoding.)
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    print(f"Tvar numerických dát: {df_numeric.shape}")

    # Diskretizácia cieľovej premennej
    df_numeric["sleep_class"] = df_numeric[target_col].apply(discretize_sleep)

    # Odstránenie pôvodnej cieľovej premennej -> prevencia leakage
    df_numeric = df_numeric.drop(columns=[target_col])

    print(f"Tvar dát po vytvorení 'sleep_class': {df_numeric.shape}")
    print("Rozdelenie tried v 'sleep_class':")
    print(df_numeric["sleep_class"].value_counts(dropna=False).sort_index())

    return df_numeric


# --------------------------------------------------
# 5. Hľadanie najlepšieho klasifikátora
# --------------------------------------------------
def find_best_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """
    Nájde najlepšie parametre rozhodovacieho stromu pomocou GridSearchCV.
    Optimalizácia prebieha podľa F1-macro.
    """
    param_grid = {
        'max_depth': [None, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 5, 10],
        'min_samples_split': [2, 5, 10],
        'max_features': [None, 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    }

    clf = DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced'
    )

    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=10,
        scoring='f1_macro',
        n_jobs=-1
    )

    print("\nSpúšťam GridSearchCV (cv=10, scoring='f1_macro')...")
    grid.fit(X_train, y_train)

    print(f"Najlepšie parametre: {grid.best_params_}")
    print(f"Najlepšie CV F1-macro: {grid.best_score_:.4f}")

    return grid.best_estimator_


# --------------------------------------------------
# 6. Hlavná funkcia
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
    # FILE_PATH = "datasets/sleep/Dokazník_merged_wo_dateTime_feelingToday_sleepAvg_sumBike_deletedDayOfExcercise.csv"
    FILE_PATH = "datasets/sleep/Dokazník_merged_wo_dateTime_feelingToday_sleepAvg_sumBike_deletedDayOfExcercise_mergedActivityTime.csv"
    TARGET_COL = "Koľko hodín v priemere spíte?"

    #FILE_PATH = "datasets/Dokazník_merged_wo_dateTime_feelingToday_sleepAvg_sumBike_deletedDayOfExcercise_mergedActivityTime.csv"
    #TARGET_COL = "Koľko hodín spíte v priemere"

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

    # Imputácia chýbajúcich hodnôt vo vstupných dátach
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

    best_model = find_best_model(X_train, y_train)

    # Predikcia a vyhodnotenie
    y_pred = best_model.predict(X_test)
    class_names = ["Málo spím", "Normálne", "Veľa spím"]

    print("\n--- Výsledok na testovacích dátach ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f} %")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --- FEATURE IMPORTANCE ---
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    print("\nTop 10 feature importances:")
    print(importances.sort_values(ascending=False).head(10))

    # --- TEXTOVÝ VÝPIS STROMU DO KONZOLY ---
    print("\n--- TEXTOVÝ VÝPIS ROZHODOVACIEHO STROMU ---")
    tree_rules = export_text(
        best_model,
        feature_names=list(X.columns),
        max_depth=4
    )
    print(tree_rules)

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

    # --- FEATURE IMPORTANCE GRAF ---
    plt.figure(figsize=(10, 6))
    importances.nlargest(10).sort_values().plot(kind='barh', color='skyblue')
    plt.title("Top 10 faktorov ovplyvňujúcich dĺžku spánku")
    plt.xlabel("Dôležitosť (Gini importance)")
    plt.tight_layout()
    plt.savefig("outputs/factors_scoring.png", bbox_inches="tight")
    plt.show()

    # --- VIZUALIZÁCIA STROMU ---
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
    plt.title("Rozhodovací strom (vizualizácia prvých úrovní)")
    plt.tight_layout()
    plt.savefig("outputs/final_tree_classification_scoring.png", bbox_inches="tight")
    plt.show()

    print("\nVýstupy boli uložené do priečinka 'outputs'.")


if __name__ == "__main__":
    main()