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
# 2. BINÁRNA DISKRETIZÁCIA CIEĽA (VARIANT A)
# --------------------------------------------------
def discretize_sleep(hours):
    if pd.isna(hours):
        return np.nan
    return 0 if hours < 7 else 1
    # 0 = málo spí
    # 1 = spí dosť / OK

# --------------------------------------------------
# 3. Načítanie a predspracovanie dát
# --------------------------------------------------
def load_and_preprocess_data(file_path, target_col):
    print(f"Načítavam súbor: {file_path}")
    df = pd.read_csv(file_path, sep=';', decimal=',')

    df = df.dropna(subset=[target_col])

    bad_cols = auto_detect_columns_to_drop(df)
    if bad_cols:
        df = df.drop(columns=bad_cols)

    df_numeric = df.select_dtypes(include=[np.number]).copy()

    df_numeric["sleep_class"] = df_numeric[target_col].apply(discretize_sleep)
    df_numeric = df_numeric.drop(columns=[target_col])

    return df_numeric

# --------------------------------------------------
# 4. GridSearch – binárny klasifikátor
# --------------------------------------------------
def find_best_model(X_train, y_train):
    param_grid = {
        'max_depth': [3, 4, 5, 6, None],
        'min_samples_leaf': [1, 5, 10],
        'min_samples_split': [2, 5, 10],
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
        scoring='f1',   # lepšie než accuracy pri binárnej klasifikácii
        n_jobs=-1
    )

    print("\nSpúšťam GridSearch (CV=10, optimalizujem F1)...")
    grid.fit(X_train, y_train)

    print(f"Najlepšie parametre: {grid.best_params_}")
    print(f"CV F1 skóre: {grid.best_score_:.4f}")

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

    best_model = find_best_model(X_train, y_train)

    y_pred = best_model.predict(X_test)

    class_names = ["Málo spím", "Spím dosť"]

    print("\n--- Výsledok na testovacích dátach ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f} %")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --- Confusion matrix ---
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predpovedané")
    plt.ylabel("Skutočné")
    plt.title("Matica zámien – binárna klasifikácia")
    plt.show()

    # --- Feature importance ---
    plt.figure(figsize=(8, 5))
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    importances.sort_values().plot(kind='barh')
    plt.title("Dôležitosť vstupných premenných")
    plt.tight_layout()
    plt.show()

    # --- Vizualizácia stromu ---
    plt.figure(figsize=(22, 10))
    plot_tree(
        best_model,
        feature_names=X.columns,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=9
    )
    plt.title("Rozhodovací strom – binárna klasifikácia")
    plt.show()


if __name__ == "__main__":
    main()
