import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------
# Automatická detekcia ID / dátumových stĺpcov
# --------------------------------------------------
def auto_detect_columns_to_drop(df):
    cols_to_drop = []

    object_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in object_cols:
        if df[col].nunique() / len(df) > 0.95:
            cols_to_drop.append(col)
            print(f"[Auto-Drop] '{col}' -> ID / dátum")

    return cols_to_drop


# --------------------------------------------------
# Diskretizácia cieľovej premennej
# --------------------------------------------------
def discretize_sleep(hours):
    if hours <= 6:
        return 0   # málo
    elif hours <= 8:
        return 1   # normálne
    else:
        return 2   # veľa


# --------------------------------------------------
# Načítanie a predspracovanie dát
# --------------------------------------------------
def load_and_preprocess_data(file_path, target_col):
    print(f"Načítavam súbor: {file_path}")
    df = pd.read_csv(file_path, sep=';', decimal=',')

    bad_cols = auto_detect_columns_to_drop(df)
    if bad_cols:
        df = df.drop(columns=bad_cols)

    df = df.select_dtypes(include=[np.number])

    # diskretizácia cieľa
    df["sleep_class"] = df[target_col].apply(discretize_sleep)
    df = df.drop(columns=[target_col])

    return df


# --------------------------------------------------
# Hľadanie najlepšieho klasifikátora
# --------------------------------------------------
def find_best_model(X_train, y_train):

    param_grid = {
        'max_depth': [None, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 5, 10],
        'min_samples_split': [2, 5, 10],
        'max_features': [None, 'sqrt', 'log2']
    }

    clf = DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced'
    )

    grid = GridSearchCV(
        clf,
        param_grid,
        cv=10,
        scoring='accuracy',
        n_jobs=-1
    )

    print("\nSpúšťam GridSearch (optimalizujem accuracy)...")
    grid.fit(X_train, y_train)

    print(f"Najlepšie parametre: {grid.best_params_}")
    print(f"CV accuracy: {grid.best_score_:.4f}")

    return grid.best_estimator_


# --------------------------------------------------
# HLAVNÁ FUNKCIA
# --------------------------------------------------
def main():

    FILE_PATH = "Dokazník_merged_adjusted_v3.csv"
    TARGET_COL = "Koľko hodín v priemere spíte?"

    df = load_and_preprocess_data(FILE_PATH, TARGET_COL)

    X = df.drop(columns=["sleep_class"])
    y = df["sleep_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

    best_model = find_best_model(X_train, y_train)

    y_pred = best_model.predict(X_test)

    print("\n--- Výsledok na testovacích dátach ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f} %\n")
    print("Classification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Málo spím", "Normálne", "Veľa spím"]
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --------------------------------------------------
    # Vizualizácia stromu
    # --------------------------------------------------
    plt.figure(figsize=(26, 14))
    plot_tree(
        best_model,
        feature_names=X.columns,
        class_names=["Málo", "Normálne", "Veľa"],
        filled=True,
        rounded=True,
        fontsize=9
    )

    plt.title("Rozhodovací strom – klasifikácia dĺžky spánku")
    plt.savefig("final_tree_classification.png", bbox_inches="tight")
    plt.show()

    print("\nStrom uložený ako 'final_tree_classification.png'")


if __name__ == "__main__":
    main()
