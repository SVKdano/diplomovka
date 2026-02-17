import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline  # dôležité: imblearn Pipeline

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

    # odstráň riadky bez cieľa
    df = df.dropna(subset=[target_col]).copy()

    # auto-drop vysoko unikátnych textových stĺpcov (ID, timestampy, poznámky...)
    bad_cols = auto_detect_columns_to_drop(df)
    if bad_cols:
        df = df.drop(columns=bad_cols)

    # y -> numeric -> int
    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df.drop(columns=[target_col])

    # ponechaj len validné hodnoty 1–10
    valid_mask = y.between(1, 10)
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(int)

    # iba numerické vstupy
    X = X.select_dtypes(include=[np.number]).copy()

    # odstráň konštantné stĺpce
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
# 3) Hľadanie najlepšieho modelu (CV-safe sampler v pipeline)
#    - SMOTE ak je to bezpečné, inak fallback na ROS
# --------------------------------------------------
def find_best_model_smote_or_ros(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    min_class = int(y_train.value_counts().min())

    # CV foldy: max 5, ale nie viac než min_class
    cv_folds = int(max(2, min(5, min_class)))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # SMOTE je bezpečné až keď máš aspoň 3 vzorky v menšinovej triede
    # (aby sa v tréningových foldoch nestratila trieda na 1 vzorku)
    use_smote = min_class >= 3

    if use_smote:
        k_neighbors = min(5, min_class - 1)
        sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
        sampler_name = f"SMOTE(k_neighbors={k_neighbors})"
    else:
        sampler = RandomOverSampler(random_state=42)
        sampler_name = "ROS(fallback)"

    print("\n--- Nastavenie vyvažovania tried ---")
    print(f"min_class(train)={min_class}")
    print(f"cv={cv_folds} (StratifiedKFold)")
    print(f"Používam: {sampler_name}")

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("sampler", sampler),
        ("clf", DecisionTreeClassifier(random_state=42))
    ])

    param_grid = {
        "clf__max_depth": [None, 3, 4, 5, 6, 8, 10, 12],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__min_samples_split": [2, 5, 10],
        "clf__max_features": [None, "sqrt", "log2"],
        "clf__criterion": ["gini", "entropy"],
    }

    print("\nSpúšťam GridSearch (optimalizujem f1_macro)...")

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        error_score="raise"
    )

    grid.fit(X_train, y_train)

    print(f"Najlepšie parametre: {grid.best_params_}")
    print(f"Najlepšie CV f1_macro: {grid.best_score_:.4f}")

    return grid.best_estimator_


# --------------------------------------------------
# 4) MAIN
# --------------------------------------------------
def main():
    os.makedirs("outputs", exist_ok=True)

    FILE_PATH = "datasets/Dokazník_feeling_wo_datetime_feeling_today_cycling_only_minutes_total_activ.csv"
    TARGET_COL = "Ako sa dnes cítite po zdravotnej stránke od 1 po 10? (1 - zle, 10 - dobre)"

    X, y = load_and_preprocess_data(FILE_PATH, TARGET_COL)

    # --- robustné rozdelenie train/test: stratify len ak to ide ---
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

    print("\nRozdelenie tried v train (pred vyvažovaním):")
    print(y_train.value_counts().sort_index())
    print("Rozdelenie tried v test:")
    print(y_test.value_counts().sort_index())

    # model (SMOTE alebo fallback ROS)
    best_model = find_best_model_smote_or_ros(X_train, y_train)

    # predikcia (pipeline si imputuje sám; sampler sa pri predict nepoužíva)
    y_pred = best_model.predict(X_test)

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

    # --- Confusion matrix ---
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names
    )
    plt.xlabel("Predpovedané")
    plt.ylabel("Skutočné")
    plt.title("Matica zámien (Confusion Matrix) - Feeling 1–10 (SMOTE/ROS)")
    plt.tight_layout()
    plt.savefig("outputs/matrix_feeling_smote_or_ros.png", bbox_inches="tight", dpi=300)
    plt.show()

    # --- Feature importance ---
    clf = best_model.named_steps["clf"]
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    importances.head(10).sort_values().plot(kind="barh")
    plt.title("Top 10 faktorov ovplyvňujúcich zdravotný pocit (1–10) – SMOTE/ROS")
    plt.xlabel("Dôležitosť (Gini importance)")
    plt.tight_layout()
    plt.savefig("outputs/factors_feeling_smote_or_ros.png", bbox_inches="tight", dpi=300)
    plt.show()

    # --- Dynamická vizualizácia stromu podľa skutočnej hĺbky ---
    tree_depth = clf.get_depth()
    tree_leaves = clf.get_n_leaves()
    print(f"\nSkutočná hĺbka stromu: {tree_depth}")
    print(f"Počet listov (leaves): {tree_leaves}")

    plot_depth = tree_depth if tree_depth <= 5 else 5
    print(f"Zobrazujem do hĺbky: {plot_depth}")

    plt.figure(figsize=(30, 14))
    plot_tree(
        clf,
        feature_names=X.columns,
        class_names=[str(i) for i in sorted(np.unique(y_train))],
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=plot_depth
    )
    plt.title(f"Rozhodovací strom – prvé {plot_depth} úrovne z {tree_depth} (SMOTE/ROS)")
    out_name = "outputs/final_tree_feeling_smote_or_ros.png"
    plt.savefig(out_name, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"\nStrom uložený ako '{out_name}'")


if __name__ == "__main__":
    main()
