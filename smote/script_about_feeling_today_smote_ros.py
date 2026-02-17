import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError as exc:
    raise ImportError("Chýba balík 'imbalanced-learn'. Nainštaluj: pip3 install imbalanced-learn") from exc

import warnings
warnings.filterwarnings("ignore")


def auto_detect_columns_to_drop(df: pd.DataFrame) -> list[str]:
    cols_to_drop = []
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        uniq_ratio = df[col].nunique(dropna=False) / max(len(df), 1)
        if uniq_ratio > 0.95:
            cols_to_drop.append(col)
            print(f"[Auto-Drop] '{col}' -> ID/unikátny text (uniq_ratio={uniq_ratio:.3f})")
    return cols_to_drop


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


def build_sampler_and_cv(y_train: pd.Series):
    """
    Vyberie bezpečný sampler pre dané rozdelenie tried a nastaví CV tak,
    aby sa minimalizovalo padanie pri veľmi malých triedach.
    """
    counts = y_train.value_counts()
    min_class = int(counts.min())

    # Stratified CV je možné len ak máš min_class >= 2
    if min_class < 2:
        # úplne extrémny prípad -> bez stratify
        return RandomOverSampler(random_state=42), None

    # POZOR: SMOTE v multi-class s triedami 2-3 vzorky padá v CV,
    # lebo v tréning foldoch zostane len 1 vzorok.
    # Preto SMOTE použijeme iba ak je min_class dostatočne veľká.
    # Praktická hranica: min_class >= 6 (vtedy v 2-5 foldoch ostane v tréningu viac než 1-2 vzorky)
    if min_class >= 6:
        # zvoľ cv_folds tak, aby v tréningu foldov ostalo dosť minoritných vzoriek
        cv_folds = min(5, min_class)  # max 5 kvôli stabilite
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # bezpečné k pre SMOTE v tréning foldoch
        n_minority_fold = max(2, math.floor(min_class * (cv_folds - 1) / cv_folds))
        smote_k = max(1, min(5, n_minority_fold - 1))

        return SMOTE(random_state=42, k_neighbors=smote_k), cv

    # inak fallback: RandomOverSampler (neklope na susedov)
    cv_folds = min(5, min_class)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    return RandomOverSampler(random_state=42), cv


def find_best_model(X_train: pd.DataFrame, y_train: pd.Series) -> ImbPipeline:
    param_grid = {
        "clf__max_depth": [None, 3, 4, 5, 6, 8, 10, 12],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__min_samples_split": [2, 5, 10],
        "clf__max_features": [None, "sqrt", "log2"],
        "clf__criterion": ["gini", "entropy"],
    }

    sampler, cv = build_sampler_and_cv(y_train)

    model = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("sampler", sampler),
        ("clf", DecisionTreeClassifier(random_state=42, class_weight="balanced"))
    ])

    print("\nSpúšťam GridSearch...")
    print("Rozdelenie tried v tréningu:", y_train.value_counts().to_dict())
    print("Sampler:", type(sampler).__name__)
    if type(sampler).__name__ == "SMOTE":
        print("SMOTE k_neighbors:", sampler.k_neighbors)
    if cv is None:
        # fallback: bez CV (alebo jednoduché cv=2)
        cv = 2
        print("CV fallback: cv=2 (bez garancie stratify)")
    else:
        print(f"Používam StratifiedKFold: n_splits={cv.n_splits}")

    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        error_score="raise"  # aby si videl reálnu chybu, nie len "all fits failed"
    )

    grid.fit(X_train, y_train)

    print(f"Najlepšie parametre: {grid.best_params_}")
    print(f"Najlepšie CV f1_macro: {grid.best_score_:.4f}")

    return grid.best_estimator_


def main():
    FILE_PATH = "datasets/Dokazník_merged_adjusted.csv"
    TARGET_COL = "Ako sa dnes cítite po zdravotnej stránke od 1 po 10? (1 - zle, 10 - dobre)"

    X, y = load_and_preprocess_data(FILE_PATH, TARGET_COL)

    # ak máš triedy s 1 vzorkou, stratify sa nedá
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

    best_model = find_best_model(X_train, y_train)
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

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predpovedané")
    plt.ylabel("Skutočné")
    plt.title("Matica zámien (Confusion Matrix) - Feeling 1–10 (SMOTE/ROS fallback)")
    plt.tight_layout()
    plt.savefig("outputs/matrix_feeling_smote_or_ros.png", bbox_inches="tight")
    plt.show()

    tree = best_model.named_steps["clf"]
    importances = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    importances.head(10).sort_values().plot(kind="barh")
    plt.title("Top 10 faktorov ovplyvňujúcich zdravotný pocit (1–10) - SMOTE/ROS fallback")
    plt.xlabel("Dôležitosť (Gini importance)")
    plt.tight_layout()
    plt.savefig("outputs/factors_feeling_smote_or_ros.png", bbox_inches="tight")
    plt.show()

    tree_depth = tree.get_depth()
    tree_leaves = tree.get_n_leaves()
    print(f"\nSkutočná hĺbka stromu: {tree_depth}")
    print(f"Počet listov (leaves): {tree_leaves}")

    plot_depth = tree_depth if tree_depth <= 5 else 5
    print(f"Zobrazujem do hĺbky: {plot_depth}")

    plt.figure(figsize=(30, 14))
    plot_tree(
        tree,
        feature_names=X.columns,
        class_names=[str(i) for i in sorted(np.unique(y_train))],
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=plot_depth
    )
    plt.title(f"Rozhodovací strom (zobrazené prvé {plot_depth} úrovne z {tree_depth})")
    out_name = "outputs/final_tree_feeling_smote_or_ros.png"
    plt.savefig(out_name, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"\nStrom uložený ako '{out_name}'")


if __name__ == "__main__":
    main()
