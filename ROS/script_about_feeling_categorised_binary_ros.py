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
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError as exc:
    raise ImportError(
        "Chýba balík 'imbalanced-learn'. Nainštaluj: pip3 install imbalanced-learn"
    ) from exc

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
def categorize_feeling(score: int) -> int:
    """Diskretizácia zdravotného pocitu:
    0 = zle (1-5), 1 = dobre (6-10)
    """
    if score <= 5:
        return 0
    return 1


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

    y_raw = y.copy()
    y = y.apply(categorize_feeling).astype(int)

    # iba numerické vstupy
    X = X.select_dtypes(include=[np.number]).copy()

    # odstráň konštantné stĺpce
    nun = X.nunique(dropna=False)
    const_cols = nun[nun <= 1].index.tolist()
    if const_cols:
        print(f"[Auto-Drop] Konštantné stĺpce odstránené: {const_cols}")
        X = X.drop(columns=const_cols)

    print(f"Počet vzoriek: {len(X)} | Počet feature: {X.shape[1]}")
    print("Rozdelenie pôvodnej cieľovej premennej (1–10):")
    print(y_raw.value_counts().sort_index())
    print("\nPoužitá kategorizácia cieľa:")
    print("0 = zle (1–5), 1 = dobre (6–10)")
    print("Rozdelenie kategórií:")
    print(y.value_counts().sort_index())

    return X, y


# --------------------------------------------------
# 3) Hľadanie najlepšieho modelu (dynamické CV + ROS)
# --------------------------------------------------
def find_best_model(X_train: pd.DataFrame, y_train: pd.Series) -> ImbPipeline:
    param_grid = {
        "clf__max_depth": [None, 3, 5, 8, 12],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__min_samples_split": [2, 5, 10],
        "clf__max_features": [None, "sqrt"],
        "clf__criterion": ["gini", "entropy"],
    }

    # dynamické CV: nesmie byť viac foldov ako min počet vzoriek v triede
    min_class = int(y_train.value_counts().min())
    cv_folds = int(max(2, min(10, min_class)))  # ROS nemá požiadavku na susedov ako SMOTE
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    model = ImbPipeline(steps=[
        ("ros", RandomOverSampler(random_state=42)),
        ("clf", DecisionTreeClassifier(random_state=42))
    ])

    print("\nSpúšťam GridSearch...")
    print(f"Min počet vzoriek v triede (train): {min_class} -> používam cv={cv_folds}")
    print("Používam: RandomOverSampler (ROS)")
    print("Optimalizujem: f1_macro")

    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
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
    FILE_PATH = "datasets/Dokazník_feeling_wo_datetime.csv"
    FILE_PATH = "datasets/Dokazník_feeling_wo_datetime_feeling_today.csv"
    FILE_PATH = "datasets/Dokazník_feeling_wo_datetime_feeling_today_cycling.csv"
    FILE_PATH = "datasets/Dokazník_feeling_wo_datetime_feeling_today_cycling_only_minutes.csv"
    FILE_PATH = "datasets/Dokazník_feeling_wo_datetime_feeling_today_cycling_only_minutes_total_activ.csv"

    TARGET_COL = "Ako sa dnes cítite po zdravotnej stránke od 1 po 10? (1 - zle, 10 - dobre)"
    CLASS_NAMES = {
        0: "Zle (1–5)",
        1: "Dobre (6–10)",
    }

    X, y = load_and_preprocess_data(FILE_PATH, TARGET_COL)

    labels = sorted(CLASS_NAMES.keys())
    target_names = [CLASS_NAMES[i] for i in labels]

    # robustné rozdelenie train/test: stratify len ak to ide
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

    # imputácia chýbajúcich hodnôt
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)

    # model (ROS v pipeline)
    best_model = find_best_model(X_train_imp, y_train)
    y_pred = best_model.predict(X_test_imp)

    print("\n--- Výsledok na testovacích dátach ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f} %")
    print(f"MAE (priemerná odchýlka medzi kategóriami): {mean_absolute_error(y_test, y_pred):.3f}")
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
    plt.title("Matica zámien (Confusion Matrix) - kategórie zdravotného pocitu (ROS)")
    plt.tight_layout()
    plt.savefig("outputs/matrix_feeling_categorised_binary_ros.png", bbox_inches="tight")
    plt.show()

    # --- Feature importance ---
    importances = pd.Series(
        best_model.named_steps["clf"].feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    importances.head(10).sort_values().plot(kind="barh")
    plt.title("Top 10 faktorov ovplyvňujúcich kategóriu zdravotného pocitu (ROS)")
    plt.xlabel("Dôležitosť (Gini importance)")
    plt.tight_layout()
    plt.savefig("outputs/factors_feeling_categorised_binary_ros.png", bbox_inches="tight")
    plt.show()

    # --- Dynamická vizualizácia stromu podľa skutočnej hĺbky ---
    best_tree = best_model.named_steps["clf"]
    tree_depth = best_tree.get_depth()
    tree_leaves = best_tree.get_n_leaves()
    print(f"\nSkutočná hĺbka stromu: {tree_depth}")
    print(f"Počet listov (leaves): {tree_leaves}")

    plot_depth = tree_depth if tree_depth <= 5 else 5
    print(f"Zobrazujem do hĺbky: {plot_depth}")

    plt.figure(figsize=(30, 14))
    plot_tree(
        best_tree,
        feature_names=X.columns,
        class_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())],
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=plot_depth
    )
    plt.title(f"Rozhodovací strom (zobrazené prvé {plot_depth} úrovne z {tree_depth})")
    out_name = "outputs/final_tree_feeling_categorised_binary_ros.png"
    plt.savefig(out_name, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"\nStrom uložený ako '{out_name}'")


if __name__ == "__main__":
    main()
