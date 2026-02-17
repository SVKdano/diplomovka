import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error

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
    0 = slabšie (1-4), 1 = dobre (5-7), 2 = výborne (8-10)
    """
    if score <= 4:
        return 0
    if score <= 7:
        return 1
    return 2


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
    print("0 = slabšie (1–6), 1 = dobre (7–8), 2 = výborne (9–10)")
    print("Rozdelenie kategórií:")
    print(y.value_counts().sort_index())

    return X, y


# --------------------------------------------------
# 3) Hľadanie najlepšieho modelu (dynamické CV)
# --------------------------------------------------
def find_best_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    param_grid = {
        "max_depth": [None, 3, 4, 5, 6, 8, 10, 12],
        "min_samples_leaf": [1, 2, 5, 10],
        "min_samples_split": [2, 5, 10],
        "max_features": [None, "sqrt", "log2"],
        "criterion": ["gini", "entropy"],
    }

    clf = DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    )

    # dynamické CV: nesmie byť viac foldov ako min počet vzoriek v triede
    min_class = int(y_train.value_counts().min())
    cv_folds = int(max(2, min(10, min_class)))  # aspoň 2, max 10

    print("\nSpúšťam GridSearch...")
    print(f"Min počet vzoriek v triede (train): {min_class} -> používam cv={cv_folds}")
    print("Optimalizujem: f1_macro")

    grid = GridSearchCV(
        clf,
        param_grid,
        cv=cv_folds,
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
    FILE_PATH = "datasets/Dokazník_feeling_wo_datetime_feeling_today.csv"
    FILE_PATH = "datasets/Dokazník_feeling_wo_datetime_feeling_today_cycling.csv"
    FILE_PATH = "datasets/Dokazník_feeling_wo_datetime_feeling_today_cycling_only_minutes.csv"
    FILE_PATH = "datasets/Dokazník_feeling_wo_datetime_feeling_today_cycling_only_minutes_total_activ.csv"
    TARGET_COL = "Ako sa dnes cítite po zdravotnej stránke od 1 po 10? (1 - zle, 10 - dobre)"
    CLASS_NAMES = {
        0: "Slabšie (1–6)",
        1: "Dobre (7–8)",
        2: "Výborne (9–10)",
    }

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

    # imputácia chýbajúcich hodnôt
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)

    # model
    best_model = find_best_model(X_train_imp, y_train)

    # predikcia
    y_pred = best_model.predict(X_test_imp)

    # labely
    labels = sorted(CLASS_NAMES.keys())
    target_names = [CLASS_NAMES[x] for x in labels]

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
    plt.title("Matica zámien (Confusion Matrix) - kategórie zdravotného pocitu")
    plt.tight_layout()
    plt.savefig("outputs/matrix_feeling_wo_sleep", bbox_inches="tight")
    plt.show()

    # --- Feature importance ---
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    importances.head(10).sort_values().plot(kind="barh")
    plt.title("Top 10 faktorov ovplyvňujúcich kategóriu zdravotného pocitu")
    plt.xlabel("Dôležitosť (Gini importance)")
    plt.tight_layout()
    plt.savefig("outputs/factors_feeling_wo_sleep.png", bbox_inches="tight")
    plt.show()

    # --- Dynamická vizualizácia stromu podľa skutočnej hĺbky ---
    tree_depth = best_model.get_depth()
    tree_leaves = best_model.get_n_leaves()
    print(f"\nSkutočná hĺbka stromu: {tree_depth}")
    print(f"Počet listov (leaves): {tree_leaves}")

    # zobraz celý, ak je malý, inak obmedz (napr. na 5)
    plot_depth = tree_depth if tree_depth <= 5 else 5
    print(f"Zobrazujem do hĺbky: {plot_depth}")

    plt.figure(figsize=(30, 14))
    plot_tree(
        best_model,
        feature_names=X.columns,
        class_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())],
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=plot_depth
    )
    plt.title(f"Rozhodovací strom (zobrazené prvé {plot_depth} úrovne z {tree_depth})")
    out_name = "outputs/final_tree_feeling_wo_feeling.png"
    plt.savefig(out_name, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"\nStrom uložený ako '{out_name}'")


if __name__ == "__main__":
    main()
