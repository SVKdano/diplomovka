import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import KBinsDiscretizer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def auto_detect_columns_to_drop(df, target_col, threshold_corr=0.85):
    """
    Algorithm for detecting columns like date, time or ID
    and delete columns with high corellation

    :param df: Input in dataframe format
    :param target_col: Column in our dataset that we want to predict
    :param threshold_corr: treshold of corelation between columns
    """

    cols_to_drop = []
  
    object_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in object_cols:
        
        if df[col].nunique() / len(df) > 0.95:
            cols_to_drop.append(col)
            print(f" [Auto-Drop] '{col}' -> Looks like Id, date or time")

    numeric_df = df.select_dtypes(include=[np.number])
    if target_col in numeric_df.columns:
        correlations = numeric_df.corrwith(numeric_df[target_col]).abs()
        for col, corr_val in correlations.items():
            if col != target_col and corr_val > threshold_corr:
                cols_to_drop.append(col)
                print(f" [Auto-Drop] '{col}' -> Corellation ({corr_val:.2f}) was marked as high")
    
    return list(set(cols_to_drop))

def load_and_preprocess_data(file_path, target_col):
    """
    Load CSV, delete non-relevant columns and filter outliers
    
    :param file_path: Path to input csv
    :param target_col: Column in our dataset that we want to predict
    """
    print(f"Load input file: {file_path}")
    df = pd.read_csv(file_path, sep=';', decimal=',')
    
    print("Detecting high corellation columns and columns with unique items")
    bad_cols = auto_detect_columns_to_drop(df, target_col)
    
    if bad_cols:
        print(f"Vyhadzujem stĺpce: {bad_cols}")
        df_cleaned = df.drop(columns=bad_cols, errors='ignore')
    else:
        df_cleaned = df

    df_numeric = df_cleaned.select_dtypes(include=[np.number])

    # --- PRIDANÝ BINNING (Diskretizácia) ---
    # Premeníme spojité čísla na kategórie pre lepšiu stabilitu stromu
    cols_to_bin = [col for col in df_numeric.columns if df_numeric[col].nunique() > 5 and col != target_col]
    if cols_to_bin:
        kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', subsample=None)
        # Imputácia NaN hodnôt pred binningom
        df_numeric[cols_to_bin] = SimpleImputer(strategy='median').fit_transform(df_numeric[cols_to_bin])
        df_numeric[cols_to_bin] = kbd.fit_transform(df_numeric[cols_to_bin])

    return df_numeric

def find_best_model(X_train, y_train):
    """
    Finding best parameters for decision tree
    
    :param X_train: All other columns
    :param y_train: Result column
    """
    # Dynamický výpočet hĺbky podľa počtu otázok
    n_features = X_train.shape[1]

    param_grid = {
        # 'max_depth': list(range(2, n_features + 1)), 
        #'min_samples_leaf': list(range(1, 31)),
        #'min_samples_split': list(range(2, 31))
        'max_depth': [2, 3, 4], 
        'min_samples_leaf': [15, 20, 25],
        'min_samples_split': [2, 5, 10]
    }
    
    dt = DecisionTreeRegressor(random_state=42)
    
    grid_search = GridSearchCV(
        estimator=dt, 
        param_grid=param_grid, 
        cv=5, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1
    )

    print("\nSpúšťam tréning a hľadanie optimálnych parametrov...")
    grid_search.fit(X_train, y_train)
    
    print(f" -> Víťazné parametre: {grid_search.best_params_}")
    return grid_search.best_estimator_

def main():
    # Nastavenie ciest a cieľa
    #FILE_PATH = 'Dokazník_merged_adjusted_v3.csv' 
    #FILE_PATH = 'Dokazník_merged_adjusted_v2.csv' 
    FILE_PATH = 'Dokazník_merged_adjusted.csv' 
    #FILE_PATH = 'dokaznik_merged.csv' 
    
    #TARGET_COL = "Koľko hodín v priemere spíte?"
    TARGET_COL = "Koľko hodín v priemere spíte cez pracovný deň?"
    
    df = load_and_preprocess_data(FILE_PATH, TARGET_COL)

    if TARGET_COL not in df.columns:
        print(f"CHYBA: Cieľový stĺpec '{TARGET_COL}' sa v dátach nenašiel!")
        return
    
    X = df.drop(columns=[TARGET_COL], errors='ignore')
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
  
    best_model = find_best_model(X_train_imp, y_train)
    
    y_pred = best_model.predict(X_test_imp)
    r2 = r2_score(y_test, y_pred) * 100
    
    print(f"\n--- Výsledok na nezávislých testovacích dátach ---")
    print(f"R2 Skóre (Presnosť): {r2:.4f}")
    
    plt.figure(figsize=(25, 12))
    plot_tree(best_model, 
              feature_names=X.columns, 
              filled=True, 
              rounded=True, 
              fontsize=10, 
              precision=2)
    
    plt.title(f"Výsledný rozhodovací strom (R2: {r2:.2f})")
    plt.savefig('final_tree.png', bbox_inches='tight')
    print("\nGraf bol uložený ako 'final_tree.png'")

if __name__ == "__main__":
    main()