import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


csv_path = "../data/Ocean_Health_Index_2018_global_scores.csv"  
target_col = "Index_"

feature_names = {
    "AO": "Artisanal Fisheries",
    "BD": "Biodiversity",
    "CP": "Coastal Protection",
    "CS": "Carbon Storage",
    "CW": "Clean Water",
    "ECO": "Economies",
    "FIS": "Wild Caught Fisheries",
    "FP": "Food Provision",
    "HAB": "Habitats",
    "ICO": "Iconic Species",
    "LE": "Livelihoods & Economies",
    "LIV": "Livelihoods",
    "LSP": "Lasting Special Places",
    "MAR": "Mariculture",
    "NP": "Natural Products",
    "SP": "Sense of Place",
    "SPP": "Species",
    "TR": "Tourism & Recreation",
    "trnd_sc": "Trend Score"
}


df = pd.read_csv(csv_path)
num = df.select_dtypes(include=[np.number]).copy()


features_present = [f for f in feature_names.keys() if f in num.columns]
if target_col not in num.columns:
    raise ValueError(f"Target column '{target_col}' not found.")
if not features_present:
    raise ValueError("No valid feature columns found in dataset.")


results = []
for feat in features_present:
    temp = num[[feat, target_col]].dropna()

    before = temp.shape[0]
    temp = temp[(temp[feat] != 0) & (temp[target_col] != 0)]
    after = temp.shape[0]
    removed_zeros = before - after

    if len(temp) < 5:
        results.append({
            "feature": feat,
            "full_name": feature_names[feat],
            "r2": np.nan,
            "n_samples": len(temp),
            "rows_removed_due_to_zeros": removed_zeros
        })
        continue

    X = temp[[feat]]
    y = temp[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "feature": feat,
        "full_name": feature_names[feat],
        "r2": r2,
        "n_samples": len(temp),
        "rows_removed_due_to_zeros": removed_zeros
    })

r2_df = pd.DataFrame(results).sort_values("r2", ascending=False, na_position="last").reset_index(drop=True)

# table
print("\nPer-feature R² (zeros removed per feature):")
print(r2_df[["feature", "full_name", "r2", "n_samples", "rows_removed_due_to_zeros"]].to_string(index=False))

# visualization
plt.figure(figsize=(12, 6))

bars = plt.bar(r2_df["full_name"], r2_df["r2"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("R² (single-feature linear model)")
plt.title("Per-feature R² predicting Ocean Health Index (zeros filtered per feature)")

for rect, val in zip(bars, r2_df["r2"]):
    if pd.isna(val):
        label = "NA"
        y = 0
    else:
        label = f"{val:.3f}"
        y = rect.get_height() + (0.01 if val >= 0 else -0.03)
    plt.text(rect.get_x() + rect.get_width()/2, y, label, ha="center", 
             va="bottom" if (not pd.isna(val) and val >= 0) else "top", fontsize=9)

plt.tight_layout()
plt.show()
