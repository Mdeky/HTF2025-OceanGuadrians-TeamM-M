import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------------------------------------------------
# 1. Data inladen en voorbereiden
# ---------------------------------------------------------
data = pd.read_csv("../data/Ocean_Health_Index_2018_global_scores.csv")

# Enkel numerieke kolommen behouden
data = data.select_dtypes(include=['float64', 'int64'])

# Kolommen kiezen
factor = 'AO'
target = 'Index_'

# Verwijder rijen waar factor of target nul of NaN is
data = data[(data[factor] != 0) & (data[target] != 0)]
data = data.dropna(subset=[factor, target])

# Data splitsen
X = data[[factor]]
y = data[target]

# ---------------------------------------------------------
# 2. Train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 3. Model trainen
# ---------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. Voorspellingen maken en prestaties berekenen
# ---------------------------------------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"R² score: {r2:.3f}")
print("Eerste 5 voorspellingen:", y_pred[:5])

# ---------------------------------------------------------
# 5. Visualisatie: Scatterplot + regressielijn
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Werkelijke waarden')
plt.plot(X_test, y_pred, color='red', label='Voorspelde lijn', linewidth=2)
plt.xlabel(factor)
plt.ylabel("Index_ (Ocean Health Index)")
plt.title(f"Relatie tussen {factor} en Ocean Health Index (zonder nullen)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 6. Interpretatie (voor je presentatie)
# ---------------------------------------------------------
"""
Interpretatie:
- Nulwaarden (die eigenlijk ontbrekende data voorstellen) zijn genegeerd.
- De R²-score toont nu een realistischer beeld van de relatie tussen Coastal Protection en de Ocean Health Index.
- Zo krijg je een zuivere correlatie zonder vervorming door ontbrekende data.
"""
