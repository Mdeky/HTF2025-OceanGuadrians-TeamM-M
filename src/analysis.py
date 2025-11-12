import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("../data/Ocean_Health_Index_2018_global_scores.csv")

# keep only numeric columns 
data = data.select_dtypes(include=['float64', 'int64']).dropna()

# X = data[['CW']]
X = data[['ECO', 'AO', 'CP', 'BD', 'FP', 'CS', 'FIS', 'HAB', 'SP', 'NP', 'TR', 'SPP', 'LSP', 'trnd_sc', 'ICO']]
y = data['Index_']

# filter out rows where any of the selected X columns are 0
X = X[(X != 0).all(axis=1)]
y = y.loc[X.index]  


# X_train, X_test, y_train, y_test = train_test_split(X[['ECO', 'CW']], y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X[['ECO', 'AO', 'CP', 'BD', 'FP', 'CS', 'FIS', 'HAB', 'SP', 'NP', 'TR', 'SPP', 'LSP', 'trnd_sc', 'ICO']], y, test_size=0.2, random_state=42)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# predict and evaluate
y_pred = model.predict(X_test)
print("R² score:", round(r2_score(y_test, y_pred), 3))
print("Eerste 5 voorspellingen:", y_pred[:5])

# plot predicted vs actual
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Echte waarden (Index_)")
plt.ylabel("Voorspelde waarden (Index_)")
plt.title("Voorspelde vs Echte Ocean Health Index")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
