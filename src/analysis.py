import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("../data/Ocean_Health_Index_2018_global_scores.csv")
data = data.select_dtypes(include=['float64', 'int64']).dropna()

X = data[['']]   
y = data['Index_']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R² score:", round(r2_score(y_test, y_pred), 3))
print("Eerste 5 voorspellingen:", y_pred[:5])