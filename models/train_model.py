import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Load Dataset

df = pd.read_csv('dataset/stock_data.csv')

# Display Dataset

print("\nDataset Preview:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Check if dataset is empty

if df.empty:
    print("\nERROR: Dataset is empty!")
    exit()

# Check required columns

required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

for col in required_columns:
    if col not in df.columns:
        print(f"\nERROR: Column '{col}' not found in dataset")
        exit()

# Features and Target

X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Models

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'KNN': KNeighborsRegressor()
}

print("\nTraining Models...\n")

best_model = None
best_score = 999999

# Train Models

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    print(f"{name} MSE: {mse}")

    # Save best model

    if mse < best_score:
        best_score = mse
        best_model = model

# Save Best Model

joblib.dump(best_model, 'models/saved_model.pkl')

print("\nBest Model Saved Successfully!")

print("\nTraining Completed Successfully!")