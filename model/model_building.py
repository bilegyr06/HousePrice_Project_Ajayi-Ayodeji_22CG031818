import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import joblib  # Use sklearn's joblib instead


# 1. Load the Real Dataset
print("Downloading dataset from GitHub...")
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

# 2. Separate Features and Target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# 3. Define Features Types for Preprocessing
# We need to treat numeric and categorical (text) data differently
numeric_features = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income"
]
categorical_features = ["ocean_proximity"]

# 4. Build the Preprocessing Pipeline
# Numeric: Fill missing values with the median
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
    # No scaler here, as requested for Random Forest
])

# Categorical: Fill missing values with 'missing' and then OneHotEncode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine them into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. Create the Full Pipeline (Preprocessor + Model)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 6. Split Data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train the Pipeline
print("Training the pipeline (preprocessing + model)...")
model_pipeline.fit(X_train, y_train)

# 8. Evaluate
print("Evaluating...")
predictions = model_pipeline.predict(X_test)
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("Model Training Complete!")
print(f"R2 Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# 9. Save the Pipeline
joblib.dump(model_pipeline, './model/house_price_model.pkl', compress=3)
print("File 'house_price_model.pkl' saved successfully.")