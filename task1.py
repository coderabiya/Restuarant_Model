import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("zomato.csv")

# Drop unnecessary large text columns
df = df.drop(columns=[
    "url", "address", "phone", 
    "reviews_list", "menu_item", "dish_liked"
], errors="ignore")

# Remove rows where rate is missing
df = df.dropna(subset=["rate"])

# Clean rate column
df["rate"] = df["rate"].str.replace("/5", "", regex=False)
df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
df = df.dropna(subset=["rate"])

# Fill remaining missing values
df = df.fillna("Unknown")

# Convert all object columns to string before encoding
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].astype(str)

# Encode categorical columns
label_encoder = LabelEncoder()
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = label_encoder.fit_transform(df[col])

# Define features and target
X = df.drop("rate", axis=1)
y = df["rate"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)