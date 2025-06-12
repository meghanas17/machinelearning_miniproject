# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Load dataset
data = pd.read_csv("Rainfall.csv")
data.columns = data.columns.str.strip()
data = data.drop(columns=["day"])

# Handle missing values correctly (no chained assignment)
data["winddirection"] = data["winddirection"].fillna(data["winddirection"].mode()[0])
data["windspeed"] = data["windspeed"].fillna(data["windspeed"].median())

# Convert categorical to numeric
data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})

# -------------------------
#  EDA: Visualizations
# -------------------------

# Set plot style
sns.set(style="whitegrid")

# Histogram plots
plt.figure(figsize=(15, 10))
for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed'], 1):
    plt.subplot(3, 3, i)
    sns.histplot(data[column], kde=True)
    plt.title(f"Distribution of {column}")
plt.tight_layout()
plt.show()

# Countplot for Rainfall
plt.figure(figsize=(6, 4))
sns.countplot(x="rainfall", data=data)
plt.title("Rainfall Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------
#  Data Preprocessing
# -------------------------

# Drop correlated temperature features
data = data.drop(columns=['maxtemp', 'temparature', 'mintemp'])

# Balance dataset using downsampling
df_majority = data[data["rainfall"] == 1]
df_minority = data[data["rainfall"] == 0]
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
df_downsampled = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and target
X = df_downsampled.drop(columns=["rainfall"])
y = df_downsampled["rainfall"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
#  Model Training
# -------------------------

rf_model = RandomForestClassifier(random_state=42)
param_grid_rf = {
    "n_estimators": [50, 100],
    "max_features": ["sqrt"],
    "max_depth": [None, 10],
    "min_samples_split": [2],
    "min_samples_leaf": [1]
}
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=0)
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_

# -------------------------
# Evaluation
# -------------------------

cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5)
print("Mean CV Score:", np.mean(cv_scores))

y_pred = best_rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------
#  Prediction on New Data
# -------------------------

input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
input_df = pd.DataFrame([input_data], columns=X.columns)
prediction = best_rf_model.predict(input_df)
print("Prediction Result:", "Rainfall" if prediction[0] == 1 else "No Rainfall")

# -------------------------
# Save & Load Model
# -------------------------

model_data = {"model": best_rf_model, "feature_names": X.columns.tolist()}
with open("rainfall_prediction_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

# Load and test
with open("rainfall_prediction_model.pkl", "rb") as file:
    loaded_data = pickle.load(file)

loaded_model = loaded_data["model"]
loaded_features = loaded_data["feature_names"]
loaded_input_df = pd.DataFrame([input_data], columns=loaded_features)
loaded_prediction = loaded_model.predict(loaded_input_df)
print("Loaded Model Prediction:", "Rainfall" if loaded_prediction[0] == 1 else "No Rainfall")

