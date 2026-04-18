import pandas as pd

# Load dataset
data = pd.read_csv("flood_data.csv")

# -------------------------------
# STEP 1: Encode categorical data
# -------------------------------
from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
le2 = LabelEncoder()

data['Land Cover'] = le1.fit_transform(data['Land Cover'])
data['Soil Type'] = le2.fit_transform(data['Soil Type'])

# -------------------------------
# STEP 2: Features and Target
# -------------------------------
X = data.drop(columns=['flood', 'Latitude', 'Longitude'])
y = data['flood']

# Debug check (important)
print("Data types:\n", X.dtypes)

# -------------------------------
# STEP 3: Train-Test Split
# -------------------------------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 4: Train Model (Random Forest)
# -------------------------------
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------
# STEP 5: Accuracy
# -------------------------------
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# STEP 6: Test with custom input
# -------------------------------
sample = pd.DataFrame([{
    'rainfall': 200,
    'temperature': 30,
    'humidity': 85,
    'River Discharge (m³/s)': 500,
    'Water Level (m)': 10,
    'Elevation (m)': 100,
    'Land Cover': le1.transform(['Forest'])[0],
    'Soil Type': le2.transform(['Clay'])[0],
    'Population Density': 5000,
    'Infrastructure': 1,
    'Historical Floods': 1
}])

# Apply scaling
sample = scaler.transform(sample)

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Flood Risk: YES 🌊")
else:
    print("Flood Risk: NO ✅")