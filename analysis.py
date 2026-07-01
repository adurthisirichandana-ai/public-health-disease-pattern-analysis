import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# STEP 1 — Load Dataset
data = pd.read_csv("data/Global Health Statistics.csv")
print("Dataset Loaded Successfully")
print("Original Shape:", data.shape)
# STEP 2 — Reduce dataset size (to make project faster)
sample_size = min(200000, len(data))
data = data.sample(n=sample_size, random_state=42)
print("Using sample size:", sample_size)
# STEP 3 — Select important columns
data = data[['Disease Name',
             'Country',
             'Gender',
             'Age Group',
             'Prevalence Rate (%)',
             'Mortality Rate (%)',
             'Population Affected']]
# STEP 4 — Create Risk Level
def risk_level(rate):
    if rate < 7:
        return "Low"
    elif rate < 14:
        return "Medium"
    else:
        return "High"
data["Risk Level"] = data["Prevalence Rate (%)"].apply(risk_level)
print("Columns after preprocessing:", data.columns)
print("\nRisk Level class distribution:")
print(data["Risk Level"].value_counts())
# STEP 5 — Encode categorical variables
le_disease = LabelEncoder()
le_country = LabelEncoder()
le_gender = LabelEncoder()
le_age = LabelEncoder()
data['Disease Name'] = le_disease.fit_transform(data['Disease Name'])
data['Country'] = le_country.fit_transform(data['Country'])
data['Gender'] = le_gender.fit_transform(data['Gender'])
data['Age Group'] = le_age.fit_transform(data['Age Group'])
# STEP 6 — Define features and target
# NOTE:
# Risk Level is created from "Prevalence Rate (%)".
# To avoid target leakage, we exclude prevalence-derived features from X.
X = data[['Disease Name',
          'Country',
          'Gender',
          'Age Group',
          'Mortality Rate (%)',
          'Population Affected']]

y = data['Risk Level']
# STEP 7 — Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training Started...")
# STEP 7.1 â€” Fix class imbalance only on training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# STEP 8 — Train models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forest": RandomForestClassifier(
        random_state=42,
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1
    )
}
results = {}
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    results[name] = {
        "accuracy": acc,
        "f1_macro": f1,
        "recall_macro": recall
    }
    print(f"\n{name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Recall   : {recall:.4f}")
# STEP 9 — Select best model automatically
best_model_name = max(results, key=lambda m: results[m]["accuracy"])
best_model = models[best_model_name]
print("\nBest Model:", best_model_name)
# STEP 9.1 — Explicit comparison for class imbalance handling
print("\nDecision Tree vs Random Forest:")
print(
    f"Decision Tree -> Accuracy: {results['Decision Tree']['accuracy']:.4f}, "
    f"F1-score: {results['Decision Tree']['f1_macro']:.4f}"
)
print(
    f"Random Forest -> Accuracy: {results['Random Forest']['accuracy']:.4f}, "
    f"F1-score: {results['Random Forest']['f1_macro']:.4f}"
)
# STEP 10 — Save Best Model
rf_has_highest_accuracy = results["Random Forest"]["accuracy"] >= max(
    result["accuracy"] for result in results.values()
)
if rf_has_highest_accuracy:
    print("Random Forest has the highest or tied-highest accuracy.")
else:
    print("Warning: Random Forest did not achieve the highest accuracy.")
pickle.dump(best_model, open("models/model.pkl", "wb"))
# STEP 11 — Save encoders
pickle.dump(le_disease, open("models/disease_encoder.pkl", "wb"))
pickle.dump(le_country, open("models/country_encoder.pkl", "wb"))
pickle.dump(le_gender, open("models/gender_encoder.pkl", "wb"))
pickle.dump(le_age, open("models/age_encoder.pkl", "wb"))
# STEP 12 — Save accuracy results (for dashboard comparison chart)
pickle.dump(results, open("models/model_results.pkl", "wb"))
print("Model and encoders saved successfully")
