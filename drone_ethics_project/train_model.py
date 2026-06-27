import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# =====================================
# LOAD DATASET
# =====================================

csv_path = os.path.join(os.path.dirname(__file__), "drone_ethics.csv")
df = pd.read_csv(csv_path)

# =====================================
# FEATURE ENGINEERING
# Add package priority as numeric
# =====================================

package_priority = {
    "medicine":    4,
    "documents":   3,
    "electronics": 2,
    "food":        1
}

landing_score_map = {
    "grass": 100,
    "soil":   90,
    "road":   70,
    "car":    40,
    "tree":   30
}

df["package_priority"] = df["package_type"].map(package_priority).fillna(1)
df["landing_score"]    = df["landing_zone"].map(landing_score_map).fillna(40)

# =====================================
# TRAIN / TEST SPLIT
# =====================================

X = df[["pedestrians", "cars", "animals", "package_priority", "landing_score"]]
y = df["choice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# TRAIN MODEL
# =====================================

model = DecisionTreeClassifier(random_state=42, max_depth=6)
model.fit(X_train, y_train)

# =====================================
# EVALUATE
# =====================================

y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 45)
print("   DRONE ETHICS MODEL — TRAINING COMPLETE")
print("=" * 45)
print(f"  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")
print(f"  Model accuracy   : {accuracy * 100:.1f}%")
print("=" * 45)

# =====================================
# SAVE MODEL
# =====================================

model_path = os.path.join(os.path.dirname(__file__), "drone_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"  Model saved to   : drone_model.pkl")
print("=" * 45)

# =====================================
# QUICK PREDICTION TEST
# =====================================

print("\n  QUICK TEST PREDICTIONS")
print("  " + "-" * 40)

test_cases = [
    ([1, 0, 0, 4, 100], "1 human, no cars, medicine, grass"),
    ([0, 3, 0, 2, 70],  "no humans, 3 cars, electronics, road"),
    ([0, 0, 2, 1, 90],  "no humans, 2 animals, food, soil"),
]

for features, desc in test_cases:
    pred = model.predict([features])[0]
    print(f"  {desc}")
    print(f"    → Decision: {pred}\n")