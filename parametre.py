from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Charger les données
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Séparer train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Définir le modèle
model = RandomForestClassifier(random_state=42)

# 4. Définir la grille d'hyperparamètres
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# 5. Recherche des meilleurs hyperparamètres
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 6. Meilleurs paramètres
print("Meilleurs hyperparamètres :", grid_search.best_params_)
print("Meilleur score CV :", grid_search.best_score_)

# 7. Évaluation sur le jeu de test
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nAccuracy sur test :", accuracy_score(y_test, y_pred))
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))
