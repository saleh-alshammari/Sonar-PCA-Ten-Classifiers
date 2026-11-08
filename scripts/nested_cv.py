import time, warnings
from collections import Counter
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
df = pd.read_csv("sonar_data.csv")
X = df.drop(columns="target").to_numpy(float)
y = LabelEncoder().fit_transform(df["target"])

def fold_scores(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return (100 * (y_true == y_pred).mean(), 100 * cm[0, 0] / cm[0].sum(), 100 * cm[1, 0] / cm[1].sum(), int(cm[0, 1]), int(cm[1, 0]))

grids = {
    "SVM": (SVC(kernel="rbf", random_state=42),
            {"clf__C": [0.5, 1, 2, 5, 10, 50], "clf__gamma": ["scale", 0.005, 0.01, 0.05, 0.1]}),
    "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=42),
                      {"clf__max_depth": [None, 5, 10], "clf__max_features": ["sqrt", 0.3]}),
    "MLP": (MLPClassifier(max_iter=2000, random_state=42),
            {"clf__hidden_layer_sizes": [(50,), (100,), (100, 50)], "clf__alpha": [1e-4, 1e-3, 1e-2, 1e-1]}),
    "Extra Trees": (ExtraTreesClassifier(n_estimators=200, random_state=42),
                    {"clf__max_depth": [None, 5, 10], "clf__max_features": ["sqrt", 0.3]}),
    "XGBoost": (XGBClassifier(subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1),
                {"clf__n_estimators": [100, 300], "clf__max_depth": [2, 4], "clf__learning_rate": [0.05, 0.1]}),
    "K-Nearest Neighbors": (KNeighborsClassifier(),
                            {"clf__n_neighbors": [1, 3, 5, 7, 9], "clf__weights": ["uniform", "distance"]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42),
                          {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3], "clf__learning_rate": [0.05, 0.1]}),
    "Logistic Regression": (LogisticRegression(max_iter=5000), {"clf__C": [0.01, 0.1, 1, 10, 100]}),
    "AdaBoost": (AdaBoostClassifier(random_state=42),
                 {"clf__n_estimators": [100, 300], "clf__learning_rate": [0.1, 0.5, 1.0]}),
    "Gaussian Naive Bayes": (GaussianNB(), {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}),
}
pca_options = [PCA(n_components=k) for k in [5, 8, 12, 20]] + ["passthrough"]

outer = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7).split(X, y))
start = time.time()
rows = []
for name, (est, grid) in grids.items():
    p = Pipeline([("scaler", StandardScaler()), ("pca", PCA()), ("clf", est)])
    scores, chosen, settings = [], Counter(), Counter()
    for i, (tr, te) in enumerate(outer):
        gs = GridSearchCV(p, dict(grid, pca=pca_options), cv=StratifiedKFold(5, shuffle=True, random_state=i),
                          scoring="accuracy", n_jobs=-1).fit(X[tr], y[tr])
        scores.append(fold_scores(y[te], gs.predict(X[te])))
        best = dict(gs.best_params_)
        chosen[getattr(best.pop("pca"), "n_components", "none")] += 1
        settings[str(sorted((k.replace("clf__", ""), v) for k, v in best.items()))] += 1
    s = np.array(scores)
    rows.append({"Model": name,
                 "Nested acc": f"{s[:, 0].mean():.2f} ± {s[:, 0].std(ddof=1):.2f}",
                 "Mine recall": round(s[:, 1].mean(), 2),
                 "False alarm": round(s[:, 2].mean(), 2),
                 "PCA chosen": f"{len(outer) - chosen['none']}/{len(outer)}",
                 "Components chosen": dict(chosen),
                 "Most chosen settings": settings.most_common(1)[0][0],
                 "_acc": s[:, 0].mean()})
    print(f"{name} done ({time.time() - start:.0f} s)")
table3 = pd.DataFrame(rows).sort_values("_acc", ascending=False)
table3.drop(columns="_acc").to_csv("results/cv_nested.csv", index=False)
print(table3.drop(columns="_acc").to_string(index=False))
print(f"\nMean: accuracy {table3['_acc'].mean():.2f}%  mine recall {table3['Mine recall'].mean():.2f}%  false alarm {table3['False alarm'].mean():.2f}%")

