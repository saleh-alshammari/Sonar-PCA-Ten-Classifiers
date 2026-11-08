import sys, time, warnings, numpy as np, pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "sonar_data.csv")
X = df.drop(columns="target").to_numpy(float)
y = LabelEncoder().fit_transform(df["target"])
outer = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7).split(X, y))

def raw(Z): return Z
def root(Z): return np.sqrt(Z)
def with_diff(Z): return np.hstack([Z, np.diff(Z, axis=1)])
def smooth(Z): return uniform_filter1d(Z, 3, axis=1)
ident, sqrt_t, diff_t, smooth_t = (FunctionTransformer(f) for f in (raw, root, with_diff, smooth))

def svm(): return Pipeline([("fe", ident), ("s", StandardScaler()), ("p", "passthrough"), ("c", SVC(kernel="rbf", random_state=42))])
Cg = {"c__C": [1, 2, 5, 10, 20, 50, 100], "c__gamma": ["scale", 0.003, 0.01, 0.02, 0.05]}
base = [("svm", Pipeline([("fe", sqrt_t), ("s", StandardScaler()), ("c", SVC(C=10, probability=True, random_state=42))])),
        ("et", ExtraTreesClassifier(n_estimators=500, max_features=0.3, random_state=42, n_jobs=1)),
        ("knn", Pipeline([("s", StandardScaler()), ("c", KNeighborsClassifier(n_neighbors=1))]))]
exps = {
    "SVM, wider grid": (svm(), dict(Cg)),
    "SVM, square root or first differences": (svm(), dict(Cg, fe=[ident, sqrt_t, diff_t])),
    "SVM, square root and PCA (20, 30)": (svm(), dict(Cg, fe=[ident, sqrt_t], p=["passthrough", PCA(20), PCA(30)])),
    "SVM, spectral smoothing": (Pipeline([("fe", ident), ("s", StandardScaler()), ("c", SVC(random_state=42))]), {"fe": [ident, smooth_t], "c__C": [2, 5, 10, 20, 50], "c__gamma": ["scale", 0.005, 0.01, 0.03]}),
    "SVM, univariate band selection": (Pipeline([("s", StandardScaler()), ("k", SelectKBest(f_classif)), ("c", SVC(random_state=42))]), {"k__k": [20, 30, 45, 60], "c__C": [2, 5, 10, 20], "c__gamma": ["scale", 0.01, 0.03]}),
    "KNN, square root and L1 distance": (Pipeline([("fe", ident), ("s", StandardScaler()), ("c", KNeighborsClassifier())]), {"fe": [ident, sqrt_t], "c__n_neighbors": [1, 3, 5], "c__weights": ["uniform", "distance"], "c__p": [1, 2]}),
    "Extra Trees, 500 trees, wider grid": (ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=1), {"max_features": ["sqrt", 0.1, 0.3, 0.5], "min_samples_leaf": [1, 2]}),
    "Random Forest, 500 trees, wider grid": (RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=1), {"max_features": ["sqrt", 0.1, 0.3], "min_samples_leaf": [1, 2]}),
    "Histogram gradient boosting": (HistGradientBoostingClassifier(random_state=42), {"learning_rate": [0.05, 0.1], "max_depth": [2, 3, None], "max_iter": [200, 400]}),
    "Soft voting (SVM, Extra Trees, KNN)": (VotingClassifier(base, voting="soft"), {"svm__c__C": [5, 10, 20]}),
    "Stacking (SVM, Extra Trees, KNN)": (StackingClassifier(base, final_estimator=LogisticRegression(), cv=5), {"svm__c__C": [5, 20]}),
}
rows = []
for name, (est, grid) in exps.items():
    start = time.time(); s = []
    for i, (tr, te) in enumerate(outer):
        gs = GridSearchCV(est, grid, cv=StratifiedKFold(5, shuffle=True, random_state=i), scoring="accuracy", n_jobs=-1).fit(X[tr], y[tr])
        p = gs.predict(X[te]); cm = confusion_matrix(y[te], p, labels=[0, 1])
        s.append((100 * (p == y[te]).mean(), 100 * cm[0, 0] / cm[0].sum(), 100 * cm[1, 0] / cm[1].sum()))
    s = np.array(s)
    rows.append({"Experiment": name, "Accuracy": s[:, 0].mean(), "SD": s[:, 0].std(ddof=1), "Mine recall": s[:, 1].mean(), "False alarm": s[:, 2].mean(), "Folds": ";".join(f"{a:.2f}" for a in s[:, 0])})
    print(f"{name:40s} {s[:, 0].mean():.2f} ± {s[:, 0].std(ddof=1):.2f}  recall {s[:, 1].mean():.2f}  FA {s[:, 2].mean():.2f}  ({time.time() - start:.0f} s)", flush=True)
pd.DataFrame(rows).round(2).to_csv("results/exploratory_nested.csv", index=False)
