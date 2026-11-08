import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
df = pd.read_csv("sonar_data.csv"); X = df.drop(columns="target").to_numpy(float); y = LabelEncoder().fit_transform(df["target"])
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42).split(X, y))
P = pd.read_pickle("results/oof_probs.pkl"); rows = []
for (name, tag), pm in P.items():
    if not pm: continue
    auc = [roc_auc_score(y[te] == 0, p) for (tr, te), p in zip(folds, pm)]
    ap = [average_precision_score(y[te] == 0, p) for (tr, te), p in zip(folds, pm)]
    rows.append({"Model": name, "Rep": tag, "ROC_AUC": 100*np.mean(auc), "AP": 100*np.mean(ap)})
A = pd.DataFrame(rows).pivot(index="Model", columns="Rep", values=["ROC_AUC", "AP"]).round(2)
print(A.sort_values(("AP","none"),ascending=False)); print(A.mean().round(2)); A.to_csv("results/auc_ap.csv")
