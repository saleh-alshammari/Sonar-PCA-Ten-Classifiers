import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from experiments_models import models
df = pd.read_csv("sonar_data.csv"); X = df.drop(columns="target").to_numpy(float); y = LabelEncoder().fit_transform(df["target"])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 5, figsize=(16, 7))
for ax, (n, m) in zip(axes.ravel(), models().items()):
    cm = confusion_matrix(yte, make_pipeline(StandardScaler(), PCA(n_components=8), m).fit(Xtr, ytr).predict(Xte), labels=[0, 1])
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=17)
    for i in range(2):
        for j in range(2): ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > 8 else "black", fontsize=16)
    ax.set_xticks([0, 1], ["Mine", "Rock"], fontsize=12); ax.set_yticks([0, 1], ["Mine", "Rock"], fontsize=12)
    ax.set_title(n.replace("Naive", "Naïve"), fontsize=14); ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("True", fontsize=12)
fig.tight_layout(); fig.savefig("figures/f_confusion_v6.png", dpi=200); plt.close(fig)
d = pd.read_csv("results/pca_k_sweep.csv", index_col=0); cols = list(d.columns); xs = range(len(cols))
plt.rcParams.update({"font.size": 8})
fig, ax = plt.subplots(figsize=(6.0, 3.3))
for key, name, col, mk, ls, lw in [("MEAN", "Mean of ten models", "#222222", "o", "-", 2.2), ("SVM", "SVM", "#1f77b4", "s", "-", 1.6), ("MLP", "MLP", "#c26a00", "^", "--", 1.6), ("Naive Bayes", "Gaussian Naïve Bayes", "#8a3fa0", "D", ":", 1.6)]:
    ax.plot(xs, d.loc[key].values, color=col, marker=mk, ms=4.5, lw=lw, ls=ls, label=name)
ax.axvline(cols.index("8"), color="#999999", lw=0.8, ls="--"); ax.text(cols.index("8") + 0.1, 61, "8 components", fontsize=7, color="#555555", va="bottom")
ax.set_xticks(list(xs)); ax.set_xticklabels([c if c != "all60" else "60\n(no PCA)" for c in cols])
ax.set_xlabel("Number of principal components"); ax.set_ylabel("Mean CV accuracy (%)"); ax.set_ylim(55, 92); ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig("figures/f_pcak_v6.png", dpi=300); plt.close(fig)
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42).split(X, y))
P = pd.read_pickle("results/oof_probs.pkl"); yy = np.concatenate([y[te] for _, te in folds]) == 0
plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
show = [("SVM", "#1f77b4", "-"), ("MLP", "#c26a00", "--"), ("Extra Trees", "#2a9d5c", "-."), ("KNN", "#8a3fa0", ":"), ("AdaBoost", "#d62728", (0, (5, 1, 1, 1)))]
for ax, tag, title in [(axes[0], "none", "All 60 standardized features"), (axes[1], "pca8", "Eight principal components")]:
    for n, col, ls in show:
        s = np.concatenate(P[(n, tag)]); pr, rc, _ = precision_recall_curve(yy, s)
        ax.plot(rc, pr, color=col, ls=ls, lw=1.6, label=n)
    ax.axhline(yy.mean(), color="#999999", lw=0.8, ls="--"); ax.set_title(title, fontsize=10)
    ax.set_xlabel("Mine recall"); ax.set_xlim(0, 1.01); ax.set_ylim(0.5, 1.01); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
axes[0].set_ylabel("Mine precision")
fig.tight_layout(); fig.savefig("figures/f_pr_v6.png", dpi=300); plt.close(fig)
print("ok")
