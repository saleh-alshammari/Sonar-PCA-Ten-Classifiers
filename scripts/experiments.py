import warnings, time, sys
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import binomtest, spearmanr
from sklearn.base import clone
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
OUT = "results/"
df = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "sonar_data.csv")
X = df.drop(columns="target").to_numpy(float)
y = LabelEncoder().fit_transform(df["target"])
PART = sys.argv[1] if len(sys.argv) > 1 else "all"

def models():
    return {
        "SVM": SVC(kernel="rbf", C=2, probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=5, random_state=42),
        "MLP": MLPClassifier(max_iter=2000, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=120, max_depth=5, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=120, learning_rate=0.12, max_depth=2, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_depth=3, random_state=42),
        "Logistic Regression": LogisticRegression(),
        "AdaBoost": AdaBoostClassifier(n_estimators=120, learning_rate=0.1, random_state=42),
        "Naive Bayes": GaussianNB(),
    }

def pipe(m, k, whiten=False):
    s = [("s", StandardScaler())] + ([("p", PCA(n_components=k, whiten=whiten))] if k else []) + [("c", clone(m))]
    return Pipeline(s)

conv_fail = 0
def fit(p, Xa, ya):
    global conv_fail
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p.fit(Xa, ya)
        conv_fail += sum(issubclass(x.category, ConvergenceWarning) for x in w)
    return p

def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); cur = 0
    for i, j in enumerate(o):
        cur = max(cur, min(1, (m - i) * p[j])); adj[j] = cur
    return adj

def cttest(d, ntr, nte):
    d = np.asarray(d, float); n = len(d); v = d.var(ddof=1)
    if v == 0: return 1.0
    t = d.mean() / np.sqrt((1 / n + nte / ntr) * v)
    return 2 * stats.t.sf(abs(t), df=n - 1)

def wilson(k, n, z=1.959964):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)

def mcnemar(a, b, yt):
    ca, cb = a == yt, b == yt; n01 = int((ca & ~cb).sum()); n10 = int((~ca & cb).sum())
    return binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

if PART in ("all", "holdout"):
    rows = []; preds = {}
    for name, m in models().items():
        r = {"Model": name}
        for k, tag in ((8, "PCA"), (None, "noPCA")):
            p = fit(pipe(m, k), Xtr, ytr); pr = p.predict(Xte); preds[(name, tag)] = pr
            cm = confusion_matrix(yte, pr, labels=[0, 1])
            r[f"acc_{tag}"] = 100 * accuracy_score(yte, pr)
            r[f"prec_{tag}"] = 100 * precision_score(yte, pr, average="macro")
            r[f"rec_{tag}"] = 100 * recall_score(yte, pr, average="macro")
            r[f"f1_{tag}"] = 100 * f1_score(yte, pr, average="macro")
            r[f"missed_{tag}"] = int(cm[0, 1]); r[f"fa_{tag}"] = int(cm[1, 0])
        r["mcnemar_p"] = mcnemar(preds[(name, "PCA")], preds[(name, "noPCA")], yte)
        rows.append(r)
    H = pd.DataFrame(rows).round(2); H.to_csv(OUT + "holdout.csv", index=False)
    print(H.to_string(index=False)); print("means", H[["acc_PCA", "acc_noPCA"]].mean().round(2).to_dict())
    print("sum missed/fa PCA", H.missed_PCA.sum(), H.fa_PCA.sum(), "noPCA", H.missed_noPCA.sum(), H.fa_noPCA.sum())
    sv = preds[("SVM", "PCA")]
    for o in ["Random Forest", "MLP"]:
        print("McNemar SVM vs", o, round(mcnemar(sv, preds[(o, "PCA")], yte), 3))
    print("Wilson 31/32", [round(v, 1) for v in wilson(31, 32)], "27/32", [round(v, 1) for v in wilson(27, 32)], "29/32", [round(v, 1) for v in wilson(29, 32)])
    for wh in (False, True):
        print("whiten", wh, {n: round(100 * (fit(pipe(m, 8, wh), Xtr, ytr).predict(Xte) == yte).mean(), 2) for n, m in models().items() if n in ("SVM", "KNN", "MLP", "Logistic Regression")})
    print("convergence warnings so far", conv_fail)

folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42).split(X, y))
ntr, nte = len(folds[0][0]), len(folds[0][1])

def foldscores(yt, pr):
    cm = confusion_matrix(yt, pr, labels=[0, 1])
    return 100 * (pr == yt).mean(), 100 * cm[0, 0] / cm[0].sum(), 100 * cm[1, 0] / cm[1].sum(), int(cm[0, 1])

if PART in ("all", "cv"):
    t0 = time.time(); rows = []; P = {}
    for name, m in models().items():
        res = {}
        for tag, k, wh in (("pca8", 8, False), ("none", None, False), ("white8", 8, True)):
            sc = []; pm = []
            for tr, te in folds:
                p = fit(pipe(m, k, wh), X[tr], y[tr]); pr = p.predict(X[te])
                sc.append(foldscores(y[te], pr))
                if tag != "white8": pm.append(p.predict_proba(X[te])[:, 0])
            res[tag] = np.array(sc); P[(name, tag)] = pm
        a, b, w = res["pca8"], res["none"], res["white8"]
        rows.append({"Model": name, "acc8": a[:, 0].mean(), "sd8": a[:, 0].std(ddof=1), "acc0": b[:, 0].mean(), "sd0": b[:, 0].std(ddof=1),
                     "p_acc": cttest(a[:, 0] - b[:, 0], ntr, nte), "rec8": a[:, 1].mean(), "rec0": b[:, 1].mean(), "p_rec": cttest(a[:, 1] - b[:, 1], ntr, nte),
                     "fa8": a[:, 2].mean(), "fa0": b[:, 2].mean(), "p_fa": cttest(a[:, 2] - b[:, 2], ntr, nte),
                     "accW": w[:, 0].mean(), "sdW": w[:, 0].std(ddof=1), "p_W": cttest(w[:, 0] - a[:, 0], ntr, nte), "recW": w[:, 1].mean()})
        print(name, round(time.time() - t0), flush=True)
    C = pd.DataFrame(rows)
    for c in ("p_acc", "p_rec", "p_fa", "p_W"): C[c + "_holm"] = holm(C[c])
    C.to_csv(OUT + "cv_repeated.csv", index=False)
    print(C.round(3).to_string(index=False))
    print("means", C[["acc8", "acc0", "rec8", "rec0", "fa8", "fa0", "accW"]].mean().round(2).to_dict())
    pd.to_pickle(P, OUT + "oof_probs.pkl")
    print("convergence warnings so far", conv_fail)

if PART in ("all", "sweep"):
    f25 = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42).split(X, y))
    T = {}
    for name, m in models().items():
        row = {}
        for k in [2, 4, 6, 8, 10, 12, 15, 20, 30, None]:
            row[str(k) if k else "all60"] = np.mean([100 * (fit(pipe(m, k), X[tr], y[tr]).predict(X[te]) == y[te]).mean() for tr, te in f25])
        T[name] = row; print(name, flush=True)
    S = pd.DataFrame(T).T; S.loc["MEAN"] = S.mean(); S = S.round(2); S.to_csv(OUT + "pca_k_sweep.csv"); print(S.to_string())
    print("convergence warnings so far", conv_fail)

if PART in ("all", "blocked"):
    block = np.empty(len(y), int)
    for c in (0, 1):
        idx = np.where(y == c)[0]
        for b, part in enumerate(np.array_split(idx, 5)): block[part] = b
    bf = [(np.where(block != b)[0], np.where(block == b)[0]) for b in range(5)]
    rows = []
    for name, m in models().items():
        r = {"Model": name}
        for k, tag in ((8, "PCA8"), (None, "noPCA")):
            sc = np.array([foldscores(y[te], fit(pipe(m, k), X[tr], y[tr]).predict(X[te])) for tr, te in bf])
            r[f"Blocked {tag}"] = sc[:, 0].mean(); r[f"SD {tag}"] = sc[:, 0].std(ddof=1); r[f"Mine recall {tag}"] = sc[:, 1].mean()
        rows.append(r)
    B = pd.DataFrame(rows).round(2); B.to_csv(OUT + "blocked_cv.csv", index=False); print(B.to_string(index=False))
    print("means", B.drop(columns="Model").mean().round(2).to_dict())
    print("convergence warnings so far", conv_fail)

if PART in ("all", "cost"):
    rows = []
    for name, m in models().items():
        for k, tag in ((8, "PCA8"), (None, "noPCA")):
            sc = {c: [] for c in (1, 5, 10)}
            for tr, te in folds:
                inner = StratifiedKFold(5, shuffle=True, random_state=0)
                pin = cross_val_predict(pipe(m, k), X[tr], y[tr], cv=inner, method="predict_proba")[:, 0]
                pt = fit(pipe(m, k), X[tr], y[tr]).predict_proba(X[te])[:, 0]
                ths = np.unique(np.r_[0, pin, 1])
                for c in sc:
                    costs = [(c * ((pin < t) & (y[tr] == 0)).sum() + ((pin >= t) & (y[tr] == 1)).sum()) for t in ths]
                    t = ths[int(np.argmin(costs))]
                    pr = np.where(pt >= t, 0, 1)
                    acc, rec, fa, missed = foldscores(y[te], pr)
                    fa_n = int(((pr == 0) & (y[te] == 1)).sum())
                    sc[c].append((acc, rec, fa, (c * missed + fa_n) / len(te), t))
            for c, v in sc.items():
                v = np.array(v)
                rows.append({"Model": name, "Rep": tag, "c": c, "acc": v[:, 0].mean(), "mine_recall": v[:, 1].mean(), "false_alarm": v[:, 2].mean(), "cost": v[:, 3].mean(), "threshold": np.median(v[:, 4])})
        print(name, flush=True)
    K = pd.DataFrame(rows).round(3); K.to_csv(OUT + "cost_threshold.csv", index=False); print(K.to_string(index=False))
