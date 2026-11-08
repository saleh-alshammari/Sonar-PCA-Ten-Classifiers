import sys
import numpy as np, pandas as pd
from sklearn.metrics import pairwise_distances

path = sys.argv[1] if len(sys.argv) > 1 else "sonar_data.csv"
df = pd.read_csv(path)
X = df.drop(columns="target").to_numpy(float)
y = df["target"].to_numpy()

for c in ["R", "M"]:
    Z = X[y == c]
    consecutive = np.linalg.norm(Z[1:] - Z[:-1], axis=1).mean()
    others = np.mean([np.linalg.norm(Z[i] - Z[j]) for i in range(len(Z)) for j in range(i + 2, len(Z))])
    print(f"{c}: consecutive {consecutive:.3f}  non-adjacent {others:.3f}  ratio {consecutive / others:.2f}")

D = pairwise_distances(X)
np.fill_diagonal(D, np.inf)
nearest = D.argmin(axis=1)
share = (np.abs(nearest - np.arange(len(X))) == 1).mean()
print(f"Nearest other return is an adjacent row for {100 * share:.1f}% of the returns")
