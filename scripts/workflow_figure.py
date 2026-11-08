import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(7.8, 6.6)); ax.set_xlim(0, 10); ax.set_ylim(0.3, 10.5); ax.axis("off")
BLUE, ORANGE, GRAY, GREEN = "#dbe9f6", "#fde3c8", "#eeeeee", "#dff1e3"
def box(x, y, w, h, text, fc, bold=False, fs=8.6):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="#333333", lw=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1))
box(5, 10.0, 7.4, 0.7, "UCI Sonar dataset: 208 returns, 60 band energies (111 mines, 97 rocks)", GRAY, True)
box(5, 8.95, 7.4, 0.8, "Pipeline fitted on training data or training folds only:\nStandardScaler  ->  PCA (k components, or none)  ->  classifier", GRAY)
arrow(5, 9.65, 5, 9.35)
box(5, 7.55, 8.6, 0.95, "1. Nested CV (primary estimate): 3 x 5 outer folds; inner 5-fold grid search over\nhyperparameters and k in {5, 8, 12, 20, none}", GREEN, True)
box(5, 6.2, 8.6, 0.95, "2. Repeated stratified CV, 10 x 5 folds, fixed settings: 8 components, 8 whitened components,\nall 60 features; corrected t-test with Holm adjustment", BLUE)
box(5, 4.95, 8.6, 0.8, "3. Cost-sensitive thresholds and ROC/PR analysis on the same folds;\nthreshold chosen by inner 5-fold CV on each training fold", BLUE)
box(5, 3.8, 8.6, 0.7, "4. Component sweep: 5 x 5 folds, k = 2 to 30 and all 60 features", BLUE)
box(5, 2.75, 8.6, 0.8, "5. Blocked 5-fold CV: contiguous blocks of returns in file order", BLUE)
box(5, 1.6, 8.6, 0.8, "6. Stratified 85/15 hold-out split (176 / 32 returns): confusion matrices, Wilson\nintervals, exact McNemar tests", ORANGE)
for y1, y2 in [(8.55, 8.03)]:
    arrow(5, y1, 5, y2)
ax.text(5, 0.6, "No step that is fitted to data (scaler, PCA, tuning, threshold) sees the returns it is evaluated on.", ha="center", fontsize=8, style="italic")
fig.tight_layout(); fig.savefig("figures/f_workflow_v6.png", dpi=250)
