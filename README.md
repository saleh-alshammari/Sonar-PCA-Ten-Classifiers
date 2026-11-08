# Sonar Rock vs. Mine Classification: Ten Classifiers With and Without PCA

Code, data, and results for the paper:

> S. Al-Shammari, M. Alazmi, and A. Al-Shammari, "Ten Classifiers With and Without PCA for Sonar Mine-versus-Rock Discrimination on the UCI Benchmark," College of Computer Science and Engineering, University of Ha'il.

## About the study

The UCI Sonar dataset contains 208 sonar returns (111 from a metal cylinder used as a mock mine and 97 from rocks), each described by 60 frequency-band energies. Published accuracies on this benchmark come from different splits and preprocessing, so they cannot be compared directly. This study evaluates ten classifiers, each with and without principal component analysis (PCA), under one protocol in which the scaler, the PCA, hyperparameter tuning, and decision thresholds are fitted on training data only.

The main results are:

- Under nested cross-validation (the primary estimate), Extra Trees (87.17%) and an RBF support vector machine (86.19%) had the highest observed mean accuracies and mine recall.
- Eight principal components reached 97.5% of the mean accuracy of all 60 features, but they lowered the mean mine recall from 84.95% to 77.78%.
- With a decision threshold tuned for a missed mine that costs five false alarms, the SVM on all 60 features detected 97.30% of mines at a false alarm rate of 38.48%.
- Blocked cross-validation over contiguous ranges of recordings lowered the mean accuracy by about 15 percentage points.

## How to run

1. Install Python 3.13 from python.org (tick "Add python.exe to PATH").
2. On Windows, double-click `setup_windows.bat`. It creates the environment `.venv313` and installs the pinned packages. On other systems, run:
   ```
   python3.13 -m venv .venv313
   source .venv313/bin/activate
   pip install -r requirements.txt
   ```
3. Open `Sonar_Full_Project.ipynb` in VS Code or Jupyter, select the `.venv313` environment, and run all cells. Keep `sonar_data.csv` in the same folder as the notebook.

The first printed line should read `numpy 2.5.3 | scipy 1.18.1 | scikit-learn 1.9.1 | xgboost 3.2.0`. The notebook prints Tables 1 to 9 and Table A1 of the paper and draws Figures 2 to 8, saving them in `figures/`. A full run takes about 90 minutes: the nested cross-validation takes about 40 minutes and Table A1 about 15.

## Contents

| Path | Description |
|---|---|
| `Sonar_Full_Project.ipynb` | The main notebook, with the outputs of a full run saved inside. |
| `sonar_data.csv` | The UCI Connectionist Bench (Sonar, Mines vs. Rocks) data: 60 features and a `target` column (M or R). |
| `figures/` | The figures of the paper. `image.png` (Figure 1) is an illustration and is not produced by the code. |
| `scripts/` | The same experiments as separate Python scripts, run from the project folder: `experiments.py all`, `nested_cv.py`, `roc_ap.py`, `figures.py`, `workflow_figure.py`, `dependence_check.py`, and `exploratory_nested.py`. |
| `setup_windows.bat` | Creates the Python 3.13 environment and installs the pinned packages on Windows. |
| `requirements.txt` | numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, matplotlib 3.11.2, scikit-learn 1.9.1, xgboost 3.2.0. |

## Notes

Mine is encoded as 0 and Rock as 1. Precision, recall, and F1-score are macro-averaged, and the mine recall and false alarm rate are reported separately with Mine as the class of interest. The MLP uses `max_iter=2000` and converges in every fit. Use Python 3.13 and the pinned versions, which reproduce the values in the paper. With other library versions, some XGBoost and Gradient Boosting results can change slightly.

## Data source

Sejnowski, T. and Gorman, R. P. (1988). Connectionist Bench (Sonar, Mines vs. Rocks). UCI Machine Learning Repository. https://doi.org/10.24432/C5T01Q
