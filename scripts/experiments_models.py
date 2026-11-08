from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

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

