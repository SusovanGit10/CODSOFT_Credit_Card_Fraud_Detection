from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, y_train):
    models = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=10)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    return models