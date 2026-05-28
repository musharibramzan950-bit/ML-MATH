"""
Core ML engine handling all algorithm types.
"""
import numpy as np
import logging
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                              confusion_matrix, classification_report,
                              roc_curve, auc, mean_absolute_error)
from sklearn.model_selection import train_test_split
from scipy import stats as scipy_stats
from backend.utils import plot_utils

logger = logging.getLogger(__name__)


def run_linear_regression(X, y, **kwargs):
    X = np.array(X).reshape(-1, 1) if np.array(X).ndim == 1 else np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X)
    y_test_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    coef = model.coef_.tolist()
    intercept = float(model.intercept_)
    equation = f"y = {coef[0]:.4f}x + {intercept:.4f}" if len(coef) == 1 else f"y = {intercept:.4f} + " + " + ".join([f"{c:.4f}*x{i+1}" for i, c in enumerate(coef)])
    chart = plot_utils.scatter_with_line(X, y, y_pred, 'Linear Regression Fit')
    return {
        'algorithm': 'Linear Regression',
        'metrics': {'R² Score': round(r2, 4), 'MSE': round(mse, 4), 'RMSE': round(float(mse**0.5), 4), 'MAE': round(mean_absolute_error(y_test, y_test_pred), 4)},
        'equation': equation,
        'coefficients': coef,
        'intercept': intercept,
        'predictions': y_pred.tolist()[:20],
        'charts': [{'title': 'Regression Fit', 'image': chart}],
        'explanation': f"Linear Regression found the best-fit line: {equation}. R² of {r2:.4f} means the model explains {r2*100:.1f}% of variance in the data."
    }


def run_polynomial_regression(X, y, degree=2, **kwargs):
    X = np.array(X).reshape(-1, 1) if np.array(X).ndim == 1 else np.array(X)
    y = np.array(y)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_poly)
    y_test_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    chart = plot_utils.poly_fit_plot(X, y, y_pred, degree)
    return {
        'algorithm': f'Polynomial Regression (Degree {degree})',
        'metrics': {'R² Score': round(r2, 4), 'MSE': round(mse, 4), 'RMSE': round(float(mse**0.5), 4)},
        'degree': degree,
        'predictions': y_pred.tolist()[:20],
        'charts': [{'title': f'Polynomial Fit (d={degree})', 'image': chart}],
        'explanation': f"Degree-{degree} polynomial fit. R²={r2:.4f}. Higher degree captures more complex patterns but risks overfitting."
    }


def run_logistic_regression(X, y, **kwargs):
    X = np.array(X).reshape(-1, 1) if np.array(X).ndim == 1 else np.array(X)
    y = np.array(y).astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    classes = [str(c) for c in model.classes_]
    cm_chart = plot_utils.confusion_matrix_plot(cm, classes)
    charts = [{'title': 'Confusion Matrix', 'image': cm_chart}]
    try:
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_score = auc(fpr, tpr)
        roc_chart = plot_utils.roc_curve_plot(fpr, tpr, auc_score)
        charts.append({'title': 'ROC Curve', 'image': roc_chart})
    except Exception:
        auc_score = None
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        'algorithm': 'Logistic Regression',
        'metrics': {'Accuracy': round(acc, 4), 'AUC': round(auc_score, 4) if auc_score else 'N/A'},
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'charts': charts,
        'explanation': f"Logistic Regression achieved {acc*100:.1f}% accuracy on test data."
    }


def run_decision_tree(X, y, **kwargs):
    X = np.array(X).reshape(-1, 1) if np.array(X).ndim == 1 else np.array(X)
    y = np.array(y).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    classes = [str(c) for c in model.classes_]
    cm_chart = plot_utils.confusion_matrix_plot(cm, classes)
    fi = model.feature_importances_
    fi_chart = plot_utils.bar_chart([f'Feature {i+1}' for i in range(len(fi))], fi, 'Feature Importances')
    return {
        'algorithm': 'Decision Tree',
        'metrics': {'Accuracy': round(acc, 4), 'Max Depth': 5},
        'confusion_matrix': cm.tolist(),
        'feature_importances': fi.tolist(),
        'charts': [{'title': 'Confusion Matrix', 'image': cm_chart}, {'title': 'Feature Importances', 'image': fi_chart}],
        'explanation': f"Decision Tree (max_depth=5) achieved {acc*100:.1f}% accuracy. Feature importances show which variables drive predictions most."
    }


def run_random_forest(X, y, task='classification', **kwargs):
    X = np.array(X).reshape(-1, 1) if np.array(X).ndim == 1 else np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if task == 'regression':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        fi = model.feature_importances_
        fi_chart = plot_utils.bar_chart([f'Feature {i+1}' for i in range(len(fi))], fi, 'Feature Importances')
        return {'algorithm': 'Random Forest Regressor', 'metrics': {'R²': round(r2, 4), 'MSE': round(mse, 4)},
                'feature_importances': fi.tolist(), 'charts': [{'title': 'Feature Importances', 'image': fi_chart}],
                'explanation': f"Random Forest (100 trees) R²={r2:.4f}. Ensemble of decision trees reduces overfitting."}
    else:
        y = y.astype(int)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fi = model.feature_importances_
        cm_chart = plot_utils.confusion_matrix_plot(cm, [str(c) for c in model.classes_])
        fi_chart = plot_utils.bar_chart([f'Feature {i+1}' for i in range(len(fi))], fi, 'Feature Importances')
        return {'algorithm': 'Random Forest Classifier', 'metrics': {'Accuracy': round(acc, 4)},
                'confusion_matrix': cm.tolist(), 'feature_importances': fi.tolist(),
                'charts': [{'title': 'Confusion Matrix', 'image': cm_chart}, {'title': 'Feature Importances', 'image': fi_chart}],
                'explanation': f"Random Forest (100 trees) accuracy={acc*100:.1f}%. Robust ensemble method."}


def run_knn(X, y, k=5, task='classification', **kwargs):
    X = np.array(X).reshape(-1, 1) if np.array(X).ndim == 1 else np.array(X)
    y = np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    if task == 'regression':
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return {'algorithm': f'KNN Regressor (k={k})', 'metrics': {'R²': round(r2, 4)},
                'charts': [], 'explanation': f"KNN (k={k}) R²={r2:.4f}. Predicts based on {k} nearest neighbors."}
    else:
        y = y.astype(int)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cm_chart = plot_utils.confusion_matrix_plot(cm, [str(c) for c in np.unique(y)])
        return {'algorithm': f'KNN Classifier (k={k})', 'metrics': {'Accuracy': round(acc, 4)},
                'confusion_matrix': cm.tolist(), 'charts': [{'title': 'Confusion Matrix', 'image': cm_chart}],
                'explanation': f"KNN (k={k}) accuracy={acc*100:.1f}%. Classifies by majority vote of nearest neighbors."}


def run_svm(X, y, task='classification', **kwargs):
    X = np.array(X).reshape(-1, 1) if np.array(X).ndim == 1 else np.array(X)
    y = np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    if task == 'regression':
        model = SVR(kernel='rbf')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return {'algorithm': 'SVR (RBF Kernel)', 'metrics': {'R²': round(r2, 4)},
                'charts': [], 'explanation': f"Support Vector Regression R²={r2:.4f}. Uses RBF kernel for non-linear mapping."}
    else:
        y = y.astype(int)
        model = SVC(probability=True, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cm_chart = plot_utils.confusion_matrix_plot(cm, [str(c) for c in model.classes_])
        return {'algorithm': 'SVM Classifier (RBF)', 'metrics': {'Accuracy': round(acc, 4)},
                'confusion_matrix': cm.tolist(), 'charts': [{'title': 'Confusion Matrix', 'image': cm_chart}],
                'explanation': f"SVM accuracy={acc*100:.1f}%. Finds optimal hyperplane separating classes."}


def run_kmeans(X, k=3, **kwargs):
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    centers = scaler.inverse_transform(model.cluster_centers_)
    inertia = model.inertia_
    chart = plot_utils.cluster_plot(X_scaled if X_scaled.shape[1] >= 2 else np.hstack([X_scaled, np.zeros((len(X_scaled),1))]),
                                    labels, model.cluster_centers_ if X_scaled.shape[1] >= 2 else None)
    return {
        'algorithm': f'K-Means Clustering (k={k})',
        'metrics': {'Inertia': round(inertia, 4), 'Clusters': k},
        'cluster_labels': labels.tolist(),
        'cluster_centers': centers.tolist(),
        'charts': [{'title': 'Cluster Visualization', 'image': chart}],
        'explanation': f"K-Means found {k} clusters. Inertia={inertia:.2f} (lower=tighter clusters). Each point is assigned to its nearest centroid."
    }


def run_naive_bayes(X, y, **kwargs):
    X = np.array(X).reshape(-1, 1) if np.array(X).ndim == 1 else np.array(X)
    y = np.array(y).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cm_chart = plot_utils.confusion_matrix_plot(cm, [str(c) for c in model.classes_])
    return {
        'algorithm': 'Gaussian Naive Bayes',
        'metrics': {'Accuracy': round(acc, 4)},
        'confusion_matrix': cm.tolist(),
        'charts': [{'title': 'Confusion Matrix', 'image': cm_chart}],
        'explanation': f"Naive Bayes accuracy={acc*100:.1f}%. Applies Bayes theorem assuming feature independence."
    }


def run_statistics(X, **kwargs):
    X = np.array(X).flatten()
    hist_chart = plot_utils.histogram_plot(X, 'Data Distribution')
    shapiro_stat, shapiro_p = scipy_stats.shapiro(X[:5000]) if len(X) <= 5000 else (None, None)
    skew = float(scipy_stats.skew(X))
    kurt = float(scipy_stats.kurtosis(X))
    return {
        'algorithm': 'Statistics Analysis',
        'metrics': {
            'Mean': round(float(np.mean(X)), 4),
            'Median': round(float(np.median(X)), 4),
            'Std Dev': round(float(np.std(X)), 4),
            'Variance': round(float(np.var(X)), 4),
            'Min': round(float(np.min(X)), 4),
            'Max': round(float(np.max(X)), 4),
            'Range': round(float(np.max(X) - np.min(X)), 4),
            'Skewness': round(skew, 4),
            'Kurtosis': round(kurt, 4),
            'Q1 (25%)': round(float(np.percentile(X, 25)), 4),
            'Q3 (75%)': round(float(np.percentile(X, 75)), 4),
            'IQR': round(float(np.percentile(X, 75) - np.percentile(X, 25)), 4),
            'Shapiro-Wilk p': round(float(shapiro_p), 4) if shapiro_p else 'N/A (large sample)'
        },
        'charts': [{'title': 'Distribution Histogram', 'image': hist_chart}],
        'explanation': f"Statistical summary of {len(X)} data points. Skewness={skew:.2f} ({'right-skewed' if skew>0 else 'left-skewed' if skew<0 else 'symmetric'}). Kurtosis={kurt:.2f}."
    }


def run_neural_network(X, y, task='classification', **kwargs):
    """Simple 2-layer neural network using numpy."""
    X = np.array(X)
    y = np.array(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

    n_in = X_train.shape[1]
    n_hid = 16
    lr = 0.01
    epochs = 300

    np.random.seed(42)
    W1 = np.random.randn(n_in, n_hid) * 0.1
    b1 = np.zeros((1, n_hid))
    W2 = np.random.randn(n_hid, 1) * 0.1
    b2 = np.zeros((1, 1))

    def relu(z): return np.maximum(0, z)
    def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    losses = []
    for ep in range(epochs):
        z1 = X_train @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        out = sigmoid(z2) if task == 'classification' else z2
        y_col = y_train.reshape(-1, 1)
        loss = float(np.mean((out - y_col)**2))
        losses.append(loss)
        d_out = 2 * (out - y_col) / len(y_train)
        if task == 'classification':
            d_out *= out * (1 - out)
        dW2 = a1.T @ d_out
        db2 = np.sum(d_out, axis=0, keepdims=True)
        d_a1 = d_out @ W2.T
        d_z1 = d_a1 * (z1 > 0)
        dW1 = X_train.T @ d_z1
        db1 = np.sum(d_z1, axis=0, keepdims=True)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

    z1t = X_test @ W1 + b1
    a1t = relu(z1t)
    z2t = a1t @ W2 + b2
    out_test = sigmoid(z2t) if task == 'classification' else z2t
    y_pred_test = (out_test.flatten() > 0.5).astype(int) if task == 'classification' else out_test.flatten()

    loss_chart = plot_utils.loss_curve(losses, 'Neural Network Training Loss')

    if task == 'classification':
        acc = accuracy_score(y_test.astype(int), y_pred_test)
        return {'algorithm': 'Neural Network (2-layer)', 'metrics': {'Accuracy': round(acc, 4), 'Final Loss': round(losses[-1], 6)},
                'charts': [{'title': 'Training Loss', 'image': loss_chart}],
                'explanation': f"2-layer NN ({n_in}→{n_hid}→1) trained for {epochs} epochs. Accuracy={acc*100:.1f}%. Loss converged to {losses[-1]:.4f}."}
    else:
        r2 = r2_score(y_test, y_pred_test)
        return {'algorithm': 'Neural Network (2-layer)', 'metrics': {'R²': round(r2, 4), 'Final Loss': round(losses[-1], 6)},
                'charts': [{'title': 'Training Loss', 'image': loss_chart}],
                'explanation': f"2-layer NN trained {epochs} epochs. R²={r2:.4f}. Loss: {losses[0]:.4f}→{losses[-1]:.4f}."}


ALGORITHM_MAP = {
    'linear_regression': run_linear_regression,
    'polynomial_regression': run_polynomial_regression,
    'logistic_regression': run_logistic_regression,
    'decision_tree': run_decision_tree,
    'random_forest': run_random_forest,
    'knn': run_knn,
    'svm': run_svm,
    'kmeans': run_kmeans,
    'naive_bayes': run_naive_bayes,
    'statistics': run_statistics,
    'neural_network': run_neural_network,
}


def run_algorithm(algorithm: str, X, y=None, **params):
    func = ALGORITHM_MAP.get(algorithm)
    if not func:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if y is not None:
        return func(X=X, y=y, **params)
    return func(X=X, **params)
