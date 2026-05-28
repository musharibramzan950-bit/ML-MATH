"""Generate matplotlib charts as base64 PNG strings."""
import io, base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DARK_BG = '#0d1117'
ACCENT = '#7c3aed'
ACCENT2 = '#06b6d4'
TEXT = '#e2e8f0'

def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _style(fig, ax, title=''):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor('#161b22')
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    if title:
        ax.set_title(title, fontsize=13, pad=12)

def scatter_with_line(X, y, y_pred, title='Regression Fit'):
    fig, ax = plt.subplots(figsize=(7, 4))
    _style(fig, ax, title)
    x_plot = X.flatten() if X.ndim > 1 else X
    ax.scatter(x_plot, y, color=ACCENT2, alpha=0.7, s=30, label='Data')
    sort_idx = np.argsort(x_plot)
    ax.plot(x_plot[sort_idx], y_pred[sort_idx], color=ACCENT, lw=2, label='Fit')
    ax.legend(facecolor='#21262d', labelcolor=TEXT)
    return _b64(fig)

def confusion_matrix_plot(cm, classes):
    fig, ax = plt.subplots(figsize=(5, 4))
    _style(fig, ax, 'Confusion Matrix')
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white', fontsize=11)
    fig.colorbar(im, ax=ax)
    return _b64(fig)

def cluster_plot(X, labels, centers=None, title='K-Means Clustering'):
    fig, ax = plt.subplots(figsize=(7, 4))
    _style(fig, ax, title)
    colors = plt.cm.tab10(np.linspace(0, 1, max(labels)+1))
    x0 = X[:, 0]; x1 = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))
    for k in range(max(labels)+1):
        mask = labels == k
        ax.scatter(x0[mask], x1[mask], color=colors[k], alpha=0.7, s=30, label=f'Cluster {k}')
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], marker='X', s=120, color='white', zorder=5, label='Centers')
    ax.legend(facecolor='#21262d', labelcolor=TEXT)
    return _b64(fig)

def histogram_plot(data, title='Distribution'):
    fig, ax = plt.subplots(figsize=(7, 4))
    _style(fig, ax, title)
    ax.hist(data, bins=20, color=ACCENT, edgecolor='#21262d', alpha=0.85)
    ax.set_xlabel('Value'); ax.set_ylabel('Frequency')
    return _b64(fig)

def bar_chart(labels, values, title='Feature Importances'):
    fig, ax = plt.subplots(figsize=(7, 4))
    _style(fig, ax, title)
    colors = [ACCENT, ACCENT2, '#f59e0b', '#10b981', '#ef4444', '#ec4899']
    bars = ax.barh(labels, values, color=[colors[i % len(colors)] for i in range(len(labels))])
    ax.set_xlabel('Importance')
    return _b64(fig)

def loss_curve(train_losses, title='Training Loss'):
    fig, ax = plt.subplots(figsize=(7, 4))
    _style(fig, ax, title)
    ax.plot(train_losses, color=ACCENT, lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    return _b64(fig)

def poly_fit_plot(X, y, y_pred, degree, title='Polynomial Regression'):
    return scatter_with_line(X, y, y_pred, f'Polynomial Regression (Degree {degree})')

def roc_curve_plot(fpr, tpr, auc_score):
    fig, ax = plt.subplots(figsize=(5, 4))
    _style(fig, ax, f'ROC Curve (AUC={auc_score:.3f})')
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f'AUC={auc_score:.3f}')
    ax.plot([0,1],[0,1],'--', color='#6b7280')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(facecolor='#21262d', labelcolor=TEXT)
    return _b64(fig)
