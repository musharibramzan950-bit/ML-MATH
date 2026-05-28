"""ML processing API routes."""
import logging, json
from flask import Blueprint, request, jsonify
from backend.utils.data_parser import parse_input, parse_csv_file
from backend.models.ml_engine import run_algorithm, ALGORITHM_MAP
from backend.models.history_store import add_entry

logger = logging.getLogger(__name__)
ml_bp = Blueprint('ml', __name__)

NEEDS_Y = {'linear_regression', 'polynomial_regression', 'logistic_regression',
           'decision_tree', 'random_forest', 'knn', 'svm', 'naive_bayes', 'neural_network'}


@ml_bp.route('/run', methods=['POST'])
def run_ml():
    try:
        algorithm = request.form.get('algorithm') or (request.json or {}).get('algorithm')
        raw_input = request.form.get('input_data') or (request.json or {}).get('input_data', '')
        params = {}

        # Extra params
        try:
            degree = int(request.form.get('degree') or (request.json or {}).get('degree', 2))
            params['degree'] = degree
        except Exception:
            pass
        try:
            k = int(request.form.get('k') or (request.json or {}).get('k', 3))
            params['k'] = k
        except Exception:
            pass
        task = request.form.get('task') or (request.json or {}).get('task', 'classification')
        params['task'] = task

        if not algorithm or algorithm not in ALGORITHM_MAP:
            return jsonify({'error': f'Invalid algorithm. Choose from: {list(ALGORITHM_MAP.keys())}'}), 400

        # Handle CSV file upload
        parsed = None
        if 'file' in request.files:
            f = request.files['file']
            parsed = parse_csv_file(f)
        elif raw_input:
            parsed = parse_input(raw_input)
        else:
            return jsonify({'error': 'No input data provided'}), 400

        X = parsed['X']
        y = parsed.get('y')

        if len(X) == 0:
            return jsonify({'error': 'Could not parse any numeric data from input'}), 400

        if algorithm in NEEDS_Y and y is None:
            if len(X) < 2:
                return jsonify({'error': 'Need at least 2 values for this algorithm'}), 400
            # Auto split X: first half features, second as target
            mid = len(X) // 2
            y = X[mid:]
            X = X[:mid]

        result = run_algorithm(algorithm, X=X, y=y, **params)
        input_summary = raw_input[:100] if raw_input else 'CSV file'
        add_entry(result.get('algorithm', algorithm), input_summary, result.get('metrics', {}))
        return jsonify({'success': True, 'result': result})

    except Exception as e:
        logger.exception("Error running ML algorithm")
        return jsonify({'error': str(e)}), 500


@ml_bp.route('/algorithms', methods=['GET'])
def list_algorithms():
    info = {
        'linear_regression': {'label': 'Linear Regression', 'category': 'Regression', 'needs_y': True, 'description': 'Predicts continuous output with a straight line'},
        'polynomial_regression': {'label': 'Polynomial Regression', 'category': 'Regression', 'needs_y': True, 'description': 'Fits curved relationships using polynomial features'},
        'logistic_regression': {'label': 'Logistic Regression', 'category': 'Classification', 'needs_y': True, 'description': 'Binary/multi-class classification using sigmoid'},
        'decision_tree': {'label': 'Decision Tree', 'category': 'Classification', 'needs_y': True, 'description': 'Tree-based rules for classification'},
        'random_forest': {'label': 'Random Forest', 'category': 'Ensemble', 'needs_y': True, 'description': 'Ensemble of decision trees, reduces overfitting'},
        'knn': {'label': 'K-Nearest Neighbors', 'category': 'Classification', 'needs_y': True, 'description': 'Classifies by nearest training examples'},
        'svm': {'label': 'Support Vector Machine', 'category': 'Classification', 'needs_y': True, 'description': 'Finds optimal separating hyperplane'},
        'kmeans': {'label': 'K-Means Clustering', 'category': 'Clustering', 'needs_y': False, 'description': 'Groups data into k clusters unsupervised'},
        'naive_bayes': {'label': 'Naive Bayes', 'category': 'Classification', 'needs_y': True, 'description': 'Probabilistic classifier using Bayes theorem'},
        'statistics': {'label': 'Statistics Analysis', 'category': 'Analysis', 'needs_y': False, 'description': 'Full statistical summary with distribution plot'},
        'neural_network': {'label': 'Neural Network', 'category': 'Deep Learning', 'needs_y': True, 'description': '2-layer feedforward neural network from scratch'},
    }
    return jsonify({'algorithms': info})


@ml_bp.route('/detect', methods=['POST'])
def auto_detect():
    """Suggest best algorithm based on input."""
    data = request.json or {}
    raw = data.get('input_data', '')
    lines = [l for l in raw.strip().split('\n') if l.strip()]
    n = len(lines)
    has_commas = any(',' in l for l in lines)

    suggestions = []
    if n < 10 and not has_commas:
        suggestions = ['statistics', 'linear_regression', 'polynomial_regression']
    elif has_commas:
        suggestions = ['linear_regression', 'logistic_regression', 'random_forest', 'svm']
    else:
        suggestions = ['statistics', 'kmeans', 'neural_network']

    return jsonify({'suggestions': suggestions})
