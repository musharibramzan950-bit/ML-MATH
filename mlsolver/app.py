"""
ML Solver - Main Flask Application
Author: Musharib Ramzan | musharibramzan950@gmail.com
"""
import os
import logging
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from backend.routes.ml_routes import ml_bp
from backend.routes.history_routes import history_bp
from backend.utils.logger import setup_logger

load_dotenv()

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')
app.secret_key = os.getenv('SECRET_KEY', 'mlsolver-secret-key-change-in-prod')
CORS(app)

setup_logger()
logger = logging.getLogger(__name__)

app.register_blueprint(ml_bp, url_prefix='/api/ml')
app.register_blueprint(history_bp, url_prefix='/api/history')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'version': '1.0.0'})

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
