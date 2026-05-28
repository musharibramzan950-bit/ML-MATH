"""History management routes."""
from flask import Blueprint, jsonify, request
from backend.models.history_store import get_history, clear_history, delete_entry

history_bp = Blueprint('history', __name__)

@history_bp.route('/', methods=['GET'])
def fetch_history():
    limit = int(request.args.get('limit', 20))
    return jsonify({'history': get_history(limit)})

@history_bp.route('/clear', methods=['DELETE'])
def clear():
    clear_history()
    return jsonify({'success': True})

@history_bp.route('/<entry_id>', methods=['DELETE'])
def delete(entry_id):
    delete_entry(entry_id)
    return jsonify({'success': True})
