"""Simple JSON-based history storage."""
import json, os, time, uuid

HISTORY_FILE = 'data/history.json'

def _load():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, 'r') as f:
        try:
            return json.load(f)
        except Exception:
            return []

def _save(data):
    os.makedirs('data', exist_ok=True)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def add_entry(algorithm, input_summary, metrics):
    history = _load()
    entry = {
        'id': str(uuid.uuid4())[:8],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'algorithm': algorithm,
        'input': input_summary,
        'metrics': metrics
    }
    history.insert(0, entry)
    history = history[:100]  # keep last 100
    _save(history)
    return entry

def get_history(limit=20):
    return _load()[:limit]

def clear_history():
    _save([])

def delete_entry(entry_id):
    history = _load()
    history = [h for h in history if h['id'] != entry_id]
    _save(history)
