# ⬡ ML Solver Studio

> Production-grade ML web application — 11 algorithms, real-time charts, explanations.

**Built by [Musharib Ramzan](mailto:musharibramzan950@gmail.com)**

---

## Features

- **11 ML Algorithms**: Linear/Polynomial/Logistic Regression, Decision Tree, Random Forest, KNN, SVM, K-Means, Naive Bayes, Neural Network (from scratch), Statistics
- **Real-time Charts**: matplotlib charts rendered as base64 PNGs (scatter plots, confusion matrices, ROC curves, feature importances, loss curves)
- **CSV Upload**: Upload your own dataset
- **Auto-detect**: Suggests algorithms based on input data
- **History**: Persistent analysis history (JSON)
- **Export**: Download results as JSON
- **Beginner/Advanced Mode**: Simplified vs full technical output
- **Dark/Light Mode**
- **Responsive**: Mobile + Desktop

---

## Quick Start

```bash
# 1. Clone / unzip project
cd mlsolver

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy env file
cp .env.example .env

# 5. Run
python app.py
```

Open: **http://localhost:5000**

---

## Docker

```bash
docker build -t mlsolver .
docker run -p 5000:5000 mlsolver
```

---

## Project Structure

```
mlsolver/
├── app.py                    # Flask entry point
├── requirements.txt
├── Dockerfile
├── .env.example
├── backend/
│   ├── models/
│   │   ├── ml_engine.py      # All ML algorithms
│   │   └── history_store.py  # JSON history
│   ├── routes/
│   │   ├── ml_routes.py      # /api/ml/* endpoints
│   │   └── history_routes.py # /api/history/* endpoints
│   └── utils/
│       ├── data_parser.py    # Input parsing
│       ├── plot_utils.py     # Chart generation
│       └── logger.py
├── frontend/
│   ├── templates/index.html
│   └── static/
│       ├── css/main.css
│       └── js/main.js
├── data/                     # history.json stored here
└── logs/                     # app.log
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ml/run` | Run ML algorithm |
| GET | `/api/ml/algorithms` | List all algorithms |
| POST | `/api/ml/detect` | Auto-detect algorithm |
| GET | `/api/history/` | Get history |
| DELETE | `/api/history/clear` | Clear history |
| DELETE | `/api/history/<id>` | Delete entry |

---

## Input Format

**Text input** — rows of comma-separated numbers. Last column = target (y):
```
1,2
2,4
3,6
```

**CSV Upload** — standard CSV, last column = target.

---

## Contact

| | |
|---|---|
| 📧 Email | musharibramzan950@gmail.com |
| 🐙 GitHub | [musharibramzan950-bit](https://github.com/musharibramzan950-bit) |
| 💼 LinkedIn | [musharib-ramzan](https://linkedin.com/in/musharib-ramzan) |
| 🔗 Linktree | [Musharib_](https://linktr.ee/Musharib_) |
| 📸 Instagram | [___musharib____](https://instagram.com/___musharib____) |

---

*MIT License — Musharib Ramzan 2024*
