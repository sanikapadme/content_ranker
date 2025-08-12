
# RL Content Ranker Dashboard

A reinforcement learning–based content ranking system with an interactive dashboard, CTR tracking, and persistent metrics storage.

---

## Project Structure
content_ranker/
│
├── app/
│ ├── frontend.html # Web dashboard UI
│ ├── main.py # FastAPI backend + API routes
│ ├── ranker.py # RL logic (Bandit, Q-Learning, InMemoryStore)
│ ├── schemas.py # Pydantic request/response schemas
│ ├── stress_test.py # Script for high-load feedback simulation
│
├── simulation/
│ ├── simulate_users.py # Virtual user feedback simulator
│
├── venv/ # Python virtual environment
├── metrics_history.json # Persistent metrics storage
├── locustfile.py # Locust load testing file
├── requirements.txt # Python dependencies
├── Dockerfile # Docker build configuration
├── .gitignore # Git ignore list
├── README.md # Project documentation



---

## Features
- Epsilon-Greedy Bandit + Q-Learning Agent ranking logic.
- Real-time Dashboard with Chart.js graphs.
- CTR Tracking (click-through rate) and average engagement.
- Persistent Metrics (saved to `metrics_history.json`).
- User Feedback (`👍` = click, `👎` = skip).
- Reset & Train Controls for experimentation.
- Stress Testing with `stress_test.py` and `simulate_users.py`.

---

## Installation

### 1. Clone repository

git clone <your-repo-url>
cd content_ranker
2. Set up Python environment

python -m venv venv
Activate environment:

Windows:


venv\Scripts\activate
Mac/Linux:
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt
Run Backend
uvicorn app.main:app --reload

Open Dashboard
http://127.0.0.1:8000

Usage
View ranked content in a table.

Click 👍 to simulate a click (positive reward).

Click 👎 to simulate a skip (negative reward).

Charts:
Average Engagement over time

CTR over time

Buttons:
Reset Order → Clears order but keeps history.

Full Reset → Wipes all data and reloads dummy content.

Train Now → Trains Q-Learning agent from stored history.

Persistence
Engagement history and metrics are stored in metrics_history.json.

Data persists across server restarts.

Stress Testing
Run simulated feedback to see ranking adapt:


python app/stress_test.py
# OR
python simulation/simulate_users.py


