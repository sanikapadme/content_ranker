# app/main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
import time, uuid, os, json
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Optional

from .ranker import InMemoryStore, EpsilonGreedyBandit, QLearningAgent

PERSIST_FILE = "metrics_history.json"

app = FastAPI(title="RL Content Ranker")

store = InMemoryStore()
store.metrics_history = []

bandit = EpsilonGreedyBandit(epsilon=0.15)
qagent = QLearningAgent(alpha=0.2, epsilon=0.1)
stable_order = []


class ContentItem(BaseModel):
    id: Optional[str]
    metadata: Dict[str, str]
    created_at: Optional[float]


class Feedback(BaseModel):
    content_id: str
    event: str  # "view", "click", "skip"
    view_time: Optional[float] = 0.0


def base_score(content_id):
    e = store.engagement.get(content_id, {"views": 0, "clicks": 0, "skips": 0, "reward": 0.0})
    clicks = e["clicks"]
    views = e["views"] or 1
    skips = e["skips"]
    reward = e["reward"]
    return (clicks * 2.0 + reward) / views - (skips * 0.5)


def compute_ctr():
    total_views = sum(v.get("views", 0) for v in store.engagement.values())
    total_clicks = sum(v.get("clicks", 0) for v in store.engagement.values())
    return (total_clicks / total_views) if total_views > 0 else 0.0


def compute_avg_view_time():
    times = [v.get("total_view_time", 0) for v in store.engagement.values()]
    total_views = sum(v.get("views", 0) for v in store.engagement.values())
    return (sum(times) / total_views) if total_views > 0 else 0.0


def save_metrics_history():
    with open(PERSIST_FILE, "w") as f:
        json.dump(store.metrics_history, f)


def load_metrics_history():
    if os.path.exists(PERSIST_FILE):
        with open(PERSIST_FILE, "r") as f:
            store.metrics_history = json.load(f)


@app.on_event("startup")
def startup():
    load_metrics_history()
    if not store.contents:
        load_dummy_content()


def add_metrics_snapshot():
    raw = store.stats()
    snapshot = {
        "timestamp": time.time(),
        "total_items": raw.get("total_content"),
        "avg_engagement": raw.get("avg_reward"),
        "ctr": compute_ctr(),
        "avg_view_time": compute_avg_view_time(),
        "top_items": [
            {
                "id": cid,
                "score": data.get("reward", 0.0),
                "ctr": (store.engagement.get(cid, {}).get("clicks", 0) /
                        (store.engagement.get(cid, {}).get("views", 1))),
                "metadata": store.get_content(cid).get("metadata", {})
            }
            for cid, data in raw.get("top", [])
        ]
    }
    store.metrics_history.append(snapshot)
    save_metrics_history()


def load_dummy_content():
    for i in range(1, 4):
        obj = {
            "id": f"test-content-{i}",
            "metadata": {"title": f"Test Content #{i}", "category": "test", "seed": i},
            "created_at": time.time(),
        }
        store.add_content(obj)


@app.post("/submit-content")
async def submit_content(item: ContentItem):
    obj = item.dict()
    if not obj.get("id"):
        obj["id"] = str(uuid.uuid4())
    obj["created_at"] = obj.get("created_at") or time.time()
    store.add_content(obj)
    return {"ok": True, "id": obj["id"]}


@app.get("/ranked-feed")
async def ranked_feed(limit: int = 10, offset: int = 0, use_qagent: bool = False):
    global stable_order
    ids = store.list_content_ids()
    if not ids:
        return []

    if not stable_order:
        stable_order = bandit.choose_order(ids, k=len(ids))

    scored_items = []
    for cid in stable_order:
        content = store.get_content(cid)
        bscore = base_score(cid)
        action = 0
        boost = 0.0

        if use_qagent:
            prev_rank = stable_order.index(cid) + 1
            state_key = qagent._state_key(content.get("metadata", {}), prev_rank)
            action = qagent.select_action(state_key)
            if action == 1:
                boost = 0.5 + qagent.q[state_key][1] * 0.1

        score = bscore + boost
        rounded_score = round(score, 3)

        e = store.engagement.get(cid, {"views": 0, "clicks": 0})
        ctr = (e["clicks"] / e["views"]) if e["views"] > 0 else 0

        scored_items.append({
            "id": cid,
            "score": rounded_score,
            "base_score": bscore,
            "boost_action": action,
            "boost_value": boost,
            "metadata": content.get("metadata", {}),
            "created_at": content.get("created_at", 0),
            "previous_rank": stable_order.index(cid) + 1,
            "current_rank": stable_order.index(cid) + 1,
            "rank_change": 0,
            "ctr": ctr
        })

    scored_items.sort(key=lambda x: (-x["score"], x["created_at"]))
    return scored_items[offset:offset + limit]


@app.post("/engagement-feedback")
async def engagement_feedback(feedback: Feedback, background_tasks: BackgroundTasks):
    content = store.get_content(feedback.content_id)
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")

    reward = store.apply_feedback(feedback.content_id, feedback.event, feedback.view_time)
    bandit.update(feedback.content_id, reward)

    prev_rank = stable_order.index(feedback.content_id) + 1 if feedback.content_id in stable_order else 0
    state_key = qagent._state_key(content.get("metadata", {}), prev_rank)
    action = 1 if reward > 0.5 else 0
    qagent.update(state_key, action, reward)

    # Immediately update metrics history for chart refresh
    add_metrics_snapshot()

    return {"ok": True, "reward": reward}


@app.post("/train-agent")
async def train_agent(iterations: int = 100):
    global stable_order
    for _ in range(iterations):
        for content_id, event, reward in list(store.history):
            content = store.get_content(content_id)
            prev_rank = stable_order.index(content_id) + 1 if content_id in stable_order else 0
            state_key = qagent._state_key(content.get("metadata", {}), prev_rank)
            action = 1 if reward > 0.5 else 0
            qagent.update(state_key, action, reward)
    stable_order = []
    return {"ok": True, "trained_iterations": iterations}


@app.post("/reset")
async def reset_system(mode: str = "order"):
    global stable_order
    store.contents.clear()
    store.history.clear()
    store.engagement.clear()
    store.prev_ranks.clear()
    if mode == "full":
        store.metrics_history.clear()
        save_metrics_history()
    stable_order = []
    load_dummy_content()
    return {"ok": True, "message": "System reset", "mode": mode}


@app.get("/metrics")
async def metrics():
    add_metrics_snapshot()
    return {
        "total_items": store.metrics_history[-1]["total_items"],
        "avg_engagement": store.metrics_history[-1]["avg_engagement"],
        "ctr": store.metrics_history[-1]["ctr"],
        "avg_view_time": store.metrics_history[-1]["avg_view_time"],
        "top_items": store.metrics_history[-1]["top_items"],
        "history": store.metrics_history
    }


@app.get("/")
def serve_frontend():
    frontend_path = r"D:\projects\content_ranker\app\frontend.html"
    if not os.path.isfile(frontend_path):
        raise HTTPException(status_code=404, detail="frontend.html not found")
    return FileResponse(frontend_path)
