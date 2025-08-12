# app/ranker.py

import random
import pickle
import os
import time
from collections import defaultdict, deque

PERSIST_FILE = "metrics_store.pkl"

class InMemoryStore:
    def __init__(self):
        self.contents = {}
        self.engagement = defaultdict(lambda: {"views": 0, "clicks": 0, "skips": 0, "reward": 0.0})
        self.history = deque(maxlen=10000)
        self.prev_ranks = {}
        self.metrics_history = []

        self._load_persisted_metrics()

    def _load_persisted_metrics(self):
        if os.path.exists(PERSIST_FILE):
            try:
                with open(PERSIST_FILE, "rb") as f:
                    self.metrics_history = pickle.load(f)
                print(f"[STORE] Loaded metrics history with {len(self.metrics_history)} snapshots.")
            except Exception as e:
                print(f"[STORE] Failed to load persisted metrics: {e}")
                self.metrics_history = []

    def _save_persisted_metrics(self):
        try:
            with open(PERSIST_FILE, "wb") as f:
                pickle.dump(self.metrics_history, f)
        except Exception as e:
            print(f"[STORE] Failed to persist metrics: {e}")

    def add_content(self, content):
        self.contents[content["id"]] = content

    def get_content(self, content_id):
        return self.contents.get(content_id)

    def list_content_ids(self):
        return list(self.contents.keys())

    def apply_feedback(self, content_id, event, view_time=0.0):
        e = self.engagement[content_id]
        if event == "view":
            e["views"] += 1
        elif event == "click":
            e["clicks"] += 1
            e["views"] += 1
        elif event == "skip":
            e["skips"] += 1
            e["views"] += 1

        reward = 0.0
        if event == "click":
            reward = 1.0
        elif event == "view" and view_time > 1.5:
            reward = 0.5
        elif event == "skip":
            reward = -0.2

        e["reward"] += reward
        self.history.append((content_id, event, reward))
        return reward

    def stats(self):
        total_content = len(self.contents)
        avg_reward = sum(v["reward"] for v in self.engagement.values()) / total_content if total_content else 0
        ctr = self._calculate_ctr()
        avg_view_time = self._calculate_avg_view_time()

        top_items = sorted(self.engagement.items(), key=lambda x: x[1]["reward"], reverse=True)[:5]
        top_items_data = []
        for cid, data in top_items:
            top_items_data.append({
                "id": cid,
                "score": data["reward"],
                "ctr": (data["clicks"] / data["views"]) if data["views"] else 0.0,
                "metadata": self.get_content(cid).get("metadata", {}) if self.get_content(cid) else {}
            })

        return {
            "total_content": total_content,
            "avg_reward": avg_reward,
            "ctr": ctr,
            "avg_view_time": avg_view_time,
            "top": [(cid, data) for cid, data in top_items],
            "top_items": top_items_data
        }

    def _calculate_ctr(self):
        total_views = sum(v["views"] for v in self.engagement.values())
        total_clicks = sum(v["clicks"] for v in self.engagement.values())
        return (total_clicks / total_views) if total_views > 0 else 0.0

    def _calculate_avg_view_time(self):
        # Placeholder: extend if you track actual times
        return 2.0  # static for now

    def snapshot_metrics(self):
        s = self.stats()
        snapshot = {
            "timestamp": time.time(),
            "total_items": s["total_content"],
            "avg_engagement": s["avg_reward"],
            "ctr": s["ctr"],
            "avg_view_time": s["avg_view_time"],
            "top_items": s["top_items"]
        }
        self.metrics_history.append(snapshot)
        self._save_persisted_metrics()

    def clear_all(self):
        self.contents.clear()
        self.engagement.clear()
        self.history.clear()
        self.prev_ranks.clear()
        self.metrics_history.clear()
        self._save_persisted_metrics()


class EpsilonGreedyBandit:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.values = defaultdict(float)

    def choose_order(self, ids, k=None):
        k = k or len(ids)
        shuffled = ids[:]
        if random.random() < self.epsilon:
            random.shuffle(shuffled)
        else:
            shuffled.sort(key=lambda x: self.values.get(x, 0.0), reverse=True)
        return shuffled[:k]

    def update(self, content_id, reward):
        self.values[content_id] += reward


class QLearningAgent:
    def __init__(self, alpha=0.1, epsilon=0.1, gamma=0.9):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q = defaultdict(lambda: [0.0, 0.0])

    def _state_key(self, metadata, prev_rank):
        return f"{metadata.get('category', '')}:{prev_rank}"

    def select_action(self, state_key):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        return max(range(2), key=lambda a: self.q[state_key][a])

    def update(self, state_key, action, reward):
        best_next = max(self.q[state_key])
        self.q[state_key][action] += self.alpha * (reward + self.gamma * best_next - self.q[state_key][action])
