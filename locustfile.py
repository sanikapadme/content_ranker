from locust import HttpUser, task, between
import random

class APISimUser(HttpUser):
    wait_time = between(0.01, 0.2)

    @task(3)
    def get_feed(self):
        self.client.get("/ranked-feed?limit=5")

    @task(1)
    def send_feedback(self):
        cid = "dummy-id"
        ev = random.choice(["view","click","skip"])
        self.client.post("/engagement-feedback", json={"content_id": cid, "event": ev, "view_time": random.random()*5})
