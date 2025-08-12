import asyncio
import httpx
import random

API = "http://127.0.0.1:8000"
NUM_USERS = 50
ACTIONS = ["view", "click", "skip"]

async def submit_initial(client, n=20):
    for i in range(n):
        payload = {"metadata": {"title": f"content-{i}", "body": "lorem ipsum", "seed": i}}
        await client.post(f"{API}/submit-content", json=payload)

async def user_sim(client, user_id):
    for _ in range(20):
        r = await client.get(f"{API}/ranked-feed?limit=5")
        feed = r.json()  # no await here
        for pos, item in enumerate(feed):
            cid = item["id"]
            rand_val = random.random()
            if rand_val < 0.3:
                await client.post(f"{API}/engagement-feedback", json={
                    "content_id": cid,
                    "event": "click",
                    "view_time": random.uniform(1, 6)
                })
            elif rand_val < 0.8:
                await client.post(f"{API}/engagement-feedback", json={
                    "content_id": cid,
                    "event": "view",
                    "view_time": random.uniform(0.5, 4.0)
                })
            else:
                await client.post(f"{API}/engagement-feedback", json={
                    "content_id": cid,
                    "event": "skip"
                })
        await asyncio.sleep(random.uniform(0.05, 0.2))

async def main():
    async with httpx.AsyncClient(timeout=60.0) as client:
        await submit_initial(client)
        tasks = [asyncio.create_task(user_sim(client, uid)) for uid in range(NUM_USERS)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
