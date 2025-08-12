import asyncio
import httpx
import random

API_URL = "http://127.0.0.1:8000"
TOTAL_USERS = 1000
MAX_CONCURRENCY = 50  # how many requests at once

semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

async def user_sim(client: httpx.AsyncClient, user_id: int):
    async with semaphore:
        try:
            # Get ranked feed
            resp = await client.get(f"{API_URL}/ranked-feed?limit=3&use_qagent=true")
            resp.raise_for_status()
            items = resp.json()

            if not items:
                return

            # Pick a random item and random feedback
            item = random.choice(items)
            event = random.choice(["click", "skip", "view"])
            view_time = random.uniform(0.5, 5.0) if event == "view" else 0.0

            fb_payload = {
                "content_id": item["id"],
                "event": event,
                "view_time": view_time
            }

            fb_resp = await client.post(f"{API_URL}/engagement-feedback", json=fb_payload)
            fb_resp.raise_for_status()

        except httpx.HTTPError as e:
            print(f"[User {user_id}] HTTP error: {e}")
        except Exception as e:
            print(f"[User {user_id}] Unexpected error: {e}")

async def main():
    async with httpx.AsyncClient(timeout=10) as client:
        tasks = [user_sim(client, i) for i in range(TOTAL_USERS)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
