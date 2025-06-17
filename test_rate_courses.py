#!/usr/bin/env python3
import os, requests

# ◼︎ Replace with your actual custom domain or API URL:
RATE_URL = os.getenv("RATE_URL", "https://api.nu-distrorecs.com/rate")
RECS_URL = os.getenv("RECS_URL", "https://api.nu-distrorecs.com/recommend")

def test_rate_course():
    payload = {"course_id": "COMP_SCI_101-0", "rating": 5}
    r = requests.post(RATE_URL, json=payload)
    print("POST /rate status:", r.status_code, r.json())

    # now fetch recommendations to see the new avg_difficulty
    q = {"course": "COMP_SCI_101-0", "k": 1}
    r2 = requests.get(RECS_URL, params=q)
    print("GET /recommend status:", r2.status_code, r2.json())

if __name__ == "__main__":
    test_rate_course()
