"""Manual test: send random session costs for a user to the auth API (session_costs table).

It simulates one chat session: each update adds a random answer (agent tokens and cost,
search embedding cost) and sometimes a random file upload, then PUTs the new running
totals to /sessions/{session_id}/cost, just like chat_gui.py does.

This WRITES to the real database behind the API. The user must exist in app_users.

Run (auth API running, supabase/session_costs.sql applied):
    uv run python tests/random_session_cost.py alice
    uv run python tests/random_session_cost.py alice --updates 10 --seed 42
    uv run python tests/random_session_cost.py alice --session-id <uuid>   # continue a session
"""

import argparse
import os
import random
import uuid

import httpx
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def random_session_updates(rng: random.Random, updates: int) -> list[dict]:
    """Running totals after each simulated step; costs only ever go up, as in a real session."""
    totals = {
        "answers": 0, "tokens": 0, "agent_usd": 0.0, "search_usd": 0.0,
        "upload_files": 0, "upload_tokens": 0, "upload_usd": 0.0,
    }
    steps = []
    for _ in range(updates):
        # one agent answer: about gpt-4o-mini prices
        tokens = rng.randint(500, 6000)
        totals["answers"] += 1
        totals["tokens"] += tokens
        totals["agent_usd"] += tokens * rng.uniform(0.15, 0.6) / 1_000_000
        totals["search_usd"] += rng.randint(0, 3) * rng.uniform(1e-6, 5e-6)  # 0-3 query embeddings
        # sometimes an upload: about text-embedding-3-large prices
        if rng.random() < 0.3:
            upload_tokens = rng.randint(5_000, 300_000)
            totals["upload_files"] += 1
            totals["upload_tokens"] += upload_tokens
            totals["upload_usd"] += upload_tokens * 0.13 / 1_000_000
        steps.append({k: round(v, 8) if isinstance(v, float) else v for k, v in totals.items()})
    return steps


def send_random_costs(
    client: httpx.Client, username: str, updates: int, session_id: str, seed: int | None = None
) -> list[httpx.Response]:
    """PUT each step's running totals to the API; stop at the first error."""
    responses = []
    for i, totals in enumerate(random_session_updates(random.Random(seed), updates), start=1):
        response = client.put(f"/sessions/{session_id}/cost", json={"username": username, **totals})
        responses.append(response)
        total = totals["agent_usd"] + totals["search_usd"] + totals["upload_usd"]
        print(
            f"[{i}/{updates}] {response.status_code}  answers={totals['answers']:<3} "
            f"files={totals['upload_files']:<2} total=${total:.6f}"
        )
        if response.is_error:
            print(f"Stopped: {response.text}")
            break
    return responses


def main() -> None:
    parser = argparse.ArgumentParser(description="Send random session costs to the auth API.")
    parser.add_argument("username", help="existing user in app_users")
    parser.add_argument("--updates", type=int, default=5, help="number of cost updates to send (default 5)")
    parser.add_argument("--session-id", default=None, help="session to update (default: a new random one)")
    parser.add_argument("--seed", type=int, default=None, help="random seed, for repeatable values")
    parser.add_argument(
        "--api-url", default=os.getenv("AUTH_API_URL", "http://127.0.0.1:8000"), help="auth API base URL"
    )
    args = parser.parse_args()

    session_id = args.session_id or str(uuid.uuid4())
    print(f"Session {session_id} for user '{args.username}' at {args.api_url}")
    try:
        with httpx.Client(base_url=args.api_url.rstrip("/"), timeout=10) as client:
            responses = send_random_costs(client, args.username, args.updates, session_id, args.seed)
    except httpx.HTTPError as e:
        raise SystemExit(f"Could not reach the auth API at {args.api_url}: {e}") from None
    if responses and not responses[-1].is_error:
        print(f"Done. Check the row in Supabase: session_costs where session_id = '{session_id}'")


if __name__ == "__main__":
    main()
