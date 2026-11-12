import os
from functools import lru_cache
from pathlib import Path

import bcrypt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import Client, create_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]
USERS_TABLE = "app_users"

load_dotenv(PROJECT_ROOT / ".env")

# Checked when a username does not exist, so both cases take the same time.
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt())

app = FastAPI(title="LLM Q&A auth API")


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    username: str


@lru_cache
def get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env.")
    return create_client(url, key)


def hash_password(password: str) -> str:
    print(bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode())
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash)
    except ValueError:
        # bcrypt rejects passwords longer than 72 bytes.
        return False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/login", response_model=LoginResponse)
def login(body: LoginRequest) -> LoginResponse:
    result = (
        get_supabase()
        .table(USERS_TABLE)
        .select("username, password_hash")
        .eq("username", body.username)
        .limit(1)
        .execute()
    )
    user = result.data[0] if result.data else None
    print(result)
    stored_hash = user["password_hash"].encode() if user else _DUMMY_HASH
    if not verify_password(body.password, stored_hash) or user is None:
        raise HTTPException(status_code=401, detail="Incorrect username or password.")

    return LoginResponse(username=user["username"])
