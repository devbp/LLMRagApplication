import base64
import json
import logging
import os
import secrets
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import bcrypt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from postgrest.exceptions import APIError
from pydantic import BaseModel, Field, field_validator
from supabase import Client, create_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]
USERS_TABLE = "app_users"
SESSION_COSTS_TABLE = "session_costs"  # created by supabase/session_costs.sql

# POST /auth/users needs the X-Admin-Key header (matching ADMIN_API_KEY in .env) when True.
# TEMPORARILY False: anyone who can reach the API can create users. Set back to True
# before the API is reachable from other machines.
REQUIRE_ADMIN_KEY = False

logger = logging.getLogger(__name__)

load_dotenv(PROJECT_ROOT / ".env")

# Checked when a username does not exist, so both cases take the same time.
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt())

app = FastAPI(title="LLM Q&A auth API")


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    username: str


class CreateUserRequest(BaseModel):
    # Letters, digits, dot, dash and underscore only, so names stay unambiguous.
    username: str = Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8)

    @field_validator("password")
    @classmethod
    def password_fits_bcrypt(cls, value: str) -> str:
        if len(value.encode()) > 72:  # bcrypt only uses the first 72 bytes
            raise ValueError("Password must be at most 72 bytes long.")
        return value


class CreateUserResponse(BaseModel):
    username: str


class SessionCostRequest(BaseModel):
    """Running totals of one chat session (the whole session so far, not just the last answer)."""

    username: str = Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")
    answers: int = Field(ge=0)
    tokens: int = Field(ge=0)
    agent_usd: float = Field(ge=0)
    search_usd: float = Field(ge=0)
    upload_files: int = Field(ge=0)
    upload_tokens: int = Field(ge=0)
    upload_usd: float = Field(ge=0)
    # Test case generation and evaluation (test_data_gui.py); 0 for older clients.
    testgen_runs: int = Field(default=0, ge=0)
    testgen_tokens: int = Field(default=0, ge=0)
    testgen_usd: float = Field(default=0.0, ge=0)
    eval_runs: int = Field(default=0, ge=0)
    eval_tokens: int = Field(default=0, ge=0)
    eval_usd: float = Field(default=0.0, ge=0)


class SessionCostResponse(BaseModel):
    session_id: uuid.UUID
    username: str
    total_usd: float


@lru_cache
def get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env.")
    if _is_public_key(key):
        # A public key is blocked by row-level security on app_users: reads look empty
        # (every login fails) and inserts fail with "violates row-level security policy".
        raise RuntimeError(
            "SUPABASE_SERVICE_ROLE_KEY holds a public key (publishable/anon). Use the secret key "
            "(sb_secret_...) or the legacy service_role key from Supabase > Project Settings > API Keys."
        )
    return create_client(url, key)


def _is_public_key(key: str) -> bool:
    """True for Supabase's public keys: new 'sb_publishable_...' or a legacy JWT with role 'anon'."""
    if key.startswith("sb_publishable_"):
        return True
    parts = key.split(".")
    if len(parts) == 3:  # legacy JWT key: check its role claim
        try:
            payload = parts[1] + "=" * (-len(parts[1]) % 4)
            return json.loads(base64.urlsafe_b64decode(payload)).get("role") == "anon"
        except ValueError:
            return False
    return False


def hash_password(password: str) -> str:
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
    try:
        result = (
            get_supabase()
            .table(USERS_TABLE)
            .select("username, password_hash")
            .eq("username", body.username)
            .limit(1)
            .execute()
        )
    except Exception:
        # Missing config, network or database error. Log the details on the server,
        # but don't send them (they may include the Supabase URL) to the client.
        logger.exception("Login lookup failed")
        raise HTTPException(status_code=503, detail="Login is temporarily unavailable.") from None

    user = result.data[0] if result.data else None
    stored = user.get("password_hash") if user else None

    # A missing user, or a row without a usable hash, can never log in. The dummy
    # check still runs so these cases take as long as a real one (no username probing).
    if not isinstance(stored, str) or not stored:
        verify_password(body.password, _DUMMY_HASH)
        raise HTTPException(status_code=401, detail="Incorrect username or password.")

    if not verify_password(body.password, stored.encode()):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")

    return LoginResponse(username=user["username"])


def require_admin_key(x_admin_key: str | None = Header(default=None)) -> None:
    """Allow the request only if the X-Admin-Key header matches ADMIN_API_KEY from .env."""
    if not REQUIRE_ADMIN_KEY:
        return
    expected = os.getenv("ADMIN_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="User creation is disabled: ADMIN_API_KEY is not set.")
    # compare_digest takes the same time for any wrong key, so the key can't be guessed by timing.
    if not x_admin_key or not secrets.compare_digest(x_admin_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing admin key.")


@app.post(
    "/auth/users",
    response_model=CreateUserResponse,
    status_code=201,
    dependencies=[Depends(require_admin_key)],
)
def create_user(body: CreateUserRequest) -> CreateUserResponse:
    """Create a login user. Needs the X-Admin-Key header when REQUIRE_ADMIN_KEY is True."""
    try:
        get_supabase().table(USERS_TABLE).insert(
            {"username": body.username, "password_hash": hash_password(body.password)}
        ).execute()
    except APIError as e:
        if e.code == "23505":  # Postgres unique_violation: the username is taken
            raise HTTPException(status_code=409, detail="Username already exists.") from None
        logger.exception("Creating user failed")
        raise HTTPException(status_code=503, detail="Could not create the user right now.") from None
    except Exception:
        logger.exception("Creating user failed")
        raise HTTPException(status_code=503, detail="Could not create the user right now.") from None

    return CreateUserResponse(username=body.username)


@app.put("/sessions/{session_id}/cost", response_model=SessionCostResponse)
def save_session_cost(session_id: uuid.UUID, body: SessionCostRequest) -> SessionCostResponse:
    """Create or update the cost row of one chat session for a user.

    Called again whenever the session's cost changes, with the new running totals, so
    the row always holds the session total. A session can't be moved to another user.
    """
    try:
        db = get_supabase()
        user = db.table(USERS_TABLE).select("id").eq("username", body.username).limit(1).execute()
        if not user.data:
            raise HTTPException(status_code=404, detail="Unknown user.")
        user_id = user.data[0]["id"]

        existing = (
            db.table(SESSION_COSTS_TABLE).select("user_id").eq("session_id", str(session_id)).limit(1).execute()
        )
        if existing.data and existing.data[0]["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="This session belongs to another user.")

        row = {
            "session_id": str(session_id),
            "user_id": user_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **body.model_dump(exclude={"username"}),
        }
        # started_at is only set by the table default on the first insert.
        db.table(SESSION_COSTS_TABLE).upsert(row, on_conflict="session_id").execute()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Saving session cost failed")
        raise HTTPException(status_code=503, detail="Could not save the session cost right now.") from None

    total = body.agent_usd + body.search_usd + body.upload_usd + body.testgen_usd + body.eval_usd
    return SessionCostResponse(session_id=session_id, username=body.username, total_usd=total)
