-- Cost of each chat session, per user. Run this once in the Supabase SQL editor
-- (after app_users exists). The auth API writes one row per session and keeps
-- updating it while the session runs.
create table if not exists public.session_costs (
    session_id uuid primary key,                -- one per login session in the chat app
    user_id bigint not null references public.app_users (id) on delete cascade,
    started_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    answers integer not null default 0 check (answers >= 0),           -- agent answers
    tokens bigint not null default 0 check (tokens >= 0),              -- agent model tokens
    agent_usd numeric(14, 8) not null default 0 check (agent_usd >= 0),    -- agent model
    search_usd numeric(14, 8) not null default 0 check (search_usd >= 0),  -- query embeddings
    upload_files integer not null default 0 check (upload_files >= 0),
    upload_tokens bigint not null default 0 check (upload_tokens >= 0),
    upload_usd numeric(14, 8) not null default 0 check (upload_usd >= 0),  -- document embeddings
    total_usd numeric(14, 8) generated always as (agent_usd + search_usd + upload_usd) stored
);

-- "All sessions of a user, newest first"
create index if not exists session_costs_user_started_idx
    on public.session_costs (user_id, started_at desc);

-- RLS on and no policies: only the backend (service role / secret key) can read or write.
alter table public.session_costs enable row level security;
