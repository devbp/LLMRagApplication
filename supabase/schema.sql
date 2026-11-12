-- Run this once in the Supabase SQL editor.
create table if not exists public.app_users (
    id bigint generated always as identity primary key,
    username text not null unique,
    password_hash text not null,
    created_at timestamptz not null default now()
);

-- With RLS on and no policies, the public anon key cannot read this table.
-- Only the backend, which uses the service role key, can.
alter table public.app_users enable row level security;
