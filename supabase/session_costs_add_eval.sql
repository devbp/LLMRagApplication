-- Upgrade an existing session_costs table (created by the first version of
-- session_costs.sql) with the test generation and evaluation costs.
-- Run once in the Supabase SQL editor. Safe to run again.
alter table public.session_costs
    add column if not exists testgen_runs integer not null default 0 check (testgen_runs >= 0),
    add column if not exists testgen_tokens bigint not null default 0 check (testgen_tokens >= 0),
    add column if not exists testgen_usd numeric(14, 8) not null default 0 check (testgen_usd >= 0),
    add column if not exists eval_runs integer not null default 0 check (eval_runs >= 0),
    add column if not exists eval_tokens bigint not null default 0 check (eval_tokens >= 0),
    add column if not exists eval_usd numeric(14, 8) not null default 0 check (eval_usd >= 0);

-- A generated column's formula can't be changed in place: drop it and add it again.
-- (Existing rows get the new total computed automatically.)
alter table public.session_costs drop column if exists total_usd;
alter table public.session_costs
    add column total_usd numeric(14, 8) generated always as
        (agent_usd + search_usd + upload_usd + testgen_usd + eval_usd) stored;
