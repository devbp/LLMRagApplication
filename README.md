# LLM Q&A

A minimal Streamlit app that sends one question to `openai/gpt-4o-mini` through
LangChain's OpenRouter integration.

## Run locally

1. Copy `.env.example` to `.env` and fill in your OpenRouter API key, Supabase
   project URL, and Supabase service role key (Project Settings → API).
2. In the Supabase SQL editor, run [`supabase/schema.sql`](supabase/schema.sql)
   to create the `app_users` table.
3. Install the project dependencies with `uv sync`.
4. Add a login user (you will be asked for the password; only a bcrypt hash is
   stored):

   ```powershell
   uv run python -m llmragap.create_user alice
   ```

5. Start the auth API in one terminal:

   ```powershell
   uv run uvicorn llmragap.auth_api:app --port 8000
   ```

6. Start the app in another terminal:

   ```powershell
   uv run streamlit run src/llmragap/main_app.py
   ```

The Q&A page is shown only after logging in. The Streamlit app sends the
username and password to `POST /auth/login` on the auth API (`AUTH_API_URL`),
which looks the user up in Supabase and checks the password hash. It returns
`200` with the username, or `401` if the login is wrong. Interactive API docs
are at <http://127.0.0.1:8000/docs>.

The service role key bypasses Supabase row-level security, so keep it in the
backend's `.env` only and never expose it to a browser.
The `.env` file is gitignored; do not commit your key.

## Dependency versions

The app's added direct dependencies are pinned in `pyproject.toml`:

- [`streamlit==1.65.0`](https://pypi.org/project/streamlit/1.65.0/) is the newest
  PyPI release compatible with the project's Python `>=3.10` requirement.
- [`python-dotenv==1.2.4`](https://pypi.org/project/python-dotenv/1.2.4/) and
  [`langchain-openrouter==0.2.9`](https://pypi.org/project/langchain-openrouter/0.2.9/)
  were the latest PyPI releases when checked on November 3, 2026.
- The OpenRouter integration package and invocation follow LangChain's
  [ChatOpenRouter documentation](https://docs.langchain.com/oss/python/integrations/chat/openrouter).

`uv.lock` records resolved versions for the full dependency graph.