# College Course Enrollment Assistant

A FastAPI-based assistant that helps students with course enrollment: it checks eligibility, prerequisites, seat availability (via RAG with Azure AI Search), and produces a concise enrollment plan or an escalation ticket for human advisors. The agent uses Azure OpenAI / OpenAI for LLM calls and supports retrieval-augmented generation with Azure AI Search.

Quick start
- Clone and enter project:
  - git clone <repo-url> && cd <repo>
- Create virtualenv and install deps:
  - python -m venv .venv && source .venv/bin/activate
  - pip install -r requirements.txt
- Create .env from .env.example and fill values.
- Run locally:
  - uvicorn agent:app --host 0.0.0.0 --port 8080
- (Optional) Docker:
  - docker build -t enrollment-agent .
  - docker run -p 8000:8000 --env-file .env enrollment-agent

Environment variables (from .env.example)
- REQUIRED for RAG + LLM:
  - AZURE_SEARCH_ENDPOINT
  - AZURE_SEARCH_API_KEY
  - AZURE_SEARCH_INDEX_NAME
  - AZURE_OPENAI_ENDPOINT
  - AZURE_OPENAI_API_KEY
  - AZURE_OPENAI_EMBEDDING_DEPLOYMENT
  - MODEL_PROVIDER (e.g., openai / azure)
  - LLM_MODEL (model/deployment name)
- Common / optional:
  - OPENAI_API_KEY (if using OpenAI provider)
  - USE_KEY_VAULT (true/false) and KEY_VAULT_URI (when using Azure Key Vault)
  - AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET (service principal; skipped if AZURE_USE_DEFAULT_CREDENTIAL=true)
  - CONTENT_SAFETY_* (AZURE_CONTENT_SAFETY_ENDPOINT, AZURE_CONTENT_SAFETY_KEY, CONTENT_SAFETY_* toggles)
  - AZURE_CONTENT_SAFETY_ENDPOINT
  - OBS_AZURE_SQL_* and OBS_DATABASE_TYPE (observability DB; optional)
  - RAG_TOP_K, SELECTED_DOCUMENT_TITLES
- Copy .env.example -> .env and populate the required keys before starting.

API endpoints
- POST /v1/enroll/query
  - Description: Primary endpoint to process enrollment requests.
  - Method: POST
  - Body (JSON): { "user_context": { ... }, "selected_document_titles": [...], "session_id": "optional" }
    - user_context: e.g., student_id, name, email, course_codes, term
    - selected_document_titles: optional list to restrict retrieval
  - Response: structured JSON with fields like summary, eligibility_status, required_actions[], timeline[], escalation_details (if any), raw_llm_response, audit_log_id
- GET /health
  - Method: GET
  - Description: Liveness/health check (returns {"status":"ok"}).

Running tests
- Prepare test environment:
  - Ensure you do not enable Key Vault for local tests (e.g. set USE_KEY_VAULT=false)
- Install test deps (requirements.txt includes pytest):
  - pip install -r requirements.txt
- Run tests:
  - pytest -q
- Note: tests use FastAPI TestClient and async mocks; see tests for environment expectations.

If you need to integrate with institution systems (SIS, calendar, payment, ticketing), implement the corresponding adapters (SISAdapter, CalendarAdapter, PaymentAdapter, EmailAdapter, TicketingAdapter) and provide the adapter credentials via environment variables.