# NOTE: If you see "Unknown pytest.mark.X" warnings, create a conftest.py file with:
# import pytest
# def pytest_configure(config):
#     config.addinivalue_line("markers", "performance: mark test as performance test")
#     config.addinivalue_line("markers", "security: mark test as security test")
#     config.addinivalue_line("markers", "integration: mark test as integration test")

# NOTE: If you see "Unknown pytest.mark.X" warnings, create a conftest.py file with:
# import pytest
# def pytest_configure(config):
#     config.addinivalue_line("markers", "performance: mark test as performance test")
#     config.addinivalue_line("markers", "security: mark test as security test")
#     config.addinivalue_line("markers", "integration: mark test as integration test")


from unittest.mock import MagicMock

import importlib
import os
import time
import json
import types
from types import EnrollmentAgentOrchestrator
import logging

import pytest
from unittest.mock import AsyncMock, Mock

import agent  # must import agent per rules
import config

from fastapi.testclient import TestClient

# Ensure agent._time is available for functions that reference it (observability may inject otherwise)
agent._time = time  # safe assignment for tests

@pytest.fixture(autouse=True)
def clear_env_vars():
    """
    Ensure environment variables used by config initialization are cleared for tests
    unless a test explicitly sets them.
    """
    keys = [
        "MODEL_PROVIDER", "LLM_MODEL",
        "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_API_KEY", "AZURE_SEARCH_INDEX_NAME",
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
        "USE_KEY_VAULT",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


def test_config_initialization_fallback_and_attribute_population():
    """Unit: Reload config module with minimal env and verify attributes exist and settings is instance."""
    # Ensure USE_KEY_VAULT is false so no Key Vault calls attempted
    os.environ["USE_KEY_VAULT"] = "false"
    # Remove potential LLM/provider envs
    os.environ.pop("MODEL_PROVIDER", None)
    os.environ.pop("LLM_MODEL", None)
    # Reload module to trigger _initialize_config
    importlib.reload(config)
    # Check attributes exist on Config class
    assert hasattr(config.Config, "MODEL_PROVIDER")
    assert isinstance(getattr(config.Config, "MODEL_PROVIDER"), (str, type(None)))
    assert hasattr(config.Config, "LLM_MODEL")
    assert isinstance(getattr(config.Config, "LLM_MODEL"), (str, type(None)))
    # AZURE_SEARCH_ENDPOINT should exist and be empty string when not set
    assert hasattr(config.Config, "AZURE_SEARCH_ENDPOINT")
    assert getattr(config.Config, "AZURE_SEARCH_ENDPOINT") in ("", None)
    # settings should be present and be an instance of Config
    assert hasattr(config, "settings")
    assert isinstance(config.settings, config.Config.__class__) or isinstance(config.settings, config.Config)


def test_get_search_client_missing_configuration_raises_value_error(monkeypatch):
    """Unit: get_search_client raises ValueError when Azure Search config missing."""
    # Ensure config attributes are absent / None on agent.Config
    monkeypatch.setattr(agent.Config, "AZURE_SEARCH_ENDPOINT", None, raising=False)
    monkeypatch.setattr(agent.Config, "AZURE_SEARCH_INDEX_NAME", None, raising=False)
    monkeypatch.setattr(agent.Config, "AZURE_SEARCH_API_KEY", None, raising=False)

    with pytest.raises(ValueError) as exc:
        agent.get_search_client()
    assert "Azure Search configuration missing" in str(exc.value)


@pytest.mark.asyncio
async def test_AzureAISearchClient_vector_keyword_search_handles_openai_embedding_failure():
    """Edge-case: embedding creation failure from openai client should bubble up."""
    # openai client factory returns object whose embeddings.create raises
    mock_openai = MagicMock()
    mock_openai.embeddings = MagicMock()
    mock_openai.embeddings.create = AsyncMock(side_effect=RuntimeError("embedding failed"))

    # search client factory not used before embedding; provide dummy callable
    def dummy_search_client_factory():
        return MagicMock()

    client = agent.AzureAISearchClient(search_client_factory=dummy_search_client_factory, openai_client_factory=lambda: mock_openai)
    with pytest.raises(RuntimeError) as exc:
        await client.vector_keyword_search(system_prompt="x", top_k=3, selected_titles=None)
    assert "embedding failed" in str(exc.value)


@pytest.mark.asyncio
async def test_chunkretriever_retrieve_chunks_returns_empty_list_on_search_client_failure():
    """Unit: ChunkRetriever.retrieve_chunks should return [] when vector_keyword_search raises."""
    class EnrollmentAgentOrchestrator:
        async def vector_keyword_search(self, *args, **kwargs):
            raise Exception("search failed")

    retriever = agent.ChunkRetriever(search_client=EnrollmentAgentOrchestrator())
    result = await retriever.retrieve_chunks(query="test", top_k=5, selected_document_titles=None)
    assert isinstance(result, list)
    assert result == []


@pytest.mark.asyncio
async def test_llmservice_call_llm_returns_llmresponse_and_extracts_token_usage():
    """Unit: LLMService.call_llm returns LLMResponse with expected text and tokens_used."""
    # Build mock response shape
    response = EnrollmentAgentOrchestrator(
        choices=[EnrollmentAgentOrchestrator(message=EnrollmentAgentOrchestrator(content="Summary: ok"))],
        usage=EnrollmentAgentOrchestrator(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    # Create openai client with chat.completions.create AsyncMock
    mock_openai = MagicMock()
    mock_openai.chat = EnrollmentAgentOrchestrator(
        completions=EnrollmentAgentOrchestrator(create=AsyncMock(return_value=response))
    )

    svc = agent.LLMService(openai_client_factory=lambda: mock_openai)
    # ensure _time is present for call
    agent._time = time
    result = await svc.call_llm(agent.SYSTEM_PROMPT, "u", ["c1"], llm_params={})
    assert isinstance(result, agent.LLMResponse)
    assert result.text == "Summary: ok"
    assert isinstance(result.tokens_used, dict)
    assert result.tokens_used["prompt_tokens"] == 10
    assert result.tokens_used["completion_tokens"] == 5
    assert result.tokens_used["total_tokens"] == 15


@pytest.mark.asyncio
async def test_enrollmentagentorchestrator_process_query_triggers_escalation_and_ticket_creation():
    """Integration: process_query should create a ticket when rules indicate unknown eligibility."""
    # Mocks
    mock_chunk_retriever = MagicMock()
    mock_chunk_retriever.retrieve_chunks = AsyncMock(return_value=["chunk1"])

    mock_llm_service = MagicMock()
    mock_llm_service.call_llm = AsyncMock(return_value=agent.LLMResponse(text="Summary: needs escalation"))

    mock_rules = MagicMock()
    mock_rules.evaluate = AsyncMock(return_value={"eligibility_status": "unknown", "reasons": ["No student_id provided"]})

    mock_ticketing = MagicMock()
    mock_ticketing.create_ticket = AsyncMock(return_value={"ticket_id": "T-12345"})

    mock_audit = MagicMock()
    mock_audit.log_event = AsyncMock(return_value="audit-1")

    orchestrator = agent.EnrollmentAgentOrchestrator(
        chunk_retriever=mock_chunk_retriever,
        llm_service=mock_llm_service,
        sis_adapter=None,
        calendar_adapter=None,
        payment_adapter=None,
        email_adapter=None,
        ticketing_adapter=mock_ticketing,
        audit_logger=mock_audit,
        rules_engine=mock_rules,
    )

    envelope = await orchestrator.process_query({"name": "Alice", "email": "a@x.com"}, selected_document_titles=None, session_id=None)
    assert isinstance(envelope, dict)
    assert envelope.get("escalation_details") is not None
    assert envelope["escalation_details"]["required"] is True
    assert envelope["escalation_details"]["ticket_id"] == "T-12345"
    # audit_log_id may be present or None; ensure key exists
    assert "audit_log_id" in envelope


def test_fastapi_enroll_query_endpoint_maps_orchestrator_envelope(monkeypatch):
    """Integration: POST /v1/enroll/query maps orchestrator envelope into EnrollmentResponse."""
    client = TestClient(agent.app)

    async def fake_process_query(user_context, selected_document_titles=None, session_id=None):
        return {
            "success": True,
            "summary": "S",
            "eligibility_status": "eligible",
            "required_actions": ["1"],
            "timeline": ["T"],
            "escalation_details": None,
            "raw_llm_response": "S",
            "audit_log_id": "log-1",
        }

    monkeypatch.setattr(agent.agent, "process_query", AsyncMock(side_effect=fake_process_query))
    resp = client.post("/v1/enroll/query", json={"user_context": {"student_id": "12345"}})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["summary"] == "S"
    assert data["eligibility_status"] == "eligible"
    assert data["required_actions"] == ["1"]
    assert data["audit_log_id"] == "log-1"


@pytest.mark.asyncio
async def test_auditlogger_log_event_redacts_sensitive_pii_fields(caplog):
    """Security: AuditLogger.log_event should redact sensitive PII keys in logged payload."""
    caplog.set_level(logging.INFO)
    audit = agent.AuditLogger()
    payload = {"ssn": "123-45-6789", "card_number": "4111111111111111", "note": "ok"}
    log_id = await audit.log_event("enrollment_request", "user-1", payload)
    assert isinstance(log_id, str) and len(log_id) > 0
    # Find a log record containing 'AUDIT [' prefix
    found = False
    for rec in caplog.records:
        if "AUDIT" in rec.getMessage():
            found = True
            msg = rec.getMessage()
            assert "[REDACTED]" in msg
            assert "ok" in msg  # non-sensitive preserved
    assert found, "Expected an AUDIT log entry"


def test_text_extraction_helpers_parse_actions_and_timeline_correctly():
    """Edge-case: helper extractors should parse numbered Actions and Timeline with continuations."""
    orch = agent.EnrollmentAgentOrchestrator()
    actions_text = "Actions:\n1) Do X\n2. Do Y\ncontinued description"
    timeline_text = "Timeline:\n1) 3 days\nNote: follow up"
    actions = orch._extract_actions_from_llm(actions_text)
    timeline = orch._extract_timeline_from_llm(timeline_text)
    assert isinstance(actions, list) and len(actions) >= 2
    assert actions[0].startswith("Do X")
    # second action should include continuation
    assert "Do Y" in actions[1] and "continued description" in actions[1]
    assert isinstance(timeline, list) and len(timeline) >= 1
    assert any("3 days" in t or t.lower().startswith("timeline") for t in timeline)


@pytest.mark.performance
@pytest.mark.asyncio
async def test_process_query_performance_under_mocked_fast_integrations():
    """Performance: process_query completes under generous threshold when integrations are mocked fast."""
    mock_chunk_retriever = MagicMock()
    mock_chunk_retriever.retrieve_chunks = AsyncMock(return_value=[])

    mock_llm_service = MagicMock()
    mock_llm_service.call_llm = AsyncMock(return_value=agent.LLMResponse(text="Summary", tokens_used=None))

    mock_rules = MagicMock()
    mock_rules.evaluate = AsyncMock(return_value={"eligibility_status": "eligible"})

    orchestrator = agent.EnrollmentAgentOrchestrator(
        chunk_retriever=mock_chunk_retriever,
        llm_service=mock_llm_service,
        rules_engine=mock_rules,
    )

    start = time.time()
    envelope = await orchestrator.process_query({"student_id": "100"}, selected_document_titles=None, session_id=None)
    duration = time.time() - start
    assert isinstance(envelope, dict)
    # Per developer rule: be generous with threshold for CI; assert under 30 seconds
    assert duration < 30.0, f"process_query too slow: {duration}s"