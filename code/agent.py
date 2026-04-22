# AUTO-FIX runtime fallbacks for unresolved names
_obs_startup_log = None

from __future__ import annotations
import time as _time
try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
    from config import settings as _obs_settings
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]
    class _ObsSettingsStub:
        AGENT_NAME: str = 'College Course Enrollment Assistant'
        PROJECT_NAME: str = '19test'
    _obs_settings = _ObsSettingsStub()


"""
Enrollment Agent - FastAPI service implementing a RAG pipeline using Azure AI Search
and Azure OpenAI (embeddings + Chat completions). This module follows the
"College Course Enrollment Assistant" agent design.

Notes:
- Observability imports (trace_step, trace_tool_call, etc.) are injected at runtime.
- Guardrails decorator is applied where appropriate.
- Azure / OpenAI clients are created lazily to avoid requiring credentials at import time.
- The retrieval pipeline uses SYSTEM_PROMPT as the query (per RAG rules) 
"""


import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    from azure.core.credentials import AzureKeyCredential
except ImportError:
    pass
try:
    from azure.search.documents import SearchClient
except ImportError:
    pass
try:
    from azure.search.documents.models import VectorizedQuery
except ImportError:
    pass
import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator
from json.decoder import JSONDecodeError

from config import Config
from modules.guardrails.content_safety_decorator import with_content_safety

# GUARDRAILS_CONFIG must be defined immediately after with_content_safety import
GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

# =============================================================================
# Internal constants (INTERNAL, not exposed via API)
# =============================================================================
SYSTEM_PROMPT: str = (
    "You are an expert college enrollment assistant. When given a user's request, perform the following: "
    "1) Ask clarifying questions if student identity, program, term, or course codes are missing. "
    "2) Use available integrations (SIS, calendar, payment gateway) to validate eligibility, prerequisites, schedule conflicts, and seat availability. "
    "3) Present a concise actionable enrollment plan including steps, required documents, deadlines, and next actions. "
    "4) If an integration call fails or information is missing, provide fallback steps and escalate to a human advisor when appropriate. "
    "Output responses in plain text with numbered steps, and include fields: summary, eligibility_status, required_actions, timeline, escalation (if needed). "
    "Do not attempt to execute payments or reveal PII; instead instruct the user on how to complete those steps securely. "
    "If enrollment cannot be completed automatically, produce a clear handoff message and ticket payload for human staff."
)

OUTPUT_FORMAT: str = (
    "Plain text with labeled sections: Summary, Eligibility, Actions (numbered), Timeline, Escalation or Next Steps. "
    "Include machine-readable JSON ticket payload in code block if escalation is required."
)

FALLBACK_RESPONSE: str = (
    "I could not retrieve the authoritative enrollment data. Please provide your student ID and program, "
    "or contact the admissions office at admissions@example.edu. I can prepare the required steps while you retrieve this information."
)

# Selected titles: empty by default. If needed, update this list in code (internal constant) 
SELECTED_DOCUMENT_TITLES: List[str] = getattr(Config, "SELECTED_DOCUMENT_TITLES", []) or []

# Validation config path
VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# RAG / retrieval defaults
RAG_TOP_K: int = int(getattr(Config, "RAG_TOP_K", 5) or 5)
EMBEDDING_MODEL = getattr(Config, "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", None) or "text-embedding-ada-002"

# Logging
logger = logging.getLogger("enrollment_agent")
logging.basicConfig(level=logging.INFO)


# =============================================================================
# Helpers: lazy client creation
# =============================================================================
def get_search_client() -> SearchClient:
    """
    Lazily create an azure.search.documents.SearchClient
    """
    endpoint = getattr(Config, "AZURE_SEARCH_ENDPOINT", None)
    index_name = getattr(Config, "AZURE_SEARCH_INDEX_NAME", None)
    api_key = getattr(Config, "AZURE_SEARCH_API_KEY", None)

    if not endpoint or not index_name or not api_key:
        raise ValueError("Azure Search configuration missing (AZURE_SEARCH_ENDPOINT / AZURE_SEARCH_INDEX_NAME / AZURE_SEARCH_API_KEY)")

    return SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(api_key),
    )


def get_openai_client() -> openai.AsyncAzureOpenAI:
    """
    Lazily create an openai.AsyncAzureOpenAI client for Azure-hosted OpenAI.
    """
    api_key = getattr(Config, "AZURE_OPENAI_API_KEY", None)
    azure_endpoint = getattr(Config, "AZURE_OPENAI_ENDPOINT", None)

    if not api_key or not azure_endpoint:
        raise ValueError("Azure OpenAI configuration missing (AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT)")

    return openai.AsyncAzureOpenAI(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint=azure_endpoint,
    )


def _get_llm_kwargs() -> dict:
    """
    Use Config.get_llm_kwargs() if available, otherwise derive simple kwargs.
    This ensures compatibility with o-series model parameter requirements.
    """
    try:
        return Config.get_llm_kwargs() or {}
    except Exception:
        # Fallback: build basic llm kwargs
        kwargs = {}
        temp = getattr(Config, "LLM_TEMPERATURE", None)
        max_t = getattr(Config, "LLM_MAX_TOKENS", None) or getattr(Config, "MAX_TOKENS", None)
        if temp is not None:
            kwargs["temperature"] = float(temp)
        if max_t is not None:
            kwargs["max_tokens"] = int(max_t)
        return kwargs


# =============================================================================
# Azure AI Search integration
# =============================================================================
class AzureAISearchClient:
    """
    Adapter for Azure AI Search vector + keyword search using SearchClient.
    """

    def __init__(self, search_client_factory=get_search_client, openai_client_factory=get_openai_client):
        self._search_client_factory = search_client_factory
        self._openai_client_factory = openai_client_factory

    async def vector_keyword_search(self, system_prompt: str, top_k: int = RAG_TOP_K, selected_titles: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """
        Embed the system_prompt using Azure OpenAI embeddings, then perform
        vector + keyword search using SearchClient.VectorizedQuery.

        Returns list of dicts with keys: "chunk", "title".
        """
        try:
            search_client = self._search_client_factory()
        except Exception as e:
            logger.warning("Azure Search client init failed: %s", e)
            raise

        try:
            openai_client = self._openai_client_factory()
        except Exception as e:
            logger.warning("Azure OpenAI client init failed: %s", e)
            raise

        # 1) Create embedding for the SYSTEM_PROMPT
        try:
            embedding_resp = await openai_client.embeddings.create(
                input=system_prompt,
                model=EMBEDDING_MODEL,
            )
            vector = embedding_resp.data[0].embedding
        except Exception as e:
            logger.warning("Embedding creation failed: %s", e)
            raise

        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=int(top_k or RAG_TOP_K),
            fields="vector",
        )

        # Build search kwargs; per spec use SYSTEM_PROMPT as search_text
        search_kwargs: dict = {
            "search_text": system_prompt,
            "vector_queries": [vector_query],
            "top": int(top_k or RAG_TOP_K),
            "select": ["chunk", "title"],
        }

        # Apply OData filter by title when selected titles provided (internal constant or parameter)
        titles = selected_titles if selected_titles is not None else SELECTED_DOCUMENT_TITLES
        if titles:
            # Validate titles are safe (naive validation: type and no quotes to avoid injection)
            safe_titles = []
            for t in titles:
                if isinstance(t, str) and "'" not in t and '"' not in t:
                    safe_titles.append(t)
                else:
                    logger.warning("Ignoring unsafe selected document title for OData filter: %r", t)
            if safe_titles:
                odata_parts = [f"title eq '{t}'" for t in safe_titles]
                search_kwargs["filter"] = " or ".join(odata_parts)

        # Execute search
        try:
            results = search_client.search(**search_kwargs)
        except Exception as e:
            logger.warning("Azure Search query failed: %s", e)
            raise

        context_chunks: List[Dict[str, str]] = []
        try:
            for r in results:
                # The index schema only contains 'chunk', 'title', 'vector'. Respect that.
                chunk_text = r.get("chunk") or r.get("content") or None
                title_text = r.get("title") or None
                if chunk_text:
                    context_chunks.append({"chunk": str(chunk_text), "title": title_text})
                if len(context_chunks) >= top_k:
                    break
        except Exception as e:
            logger.warning("Failed to iterate search results: %s", e)

        return context_chunks


# =============================================================================
# Chunk retriever orchestrator
# =============================================================================
class ChunkRetriever:
    """
    High-level orchestrator that uses AzureAISearchClient to retrieve top-k chunks.
    """

    def __init__(self, search_client: Optional[AzureAISearchClient] = None):
        self.search_client = search_client or AzureAISearchClient()

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, top_k: int = RAG_TOP_K, selected_document_titles: Optional[List[str]] = None) -> List[str]:
        """
        Per RAG rules: Use SYSTEM_PROMPT (not user query) to perform retrieval.
        But we accept `query` param for logging/observability only.
        """
        try:
            async with trace_step(
                "retrieve_chunks",
                step_type="tool_call",
                decision_summary="Retrieve relevant chunks from Azure AI Search using SYSTEM_PROMPT as query",
                output_fn=lambda r: f"chunks={len(r)}",
            ) as step:
                # The retrieval client embeds SYSTEM_PROMPT (as required)
                chunks = await self.search_client.vector_keyword_search(SYSTEM_PROMPT, top_k=top_k, selected_titles=selected_document_titles)
                # Keep only chunk texts per pipeline rules — do not parse the chunk contents
                chunk_texts = [c.get("chunk") for c in chunks if c.get("chunk")]
                step.capture({"chunks": chunk_texts})
                return chunk_texts
        except Exception as e:
            logger.exception("Chunk retrieval error: %s", e)
            # On repeated failure return empty list per spec and let orchestrator fallback
            return []


# =============================================================================
# LLM Service
# =============================================================================
class LLMResponse(BaseModel):
    text: str
    tokens_used: Optional[Dict[str, int]] = None
    raw_payload: Optional[dict] = None


class LLMService:
    """
    Wraps calls to Azure OpenAI chat completions (AsyncAzureOpenAI) 
    """

    def __init__(self, openai_client_factory=get_openai_client):
        self._openai_client_factory = openai_client_factory
        self.model = getattr(Config, "LLM_MODEL", None) or getattr(Config, "MODEL", None) or "gpt-5-mini"

    @with_content_safety(config=GUARDRAILS_CONFIG)
    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    async def call_llm(self, system_prompt: str, user_prompt: str, context_chunks: List[str], llm_params: Optional[dict] = None) -> LLMResponse:
        """
        Compose messages and call the LLM. Append OUTPUT_FORMAT to system prompt per spec.
        The LLM interprets the context chunks — the code does not parse them.
        """
        llm_params = llm_params or {}
        openai_client = None
        try:
            openai_client = self._openai_client_factory()
        except Exception as e:
            logger.warning("LLM client init failed: %s", e)
            raise

        system_msg = system_prompt + "\n\nOutput Format: " + OUTPUT_FORMAT
        # Build user message including context documents
        context_text = ""
        if context_chunks:
            context_entries = []
            for i, c in enumerate(context_chunks, start=1):
                # label each chunk minimally
                context_entries.append(f"[Context #{i}]\n{c}")
            context_text = "\n\n".join(context_entries)

        user_content_parts = []
        if context_text:
            user_content_parts.append("Context Documents:\n" + context_text)
        user_content_parts.append("User Request:\n" + (user_prompt or ""))
        user_content = "\n\n".join(user_content_parts)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        _llm_kwargs = {**_get_llm_kwargs(), **llm_params}
        try:
            async with trace_step(
                "generate_response",
                step_type="llm_call",
                decision_summary="Call LLM to generate an enrollment plan using system prompt + context chunks",
                output_fn=lambda r: f"len={len(r) if r else 0}",
            ) as step:
                _obs_t0 = _time.time()
                response = await openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **_llm_kwargs,
                )
                try:
                    trace_model_call(
                        provider='azure',
                        model_name=(getattr(self, "model", None) or getattr(getattr(self, "config", None), "model", None) or "unknown"),
                        prompt_tokens=(getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0),
                        completion_tokens=(getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0),
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                    )
                except Exception:
                    pass

                # Access content safely
                content = ""
                try:
                    content = response.choices[0].message.content
                except Exception:
                    # Some models may return alternative shapes
                    content = getattr(response.choices[0].message, "content", "") if response.choices else ""

                # Attempt to capture token usage if available (Azure response shapes vary)
                tokens_used = None
                try:
                    usage = getattr(response, "usage", None)
                    if usage:
                        tokens_used = {
                            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                        }
                except Exception:
                    tokens_used = None

                step.capture(content)
                return LLMResponse(text=content or "", tokens_used=tokens_used, raw_payload=response.__dict__ if hasattr(response, "__dict__") else None)
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            raise


# =============================================================================
# Integration adapter stubs (SIS, Calendar, Payment, Email, Ticketing)
# These are minimal adapters that return simulated responses. In production,
# each adapter would be implemented to call external services with retry and
# observability instrumentation.
# =============================================================================
class IntegrationAdapter:
    """Base interface for adapters."""

    async def call(self, *args, **kwargs) -> dict:
        raise NotImplementedError


class SISAdapter(IntegrationAdapter):
    async def get_student_profile(self, student_id: str) -> dict:
        # Stubbed: real implementation calls SIS API
        logger.debug("SISAdapter.get_student_profile called for %s", student_id)
        return {"student_id": student_id, "program_status": "active", "email": "student@example.edu"}


class CalendarAdapter(IntegrationAdapter):
    async def get_schedule(self, student_id: str, term: str) -> dict:
        logger.debug("CalendarAdapter.get_schedule called for %s %s", student_id, term)
        return {"conflicts": False, "scheduled_courses": []}


class PaymentAdapter(IntegrationAdapter):
    async def create_invoice(self, student_id: str, items: list) -> dict:
        logger.debug("PaymentAdapter.create_invoice called for %s", student_id)
        return {"invoice_id": str(uuid4()), "status": "pending"}


class EmailAdapter(IntegrationAdapter):
    async def send_email(self, recipient: str, subject: str, body: str) -> dict:
        logger.debug("EmailAdapter.send_email called to %s", recipient)
        return {"message_id": str(uuid4()), "status": "sent"}


class TicketingAdapter(IntegrationAdapter):
    async def create_ticket(self, payload: dict) -> dict:
        logger.debug("TicketingAdapter.create_ticket called")
        return {"ticket_id": f"T-{int(time.time())}", "status": "created"}


# =============================================================================
# Audit logger (simple implementation for the example)
# =============================================================================
class AuditLogger:
    """Record an audit event. Real implementation would persist to secure store."""

    async def log_event(self, event_type: str, user_id: Optional[str], payload: dict) -> str:
        # Minimal redaction: do not store raw PII - replace common PII keys
        redacted = dict(payload)
        for k in list(redacted.keys()):
            if k and any(tok in k.lower() for tok in ("ssn", "dob", "birth", "card", "credit", "cvv")):
                redacted[k] = "[REDACTED]"
        log_id = str(uuid4())
        logger.info("AUDIT [%s] user=%s payload=%s", log_id, user_id, json.dumps(redacted, default=str)[:1000])
        return log_id


# =============================================================================
# Business rules engine (stub)
# =============================================================================
class BusinessRulesEngine:
    async def evaluate(self, user_context: dict, integration_results: Optional[dict] = None) -> dict:
        # Stubbed evaluation: in real life, this runs the rule sets described.
        logger.debug("BusinessRulesEngine.evaluate called")
        # Minimal outcome: assume eligible unless transcript absent
        eligibility = "unknown"
        reasons = []
        if user_context.get("student_id"):
            eligibility = "eligible"
        else:
            eligibility = "unknown"
            reasons.append("No student_id provided")
        return {"eligibility_status": eligibility, "reasons": reasons}


# =============================================================================
# Orchestrator - main agent class
# =============================================================================
class EnrollmentAgentOrchestrator:
    """
    Main orchestrator that wires retrieval, LLM, business rules, integration adapters,
    and auditing to produce a response.
    """

    def __init__(
        self,
        chunk_retriever: Optional[ChunkRetriever] = None,
        llm_service: Optional[LLMService] = None,
        sis_adapter: Optional[SISAdapter] = None,
        calendar_adapter: Optional[CalendarAdapter] = None,
        payment_adapter: Optional[PaymentAdapter] = None,
        email_adapter: Optional[EmailAdapter] = None,
        ticketing_adapter: Optional[TicketingAdapter] = None,
        audit_logger: Optional[AuditLogger] = None,
        rules_engine: Optional[BusinessRulesEngine] = None,
    ):
        self.chunk_retriever = chunk_retriever or ChunkRetriever()
        self.llm_service = llm_service or LLMService()
        self.sis = sis_adapter or SISAdapter()
        self.calendar = calendar_adapter or CalendarAdapter()
        self.payment = payment_adapter or PaymentAdapter()
        self.email = email_adapter or EmailAdapter()
        self.ticketing = ticketing_adapter or TicketingAdapter()
        self.audit = audit_logger or AuditLogger()
        self.rules = rules_engine or BusinessRulesEngine()
        # Agent metadata read from Config
        self.agent_name = getattr(Config, "AGENT_NAME", "EnrollmentAgent")
        self.agent_version = getattr(Config, "SERVICE_VERSION", getattr(Config, "AGENT_VERSION", None))

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_query(self, user_context: Dict[str, Any], selected_document_titles: Optional[List[str]] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Top-level entry:
        - Retrieve chunks using SYSTEM_PROMPT (per RAG rules)
        - Call LLM with system prompt (enhanced) + retrieved chunks + user context
        - Apply business rules
        - Optionally perform integration adapter calls (stubbed)
        - Audit log the interaction
        Returns structured envelope and plain text response.
        """
        # Ensure default selected titles list
        selected_document_titles = selected_document_titles or SELECTED_DOCUMENT_TITLES or []

        # 1) Retrieval
        try:
            async with trace_step(
                "orchestrator_retrieve",
                step_type="tool_call",
                decision_summary="Retrieve contextual chunks for LLM using SYSTEM_PROMPT",
                output_fn=lambda r: f"chunks={len(r.get('chunks', [])) if isinstance(r, dict) else '?'}",
            ) as step:
                chunks = await self.chunk_retriever.retrieve_chunks(query=SYSTEM_PROMPT, top_k=RAG_TOP_K, selected_document_titles=selected_document_titles)
                step.capture({"chunks": chunks})
        except Exception as e:
            logger.exception("Retrieval stage failed: %s", e)
            # Continue with empty chunks (fallback)
            chunks = []

        # 2) Compose user prompt from user_context
        # The system prompt is fixed (SYSTEM_PROMPT). The user prompt should summarize the user_context.
        user_prompt = self._format_user_context(user_context)

        # 3) Call LLM
        try:
            llm_response = await self.llm_service.call_llm(SYSTEM_PROMPT, user_prompt, chunks, llm_params={
                "temperature": getattr(Config, "LLM_TEMPERATURE", 0.7),
                "max_tokens": getattr(Config, "LLM_MAX_TOKENS", 2000),
            })
            raw_text = llm_response.text or ""
        except Exception:
            logger.exception("LLM call failed, using fallback response")
            raw_text = FALLBACK_RESPONSE
            llm_response = LLMResponse(text=raw_text, tokens_used=None, raw_payload=None)

        # 4) Business rules (evaluate)
        try:
            async with trace_step(
                "business_rules",
                step_type="process",
                decision_summary="Evaluate business rules (eligibility, prereqs, capacity)",
                output_fn=lambda r: f"elig={r.get('eligibility_status','?')}",
            ) as step:
                rules_result = await self.rules.evaluate(user_context, integration_results=None)
                step.capture(rules_result)
        except Exception as e:
            logger.exception("Business rules evaluation failed: %s", e)
            rules_result = {"eligibility_status": "unknown", "reasons": ["rules_evaluation_failed"]}

        # 5) Decide whether escalation/ticket is needed (simple heuristic: if rules indicate hard stop or unknown)
        escalation_payload = None
        escalation_ticket_id = None
        need_escalation = False
        try:
            if rules_result.get("eligibility_status") in ("unknown", "ineligible"):
                need_escalation = True

            if need_escalation:
                # Prepare a minimal ticket payload and create ticket via adapter
                ticket_payload = {
                    "user_context": user_context,
                    "rules_result": rules_result,
                    "llm_excerpt": (raw_text[:200] + "...") if raw_text else "",
                    "requested_at": int(time.time()),
                }
                async with trace_step(
                    "create_ticket",
                    step_type="tool_call",
                    decision_summary="Create escalation ticket via ticketing adapter",
                    output_fn=lambda r: f"ticket={r.get('ticket_id','?') if isinstance(r, dict) else '?'}",
                ) as step:
                    try:
                        ticket_resp = await self.ticketing.create_ticket(ticket_payload)
                        escalation_ticket_id = ticket_resp.get("ticket_id")
                        escalation_payload = ticket_payload
                        step.capture(ticket_resp)
                    except Exception as e:
                        logger.exception("Ticket creation failed: %s", e)
                        escalation_ticket_id = None
                        escalation_payload = ticket_payload
                        step.capture({"error": str(e)})
        except Exception as e:
            logger.exception("Escalation decision failed: %s", e)

        # 6) Audit log the run (redacting sensitive keys)
        try:
            audit_payload = {
                "user_context": user_context,
                "rules_result": rules_result,
                "escalation_ticket_id": escalation_ticket_id,
            }
            audit_id = await self.audit.log_event("enrollment_request", user_context.get("student_id") or user_context.get("email"), audit_payload)
        except Exception as e:
            logger.exception("Audit logging failed: %s", e)
            audit_id = None

        # 7) Build structured response envelope
        envelope = {
            "success": True,
            "summary": self._extract_summary_from_llm(raw_text) or "",
            "eligibility_status": rules_result.get("eligibility_status"),
            "required_actions": self._extract_actions_from_llm(raw_text),
            "timeline": self._extract_timeline_from_llm(raw_text),
            "escalation_details": {
                "required": need_escalation,
                "ticket_id": escalation_ticket_id,
                "ticket_payload": escalation_payload,
            } if need_escalation else None,
            "raw_llm_response": raw_text,
            "audit_log_id": audit_id,
        }

        return envelope

    # Helper extractors (delegated to LLM in real flow; here we do minimal heuristics)
    def _format_user_context(self, user_context: Dict[str, Any]) -> str:
        """
        Create a concise user_prompt string for the LLM from structured user_context.
        This is NOT the system prompt; the LLM is guided by SYSTEM_PROMPT (internal) 
        """
        try:
            parts = []
            if not user_context:
                return ""
            for k, v in user_context.items():
                if v is None:
                    continue
                if isinstance(v, (dict, list)):
                    try:
                        parts.append(f"{k}: {json.dumps(v, default=str)}")
                    except Exception:
                        parts.append(f"{k}: {str(v)}")
                else:
                    parts.append(f"{k}: {str(v)}")
            return "\n".join(parts)
        except Exception:
            return str(user_context)

    def _extract_summary_from_llm(self, llm_text: str) -> str:
        # Minimal: attempt to return first 400 chars or first "Summary:" section
        if not llm_text:
            return ""
        try:
            low = llm_text.lower()
            idx = low.find("summary:")
            if idx >= 0:
                # extract up to next double newline or 500 chars
                rest = llm_text[idx:]
                parts = rest.split("\n\n", 1)
                return parts[0].strip()[:800]
        except Exception:
            pass
        return llm_text.strip()[:800]

    def _extract_actions_from_llm(self, llm_text: str) -> List[str]:
        # Attempt to find "Actions" section and extract numbered list
        if not llm_text:
            return []
        try:
            low = llm_text.lower()
            start = low.find("actions")
            if start >= 0:
                # slice from "actions" and split lines
                sub = llm_text[start:]
                lines = sub.splitlines()
                actions = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line[0].isdigit() and (line[1] == ")" or line[1] == "." or line[1].isdigit()):
                        # simple numbered line
                        # remove leading "1)" or "1."
                        cleaned = line.lstrip("0123456789.) ").strip()
                        actions.append(cleaned)
                    elif line.lower().startswith("actions"):
                        continue
                    elif actions:
                        # continuation of the last action
                        actions[-1] = actions[-1] + " " + line
                return actions[:20]
        except Exception:
            pass
        return []

    def _extract_timeline_from_llm(self, llm_text: str) -> List[str]:
        if not llm_text:
            return []
        try:
            low = llm_text.lower()
            start = low.find("timeline")
            if start >= 0:
                sub = llm_text[start:]
                lines = sub.splitlines()
                timeline = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if any(ch.isdigit() for ch in line[:4]) or line.lower().startswith("timeline"):
                        timeline.append(line)
                    elif timeline:
                        timeline[-1] = timeline[-1] + " " + line
                return timeline[:20]
        except Exception:
            pass
        return []


# =============================================================================
# FastAPI request/response models and validators
# =============================================================================
MAX_TEXT_LENGTH = 50000


class EnrollmentQueryRequest(BaseModel):
    user_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User-provided context such as student_id, name, email, course_codes, term")
    selected_document_titles: Optional[List[str]] = Field(default_factory=list, description="Optional document titles to restrict retrieval")
    session_id: Optional[str] = Field(None, description="Optional session id for tracing")

    @field_validator("user_context")
    def _validate_user_context(cls, v):
        # Ensure it's a dict and not too large when serialized
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("user_context must be a JSON object")
        s = json.dumps(v, default=str)
        if len(s) > MAX_TEXT_LENGTH:
            raise ValueError(f"user_context too large (max={MAX_TEXT_LENGTH} characters)")
        return v

    @field_validator("selected_document_titles")
    def _validate_titles(cls, v):
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("selected_document_titles must be a list of strings")
        safe = []
        for t in v:
            if not isinstance(t, str):
                continue
            t2 = t.strip()
            if len(t2) == 0:
                continue
            if len(t2) > 300:
                t2 = t2[:300]
            safe.append(t2)
        return safe


class EnrollmentResponse(BaseModel):
    success: bool
    summary: Optional[str] = None
    eligibility_status: Optional[str] = None
    required_actions: Optional[List[str]] = None
    timeline: Optional[List[str]] = None
    escalation_details: Optional[dict] = None
    raw_llm_response: Optional[str] = None
    audit_log_id: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# FastAPI application and observability lifespan
# =============================================================================
# NOTE: The runtime injects observability imports and variables including
# _obs_startup_log used below. Do not re-import observability.* modules here.
_obs_startup_logger = _obs_startup_log.getLogger(__name__)  # runtime-injected variable

@asynccontextmanager
async def _obs_lifespan(application: FastAPI):
    """Initialise observability on startup, clean up on shutdown."""
    # Log guardrails configuration (must be present when GUARDRAILS_CONFIG defined)
    yield

app = FastAPI(
    title="College Course Enrollment Assistant",
    description="Enrollment Agent that uses Azure AI Search + Azure OpenAI to provide enrollment guidance.",
    version=getattr(Config, "SERVICE_VERSION", "1.0.0"),
    lifespan=_obs_lifespan,
)


# Instantiate orchestrator (singleton for app lifetime)
agent = EnrollmentAgentOrchestrator()


# =============================================================================
# FastAPI exception handlers
# =============================================================================
@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning("Validation error for request %s: %s", request.url.path, exc)
    return JSONResponse(status_code=422, content={"success": False, "error": "Invalid request payload", "details": str(exc)})


@app.exception_handler(JSONDecodeError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def json_decode_exception_handler(request: Request, exc: JSONDecodeError):
    logger.warning("JSON parse error for request %s: %s", request.url.path, exc)
    tips = "Check JSON formatting (quotes, commas). Maximum text length: 50,000 characters."
    return JSONResponse(status_code=400, content={"success": False, "error": "Malformed JSON", "details": str(exc), "tips": tips})


# =============================================================================
# API endpoints
# =============================================================================
@app.post("/v1/enroll/query", response_model=EnrollmentResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def enroll_query(req: EnrollmentQueryRequest):
    """
    Public endpoint to process enrollment queries.

    Accepts:
    - user_context (dict): student_id, name, email, requested courses, term, etc.
    - selected_document_titles (list[str]): optional titles to filter retrieval
    - session_id: optional session identifier for tracing

    The SYSTEM_PROMPT is internal and not accepted from clients.
    """
    try:
        # Set trace context ids if provided (optional). set_trace_context_ids is injected at runtime.
        try:
            if req.session_id:
                # set_trace_context_ids expects UUID objects; we pass as-is for the runtime to handle safely
                set_trace_context_ids(session_id=req.session_id)
        except Exception:
            # Non-fatal: best-effort
            logger.debug("Failed to set trace context ids (non-fatal)")

        # Orchestrate processing
        async with trace_step(
            "handle_user_query",
            step_type="process",
            decision_summary="Top-level orchestration: retrieval, LLM, business rules, audit",
            output_fn=lambda r: f"success={r.get('success', False)}" if isinstance(r, dict) else "?"
        ) as step:
            result = await agent.process_query(user_context=req.user_context, selected_document_titles=req.selected_document_titles, session_id=req.session_id)
            step.capture(result)
    except Exception as e:
        logger.exception("Unhandled error in /v1/enroll/query: %s", e)
        # Attempt audit log of failure (best-effort)
        try:
            _audit = AuditLogger()
            _ = await _audit.log_event("enrollment_request_error", req.user_context.get("student_id") if req.user_context else None, {"error": str(e)})
        except Exception:
            pass
        return EnrollmentResponse(success=False, error=str(e))
    finally:
        # Clear trace context ids (best-effort)
        try:
            clear_trace_context_ids()
        except Exception:
            pass

    # Map orchestrator envelope into response model
    return EnrollmentResponse(**{
        "success": result.get("success", True),
        "summary": result.get("summary"),
        "eligibility_status": result.get("eligibility_status"),
        "required_actions": result.get("required_actions"),
        "timeline": result.get("timeline"),
        "escalation_details": result.get("escalation_details"),
        "raw_llm_response": result.get("raw_llm_response"),
        "audit_log_id": result.get("audit_log_id"),
        "error": None,
    })


@app.get("/health")
async def health():
    return {"status": "ok"}


# =============================================================================
# Run server (async-compatible uvicorn server start inside __main__)
# =============================================================================


async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    try:
        import uvicorn
    except ImportError:
        pass
    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            # uvicorn internals
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            # agent application loggers
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            # observability / tracing namespace
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            # config / settings namespace
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            # suppress noisy azure-sdk logs
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_agent())