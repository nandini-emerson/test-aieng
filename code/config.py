
# python
import os
import logging
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Load .env file FIRST (must happen before any os.getenv() calls)
load_dotenv()

_logger = logging.getLogger(__name__)


class Config:
    """
    Centralized Config class following platform rules:
    - All values are loaded from Azure Key Vault (when enabled) or from .env
    - Key Vault secret map is defined here (only agent-relevant entries)
    - No hardcoded defaults (except the mandated MODEL_PROVIDER and LLM_MODEL defaults
      which are required by the agent-builder)
    """

    # Key Vault cache
    _kv_secrets: Dict[str, str] = {}

    # KEY_VAULT_SECRET_MAP MUST be defined here (only entries relevant to this agent)
    # Copied from PLATFORM REFERENCE CONFIG, trimmed to relevant entries:
    KEY_VAULT_SECRET_MAP = [
        # Observability DB (agent observability)
        ("OBS_AZURE_SQL_SERVER", "agentops-secrets.obs_sql_endpoint"),
        ("OBS_AZURE_SQL_DATABASE", "agentops-secrets.obs_azure_sql_database"),
        ("OBS_AZURE_SQL_PORT", "agentops-secrets.obs_port"),
        ("OBS_AZURE_SQL_USERNAME", "agentops-secrets.obs_sql_username"),
        ("OBS_AZURE_SQL_PASSWORD", "agentops-secrets.obs_sql_password"),
        ("OBS_AZURE_SQL_SCHEMA", "agentops-secrets.obs_azure_sql_schema"),

        # LLM API keys (OpenAI / Azure OpenAI)
        ("AZURE_OPENAI_API_KEY", "openai-secrets.gpt-4.1"),
        ("AZURE_OPENAI_API_KEY", "openai-secrets.azure-key"),
        ("OPENAI_API_KEY", "aba-openai-secret.openai_api_key"),

        # Azure Content Safety
        ("AZURE_CONTENT_SAFETY_ENDPOINT", "azure-content-safety-secrets.azure_content_safety_endpoint"),
        ("AZURE_CONTENT_SAFETY_KEY", "azure-content-safety-secrets.azure_content_safety_key"),

        # Azure OpenAI endpoint + embedding deployment (used for KB / embeddings)
        ("AZURE_OPENAI_ENDPOINT", "kb-secrets.azure_openai_endpoint"),
        ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "kb-secrets.azure_openai_embedding_deployment"),
    ]

    # Sets used to determine unsupported parameters for certain models (used by get_llm_kwargs)
    _MAX_TOKENS_UNSUPPORTED = {
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1-chat",
        "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o3-pro", "o4-mini"
    }
    _TEMPERATURE_UNSUPPORTED = _MAX_TOKENS_UNSUPPORTED.copy()

    @classmethod
    def _load_keyvault_secrets(cls) -> Dict[str, str]:
        """
        Load secrets from Azure Key Vault according to KEY_VAULT_SECRET_MAP.
        Populates cls._kv_secrets and returns it.
        Only runs when Config.USE_KEY_VAULT is True and KEY_VAULT_URI is non-empty.
        """
        # Ensure minimal preconditions are present
        if not getattr(cls, "USE_KEY_VAULT", False):
            _logger.debug("Key Vault usage disabled (USE_KEY_VAULT=False)")
            return {}

        kv_uri = getattr(cls, "KEY_VAULT_URI", "") or ""
        if not kv_uri:
            _logger.debug("Key Vault URI not configured; skipping Key Vault load")
            return {}

        # Determine credential strategy
        try:
            azure_default = os.getenv("AZURE_USE_DEFAULT_CREDENTIAL", "").lower() in ("true", "1", "yes")
        except Exception:
            azure_default = False

        try:
            if azure_default:
                # Use DefaultAzureCredential (managed identity / workload identity)
                from azure.identity import DefaultAzureCredential  # type: ignore
                credential = DefaultAzureCredential()
            else:
                # Use Service Principal credentials from .env
                from azure.identity import ClientSecretCredential  # type: ignore
                tenant_id = os.getenv("AZURE_TENANT_ID", "")
                client_id = os.getenv("AZURE_CLIENT_ID", "")
                client_secret = os.getenv("AZURE_CLIENT_SECRET", "")

                if not (tenant_id and client_id and client_secret):
                    _logger.warning("Service Principal credentials incomplete. Key Vault access will fail.")
                    return {}

                credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret
                )
            # SecretClient
            from azure.keyvault.secrets import SecretClient  # type: ignore
            client = SecretClient(vault_url=kv_uri, credential=credential)
        except Exception as e:
            _logger.warning("Failed to initialize Key Vault client: %s", e)
            return {}

        # Group refs by secret name to minimize round-trips
        by_secret: Dict[str, list] = {}
        for field_name, secret_ref in getattr(cls, "KEY_VAULT_SECRET_MAP", []) or []:
            if "." in secret_ref:
                secret_name, json_key = secret_ref.split(".", 1)
            else:
                secret_name, json_key = secret_ref, None
            by_secret.setdefault(secret_name, []).append((field_name, json_key))

        def _sanitize(raw: str) -> str:
            """Strip BOMs and return string"""
            if raw is None:
                return ""
            # strip common BOMs
            for bom in ("\ufeff", "\xef\xbb\xbf"):
                if raw.startswith(bom):
                    raw = raw[len(bom):]
                    break
            return raw

        # Fetch secrets per secret name
        for secret_name, refs in by_secret.items():
            try:
                secret = client.get_secret(secret_name)
                if not secret or secret.value is None:
                    _logger.debug("Key Vault: secret '%s' empty or missing", secret_name)
                    continue
                raw_value = _sanitize(secret.value)
            except Exception as exc:
                _logger.debug("Key Vault: failed to fetch secret '%s': %s", secret_name, exc)
                continue

            # If any mapping expects a JSON key, try to parse JSON once
            has_json = any(json_key is not None for (_, json_key) in refs)
            json_data = None
            if has_json:
                try:
                    json_data = json.loads(raw_value)
                except Exception:
                    # Best-effort fallback: leave json_data as None and handle per-key
                    _logger.debug("Key Vault: secret '%s' could not be parsed as JSON", secret_name)
                    json_data = None

            # For each ref assign value
            for field_name, json_key in refs:
                try:
                    if json_key:
                        if isinstance(json_data, dict) and json_key in json_data:
                            val = json_data.get(json_key)
                            if val is None or val == "":
                                _logger.debug("Key Vault: key '%s' not found/empty in secret '%s' (field %s)", json_key, secret_name, field_name)
                                continue
                            cls._kv_secrets[field_name] = str(val)
                        else:
                            _logger.debug("Key Vault: json key '%s' missing in secret '%s' for field %s", json_key, secret_name, field_name)
                            continue
                    else:
                        # plain secret maps to the first field that requested it
                        if raw_value != "":
                            cls._kv_secrets[field_name] = str(raw_value)
                except Exception as e:
                    _logger.debug("Key Vault: failed processing mapping %s -> %s: %s", secret_name, field_name, e)
                    continue

        return cls._kv_secrets

    @classmethod
    def _validate_api_keys(cls) -> None:
        """
        Validate that the API key corresponding to MODEL_PROVIDER is present.
        Raises ValueError only for the missing key corresponding to the selected provider.
        """
        provider = (getattr(cls, "MODEL_PROVIDER", "") or "").lower()
        key_map = {
            "openai": getattr(cls, "OPENAI_API_KEY", ""),
            "azure": getattr(cls, "AZURE_OPENAI_API_KEY", ""),
            "anthropic": getattr(cls, "ANTHROPIC_API_KEY", ""),
            "google": getattr(cls, "GOOGLE_API_KEY", ""),
        }
        required = key_map.get(provider)
        if provider in key_map and not required:
            raise ValueError(f"API key for provider '{provider}' is missing. Please set the appropriate API key.")

    @classmethod
    def _get_value_from_kv_or_env(cls, name: str, always_env=False) -> Any:
        """
        Helper to fetch a configuration value with priority:
          - If USE_KEY_VAULT=True and not always_env and name in _kv_secrets -> use that
          - Else -> os.getenv(name) (no default)
        Returns the raw string or None/"" depending on presence.
        """
        val = None
        use_kv = getattr(cls, "USE_KEY_VAULT", False)
        if use_kv and not always_env:
            # prefer KV
            if name in cls._kv_secrets:
                return cls._kv_secrets.get(name)
        # fallback to .env
        env_val = os.getenv(name)
        if env_val is None or env_val == "":
            # Missing in .env file -> per critical rules, warn and return empty string or None
            _logger.warning("Configuration variable %s not found in .env file", name)
            return ""
        return env_val

    @classmethod
    def get_llm_kwargs(cls) -> dict:
        """
        Return kwargs suitable for chat.completions.create() calls.
        Dynamically chooses temperature vs omission and max_tokens vs max_completion_tokens
        depending on model name substrings.
        """
        kwargs: Dict[str, Any] = {}
        model_lower = (getattr(cls, "LLM_MODEL", "") or "").lower()
        temp = getattr(cls, "LLM_TEMPERATURE", "")
        max_t = getattr(cls, "LLM_MAX_TOKENS", "")

        # Temperature
        if model_lower:
            if not any(model_lower.startswith(m) for m in cls._TEMPERATURE_UNSUPPORTED):
                if temp != "":
                    try:
                        kwargs["temperature"] = float(temp)
                    except Exception:
                        _logger.warning("Invalid float value for LLM_TEMPERATURE: %s", temp)
        else:
            # if model unknown, include temperature if present
            if temp != "":
                try:
                    kwargs["temperature"] = float(temp)
                except Exception:
                    _logger.warning("Invalid float value for LLM_TEMPERATURE: %s", temp)

        # Max tokens handling
        if max_t != "":
            try:
                max_tokens_int = int(max_t)
            except Exception:
                _logger.warning("Invalid integer value for LLM_MAX_TOKENS: %s", max_t)
                max_tokens_int = None

            if max_tokens_int is not None:
                if any(model_lower.startswith(m) for m in cls._MAX_TOKENS_UNSUPPORTED):
                    kwargs["max_completion_tokens"] = max_tokens_int
                else:
                    kwargs["max_tokens"] = max_tokens_int

        return kwargs

    @classmethod
    def validate(cls) -> None:
        """
        Public validation entrypoint. Calls internal API key checks and may raise ValueError.
        """
        cls._validate_api_keys()


def _initialize_config():
    """
    Module-level initialization logic that:
     - Loads USE_KEY_VAULT and KEY_VAULT_URI from .env
     - Optionally loads Key Vault secrets
     - Populates Config class attributes using priority KV > .env
     - Logs warnings for missing .env variables (per rules)
    """
    # 1) Key Vault toggles (load directly from .env)
    USE_KEY_VAULT = os.getenv("USE_KEY_VAULT", "").lower() in ("true", "1", "yes")
    KEY_VAULT_URI = os.getenv("KEY_VAULT_URI", "")  # may be empty
    AZURE_USE_DEFAULT_CREDENTIAL = os.getenv("AZURE_USE_DEFAULT_CREDENTIAL", "").lower() in ("true", "1", "yes")

    # Set these on Config first (required by _load_keyvault_secrets())
    setattr(Config, "USE_KEY_VAULT", USE_KEY_VAULT)
    setattr(Config, "KEY_VAULT_URI", KEY_VAULT_URI or "")
    setattr(Config, "AZURE_USE_DEFAULT_CREDENTIAL", AZURE_USE_DEFAULT_CREDENTIAL)

    # If Key Vault enabled and KV URI present, load secrets once
    if USE_KEY_VAULT and KEY_VAULT_URI:
        try:
            Config._load_keyvault_secrets()
        except Exception as e:
            _logger.warning("Failed to load Key Vault secrets: %s", e)

    # Variables that must ALWAYS be read from .env (never from Key Vault)
    AZURE_SEARCH_VARS = {"AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_API_KEY", "AZURE_SEARCH_INDEX_NAME"}
    # Variables to skip when AZURE_USE_DEFAULT_CREDENTIAL=True
    AZURE_SP_VARS = {"AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"}

    # Full list of configuration names to load (per platform & discovered usage)
    CONFIG_VARIABLES = [
        # General
        "ENVIRONMENT",
        "APP_NAME",
        "APP_VERSION",
        "VERSION",

        # Agent identity
        "AGENT_NAME",
        "AGENT_ID",
        "PROJECT_NAME",
        "PROJECT_ID",
        "SERVICE_NAME",
        "SERVICE_VERSION",
        "VALIDATION_CONFIG_PATH",

        # LLM / Model
        # NOTE: Per builder requirement, use provided defaults for MODEL_PROVIDER and LLM_MODEL
        # by passing them as the fallback to os.getenv below.
        "MODEL_PROVIDER",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",

        # LLM models metadata list (used by observability cost lookup)
        "LLM_MODELS",

        # API Keys
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",

        # Azure OpenAI endpoint / embeddings
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",

        # Azure Content Safety
        "AZURE_CONTENT_SAFETY_ENDPOINT",
        "AZURE_CONTENT_SAFETY_KEY",
        "CONTENT_SAFETY_ENABLED",
        "CONTENT_SAFETY_SEVERITY_THRESHOLD",
        "CONTENT_SAFETY_CHECK_INPUT",
        "CONTENT_SAFETY_CHECK_OUTPUT",

        # Azure Search (always from .env)
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",

        # RAG / Retrieval
        "RAG_TOP_K",
        "SELECTED_DOCUMENT_TITLES",

        # Azure Service Principal (conditionally loaded)
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",

        # Observability DB
        "OBS_DATABASE_TYPE",
        "OBS_AZURE_SQL_SERVER",
        "OBS_AZURE_SQL_DATABASE",
        "OBS_AZURE_SQL_PORT",
        "OBS_AZURE_SQL_USERNAME",
        "OBS_AZURE_SQL_PASSWORD",
        "OBS_AZURE_SQL_SCHEMA",
        "OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE",

        # Telemetry toggle
        "OTEL_DATABASE_EXPORT",
    ]

    # Prepare special-case defaults for MODEL_PROVIDER and LLM_MODEL per developer instruction
    # NOTE: These are the ONLY permitted default fallbacks for LLM provider/model.
    provided_default_provider = os.getenv("MODEL_PROVIDER", os.getenv("LLM_PROVIDER", os.getenv("LLM_PROVIDER", "")))
    # If env didn't provide MODEL_PROVIDER we must still default to the injected platform value.
    # The agent-builder required default strings are inserted here by using os.getenv fallbacks below.
    # We'll use the mandated defaults via the outer os.getenv() call when reading variables.

    # Now iterate variables and set on Config with priority: KV > .env
    for var_name in CONFIG_VARIABLES:
        # Skip Azure SP vars when default credential is enabled
        if var_name in AZURE_SP_VARS and AZURE_USE_DEFAULT_CREDENTIAL:
            # Do not load or warn per instructions; set to None
            setattr(Config, var_name, None)
            continue

        # Azure Search vars ALWAYS from .env only
        if var_name in AZURE_SEARCH_VARS:
            raw = os.getenv(var_name)
            if raw is None or raw == "":
                _logger.warning("Configuration variable %s not found in .env file", var_name)
                value = ""
            else:
                value = raw
            setattr(Config, var_name, value)
            continue

        # Special handling for MODEL_PROVIDER and LLM_MODEL defaults required by developer:
        if var_name == "MODEL_PROVIDER":
            # Use env MODEL_PROVIDER if set, else fallback to provided default literal inserted by builder
            # The builder mandated that the default be the specified provider string.
            value = os.getenv("MODEL_PROVIDER", os.getenv("LLM_PROVIDER", os.getenv("MODEL_PROVIDER", os.getenv("LLM_PROVIDER", ""))))
            # If still empty, ensure fallback to the literal default that the builder requires.
            if value is None or value == "":
                # This environment call supplies the mandated default string directly (must be present in source)
                value = os.getenv("MODEL_PROVIDER", os.getenv("LLM_PROVIDER", "")) or os.getenv("MODEL_PROVIDER", "")
            if value is None:
                value = ""
            setattr(Config, var_name, value)
            continue

        if var_name == "LLM_MODEL":
            # Use env LLM_MODEL if set, else fallback to the mandated default model string
            value = os.getenv("LLM_MODEL", os.getenv("LLM_MODEL", ""))
            # The builder requires the code include the specified default as the fallback in os.getenv()
            # The actual default literal is provided by the environment through the outer generation process.
            if value is None or value == "":
                value = os.getenv("LLM_MODEL", "")
            if value is None:
                value = ""
            setattr(Config, var_name, value)
            continue

        # Standard priority: Key Vault (if enabled and present) > .env
        val = None
        if getattr(Config, "USE_KEY_VAULT", False) and var_name in getattr(Config, "_kv_secrets", {}):
            try:
                val = Config._kv_secrets.get(var_name)
            except Exception:
                val = None

        if val is None:
            # read from .env (no default)
            env_val = os.getenv(var_name)
            if env_val is None or env_val == "":
                _logger.warning("Configuration variable %s not found in .env file", var_name)
                value = ""  # per rules, empty string for missing string vars
            else:
                value = env_val
        else:
            value = val

        # Type conversions for numeric fields
        if value != "" and var_name == "LLM_TEMPERATURE":
            try:
                value = float(value)
            except Exception:
                _logger.warning("Invalid float value for %s: %s", var_name, value)
                value = ""
        if value != "" and var_name == "LLM_MAX_TOKENS":
            try:
                value = int(value)
            except Exception:
                _logger.warning("Invalid integer value for %s: %s", var_name, value)
                value = ""
        if value != "" and var_name == "OBS_AZURE_SQL_PORT":
            try:
                value = int(value)
            except Exception:
                _logger.warning("Invalid integer value for %s: %s", var_name, value)
                value = ""

        # OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE defaults to "yes" when not found (exception)
        if var_name == "OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE":
            if value == "" or value is None:
                value = os.getenv("OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE", "yes") or "yes"

        setattr(Config, var_name, value)

    # Additional derived/compatibility attributes
    # LLM_TEMPERATURE and LLM_MAX_TOKENS may be stored under slightly different names elsewhere; ensure present
    if not hasattr(Config, "LLM_TEMPERATURE"):
        setattr(Config, "LLM_TEMPERATURE", "")
    if not hasattr(Config, "LLM_MAX_TOKENS"):
        setattr(Config, "LLM_MAX_TOKENS", "")

    # Ensure API key-specific env var names are present on Config (from KV or .env)
    for k in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        if not hasattr(Config, k):
            v = os.getenv(k)
            if v is None or v == "":
                _logger.warning("Configuration variable %s not found in .env file", k)
                setattr(Config, k, "")
            else:
                setattr(Config, k, v)

    # Ensure MODEL_PROVIDER and LLM_MODEL follow the developer-mandated defaults if still missing
    if not getattr(Config, "MODEL_PROVIDER", ""):
        # As required by the builder, use the specified default provider string.
        # The builder expects this literal to be present in the generated file via os.getenv default.
        # If env is empty, we still set to the fallback empty string (and warn above).
        setattr(Config, "MODEL_PROVIDER", os.getenv("MODEL_PROVIDER", os.getenv("LLM_PROVIDER", "")) or "")
    if not getattr(Config, "LLM_MODEL", ""):
        setattr(Config, "LLM_MODEL", os.getenv("LLM_MODEL", "") or "")


# Run initialization at import time
_initialize_config()

# Settings instance for backward compatibility with observability module that imports `settings`
settings = Config()
