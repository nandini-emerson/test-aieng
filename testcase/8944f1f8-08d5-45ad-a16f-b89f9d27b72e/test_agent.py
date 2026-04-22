# NOTE: If you see "Unknown pytest.mark.X" warnings, create a conftest.py file with:
# import pytest
# def pytest_configure(config):
#     config.addinivalue_line("markers", "performance: mark test as performance test")
#     config.addinivalue_line("markers", "security: mark test as security test")
#     config.addinivalue_line("markers", "integration: mark test as integration test")


import re
import logging
import pytest

from agent import AuditLogger

@pytest.mark.security
@pytest.mark.asyncio
async def AuditLogger(caplog):
    """
    Security test to ensure AuditLogger.log_event redacts common PII keys (ssn, card, cvv, dob, credit)
    in the logged payload and returns a UUID-like log id without raising.
    """
    caplog.set_level(logging.INFO, logger="enrollment_agent")

    audit = AuditLogger()
    payload = {
        "name": "Alice",
        "ssn": "123-45-6789",
        "credit_card": "4111-1111-1111-1111",
        "notes": "ok"
    }

    # Call the async log_event and ensure no exception is raised
    log_id = await audit.log_event("enrollment_request", "student-123", payload)

    # Basic return value checks
    assert log_id is not None
    assert isinstance(log_id, str)
    # UUID v4-like pattern (hex groups)
    uuid_re = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
    assert uuid_re.match(log_id), f"Returned log_id does not match UUID pattern: {log_id}"

    # Logger output should contain the redacted marker and not contain raw PII values
    logged = caplog.text

    assert "[REDACTED]" in logged, "Expected '[REDACTED]' to appear in logs for PII redaction"
    assert "123-45-6789" not in logged, "Raw SSN should not appear in logs"
    assert "4111-1111-1111-1111" not in logged, "Raw credit card should not appear in logs"
    # Also ensure the user id appears in the audit log entry for context
    assert "student-123" in logged