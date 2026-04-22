

FROM python:3.11-slim

# Install minimal OS packages required by our Python dependencies + curl for HEALTHCHECK
# unixodbc-dev is required for pyodbc; build-essential/gcc/g++ may be needed for packages that compile wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        unixodbc-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install Python dependencies (cacheable layer)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code into working directory
COPY code/ .

# Add a non-root user and switch to it
RUN addgroup --system appgroup && adduser --system --ingroup appgroup --no-create-home appuser

USER appuser

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AGENT_NAME=""

# Expose application port
EXPOSE 8000

# Healthcheck: use curl to poll the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run FastAPI app using uvicorn (agent.py defines `app`)
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]

