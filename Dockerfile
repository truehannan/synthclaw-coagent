FROM python:3.12-slim

WORKDIR /opt/conclave

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Build frontend (if node available)
RUN if command -v node > /dev/null 2>&1; then \
      cd frontend && npm install && npm run build; \
    fi || true

# Create workspace
RUN mkdir -p /opt/conclave/workspace

# Environment
ENV CONCLAVE_BASE_DIR=/opt/conclave
ENV CONCLAVE_API_PORT=8000
ENV CONCLAVE_API_HOST=0.0.0.0
ENV INTERFACE_MODE=cli

EXPOSE 8000

CMD ["python3", "main.py"]
