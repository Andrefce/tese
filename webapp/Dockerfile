FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies — separate layer so code changes don't bust the cache
COPY requirements.txt .
RUN uv pip install --system --compile-bytecode -r requirements.txt

# Copy application code (model/ is excluded — inference runs remotely)
COPY app.py .
COPY core/ core/
COPY demo-data/ demo-data/
COPY templates/ templates/
COPY static/ static/

# Create instance directory for uploads
RUN mkdir -p instance/uploads

ENV PORT=8080

EXPOSE 8080

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "600"]
