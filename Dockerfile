FROM python:3.9-slim

WORKDIR /app

# Install git for repository cloning
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create a writable directory for the workspace in cloud environments
RUN mkdir -p /tmp/workspace && chmod 777 /tmp/workspace

# Set environment variable for workspace
ENV WORKSPACE_DIR=/tmp/workspace
ENV PYTHONUNBUFFERED=1

# Expose port (Render will override this, but good for documentation)
EXPOSE 8000

# Use shell form to expand environment variable
CMD sh -c "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"
