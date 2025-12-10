FROM python:3.11-slim

# System updates & FAISS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to backend
WORKDIR /app

# Copy dependency file first to leverage Docker caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app /app/app

# Expose port for FastAPI
EXPOSE 8000

# Set environment variables here or override in deployment
ENV OPENAI_MODEL="gpt-4o-mini"

# Production server command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
