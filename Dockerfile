# 1. Use an official Python 3.11 slim image
# 'slim' is a lightweight base image, ideal for production.
FROM python:3.11-slim

# 2. Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk.
# PYTHONUNBUFFERED: Ensures Python output (logs) is sent straight to the terminal without buffering.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install system dependencies
# These tools (gcc, libpq-dev, etc.) are REQUIRED for installing 'psycopg2' (Postgres) and 'faiss'.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements file first
# This allows Docker to cache the installed dependencies layer if requirements.txt hasn't changed.
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of the application code
COPY . .

# 8. Expose the port (Documentation only)
EXPOSE 8000

# 9. Start the application
# We use 'sh -c' to support the ${PORT:-8000} syntax.
# This means: "Use the PORT environment variable if it exists (for Cloud); otherwise, default to 8000 (for Local)."
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"