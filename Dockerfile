FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Railway injects $PORT at runtime
ENV PORT=8000

# Start with python (not uvicorn directly) for better error output
CMD ["python", "main.py"]
