# 1. Use Python 3.11 (the small, fast version)
FROM python:3.11-slim

# 2. Set environment settings to keep Python fast and quiet
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8001

# 3. Create and go to the /app folder
WORKDIR /app

# 4. Install Git (needed by some Python libraries)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 5. Copy and install your Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 6. Copy your code and frontend files
COPY . .
# 7. Create folders for your database and PDF uploads
RUN mkdir -p /app/data /app/uploads/pdfs

# 8. Open Port 8001 for your browser
EXPOSE 8001

# 9. Start the server
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
