FROM python:3.11-slim

WORKDIR /app

# Install system deps + Chrome for Selenium
RUN apt-get update && apt-get install -y \
    gcc g++ curl wget gnupg unzip \
    chromium chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Set Chrome path for Selenium
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create dirs
RUN mkdir -p data models app/static

EXPOSE 5000
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python", "app/app.py"]
