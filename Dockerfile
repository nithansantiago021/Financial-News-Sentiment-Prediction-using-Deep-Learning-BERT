FROM python:3.11-slim

# Install system dependencies if any are needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /lib/apt/lists/*

WORKDIR /app

# 1. Copy and install requirements as ROOT first (Fixes the writeable error)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 2. Now create the secure, unprivileged user for Hugging Face Spaces compliance
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 3. Copy the rest of your app code
COPY --chown=user . .

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]