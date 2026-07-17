FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better build caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything: root frontend files (index.html, config.js, script.js,
# styles.css), notebooks/ (app.py, inference.py, utils), and data/
# (model artifacts) — app.py serves the frontend from the project root
# AND the API from the same process, so both need to be in the image.
COPY . .

# app.py does "from inference import ..." (not "notebooks.inference"),
# so uvicorn has to run with notebooks/ as the working directory.
WORKDIR /app/notebooks

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
