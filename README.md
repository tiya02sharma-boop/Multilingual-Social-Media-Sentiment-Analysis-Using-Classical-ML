# 🌌 VibeCheck

**VibeCheck** is a premium, AI-powered YouTube Comment Sentiment Analysis platform. It automatically extracts comments from any YouTube video, cleans them, translates them if necessary, and analyzes them using a trained Machine Learning model. 

Instead of manually reading thousands of comments, users get an interactive, beautifully designed dark-mode dashboard displaying real-time audience sentiment, key metrics, emotion mapping, visual charts, and an AI-powered executive summary.

---

## ✨ Features

- **💡 Sentiment Metrics**: Automatically calculates Sentiment Score, Reception Rate, Engagement Level, Satisfaction Rate, and Toxicity estimation.
- **📊 Visual Analytics**: Custom interactive charts including a Sentiment Breakdown Donut, a Sentiment Timeline, and a Sentiment Distribution Bar Chart.
- **🏷️ Keyword Cloud & Emotion Analysis**: Highlights trending words and maps comments into 5 core emotions (Happy, Neutral, Angry, Appreciation, Curiosity).
- **🔎 Comment Explorer**: Filter and search positive, neutral, and negative comments interactively.
- **✦ AI Summaries**: Uses Google Gemini to generate high-level, human-like summaries and actionable recommendations for creators.
- **📄 Report Exports**: Single-click PDF Report exporting (formatted for presentations) and raw data CSV downloads.

---

## 🛠️ Technology Stack & Architecture

### Backend (Python / FastAPI)
- **FastAPI**: Serves both the ML inference API and hosts the frontend static files.
- **Scikit-learn**: Core ML model utilizing a trained `LinearSVC` classifier.
- **TF-IDF Vectorization**: Word-level and char-level combined transformer for text representation.
- **Deep Translator & Langdetect**: Automatically detects comment language and translates to English for analysis.
- **Google Gemini API**: Generates advanced, contextual summaries.

### Frontend (HTML5 / Vanilla CSS3 / JS)
- **Design System**: Premium glassmorphism dark-mode theme utilizing pure, modern CSS variables, blur filters, and micro-interactions.
- **Chart.js**: Client-side rendering of interactive donut and timeline charts.
- **html2pdf.js**: Generates client-side PDF reports directly from dashboard elements.

### Directory Structure

```
├── data/                         # ML Model Artifacts & Datasets
│   ├── best_sentiment_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── label_encoder.joblib
├── notebooks/                    # Backend API & Jupyter Notebooks
│   ├── app.py                    # FastAPI server entry point
│   ├── inference.py              # ML inference pipeline (cleaning, translating, prediction)
│   ├── youtube_sentiment.py      # YouTube API fetch helper
│   ├── 01_eda.ipynb ...          # Step-by-step notebooks (EDA, Training, Evaluation)
│   └── .env                      # Backend API keys config
├── config.js                     # Frontend API keys config
├── index.html                    # Frontend main template
├── script.js                     # Frontend application logic
├── styles.css                    # Premium glassmorphism style rules
├── requirements.txt              # Python packages dependencies
└── Dockerfile                    # Container configuration file
```

---

## 🚀 Quick Start (Local Run)

### 1. Setup API Keys
Before running, you need keys for **YouTube Data API v3** and **Google Gemini API**.

- Rename/create `.env` in the `notebooks/` directory and add your keys:
  ```env
  YOUTUBE_API_KEY=your_youtube_api_key_here
  GEMINI_API_KEY=your_gemini_api_key_here
  ```
- Update `config.js` in the project root with the same keys:
  ```javascript
  const CONFIG = {
    YOUTUBE_API_KEY: "your_youtube_api_key_here",
    GEMINI_API_KEY: "your_gemini_api_key_here"
  };
  ```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Run the Unified Server
Run the FastAPI app from the `notebooks` directory:
```bash
cd notebooks
uvicorn app:app --reload
```

### 4. Access the App
Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 🐳 Docker Deployment

You can also run VibeCheck completely containerized:

### 1. Build the Docker Image
```bash
docker build -t vibecheck .
```

### 2. Run the Container
Make sure to mount your `.env` or pass the environment keys during the run:
```bash
docker run -p 8000:8000 vibecheck
```
Access the application at `http://localhost:8000`.
