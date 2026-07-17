import os
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference import run_pipeline, run_pipeline_batch

app = FastAPI(title="Comment Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root directory of the project containing frontend assets
ROOT_DIR = Path(__file__).resolve().parent.parent


class CommentRequest(BaseModel):
    text: str


class BatchCommentRequest(BaseModel):
    texts: List[str]


@app.get("/")
def root():
    index_path = ROOT_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"status": "ok", "message": "Frontend index.html not found"}


@app.post("/predict")
def predict(payload: CommentRequest):
    return run_pipeline(payload.text)


@app.post("/predict/batch")
def predict_batch(payload: BatchCommentRequest):
    return run_pipeline_batch(payload.texts)


# Mount the root directory to serve static assets (css, js, config)
app.mount("/", StaticFiles(directory=ROOT_DIR), name="static")