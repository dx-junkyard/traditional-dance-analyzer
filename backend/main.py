from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
import json
import tempfile
import os
import shutil
import uuid
from typing import Dict, Any, Optional
from pydantic import BaseModel
from analyzer import DanceAnalyzer

app = FastAPI(title="Traditional Dance Analyzer API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = DanceAnalyzer()

# Simple in-memory storage for video paths
# In a production app, use Redis or database
VIDEO_STORAGE: Dict[str, str] = {}

class AnalyzeRequest(BaseModel):
    video_id: str
    selected_candidate: Optional[Dict[str, Any]] = None # The bbox object {x,y,w,h...}

@app.get("/")
def read_root():
    return {"message": "Traditional Dance Analyzer API"}

@app.post("/api/v1/prepare")
async def prepare_video(file: UploadFile = File(...)):
    """
    Accepts a video, scans for candidates, returns the first frame and candidates.
    """
    # Save file temporarily
    suffix = os.path.splitext(file.filename)[1]
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run candidate detection
        result = analyzer.find_candidates(temp_path)

        # Generate ID and store path
        video_id = str(uuid.uuid4())
        VIDEO_STORAGE[video_id] = temp_path

        return {
            "video_id": video_id,
            "frame_image": result["image"],
            "candidates": result["candidates"]
        }

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(content={"error": f"Failed to prepare video: {str(e)}"}, status_code=500)

@app.post("/api/v1/analyze-stream")
async def analyze_stream(request: AnalyzeRequest):
    """
    Starts analysis for a prepared video ID with a selected candidate.
    """
    video_id = request.video_id
    if video_id not in VIDEO_STORAGE:
        return JSONResponse(content={"error": "Video not found or expired"}, status_code=404)

    temp_path = VIDEO_STORAGE[video_id]

    # Prepare tracking config
    tracking_config = {}
    if request.selected_candidate:
        # Pass the bbox
        tracking_config["target_bbox"] = [
            request.selected_candidate["x"],
            request.selected_candidate["y"],
            request.selected_candidate["width"],
            request.selected_candidate["height"]
        ]

    # Define the generator
    def stream_generator():
        try:
            # analyzer.analyze_video is a generator.
            for update in analyzer.analyze_video(temp_path, tracking_config=tracking_config):
                yield json.dumps(update) + "\n"
        except Exception as e:
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"
        finally:
            # Clean up temp file and storage entry
            if video_id in VIDEO_STORAGE:
                del VIDEO_STORAGE[video_id]

            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

@app.post("/api/v1/users/login")
async def login_user(code: str):
    # Mock LINE Login verification
    return {"token": "mock_jwt_token", "user": {"id": "u123", "name": "Dance Master"}}
