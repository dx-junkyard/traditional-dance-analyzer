from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Body, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
import asyncio
import json
import tempfile
import os
import shutil
import uuid
from typing import Dict, Any, Optional
from pydantic import BaseModel
from analyzer import DanceAnalyzer
from database import save_analysis_result, SessionLocal, create_tables, get_db, DanceAnalysis

# Create tables on startup (simple approach for this task)
create_tables()

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
                # Check for completion to save data
                if update.get("status") == "complete" and "result" in update:
                    try:
                        db = SessionLocal()
                        try:
                            save_analysis_result(db, video_id, update["result"])
                        finally:
                            db.close()
                    except Exception as db_e:
                        print(f"Failed to save analysis result: {db_e}")
                        # We don't stop the stream, just log error (or send to client)

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

@app.get("/api/v1/results/{video_id}")
def get_analysis_result(video_id: str, db: Session = Depends(get_db)):
    """
    Retrieve a saved analysis result by video_id.
    """
    result = db.query(DanceAnalysis).filter(DanceAnalysis.video_id == video_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Analysis result not found")

    return {
        "id": result.id,
        "video_id": result.video_id,
        "created_at": result.created_at,
        "metrics": {
            "stability_score": result.stability_score,
            "rhythm_score": result.rhythm_score,
            "dynamism_score": result.dynamism_score
        },
        "audio_summary": {
            "tempo": result.tempo,
            "duration": result.duration
        },
        "pose_data": result.pose_data,
        "audio_data": result.audio_data
    }

@app.post("/api/v1/users/login")
async def login_user(code: str):
    # Mock LINE Login verification
    return {"token": "mock_jwt_token", "user": {"id": "u123", "name": "Dance Master"}}
