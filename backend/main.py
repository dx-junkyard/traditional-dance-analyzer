from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
import json
import tempfile
import os
import shutil
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

@app.get("/")
def read_root():
    return {"message": "Traditional Dance Analyzer API"}

@app.post("/api/v1/analyze-stream")
async def analyze_stream(file: UploadFile = File(...)):
    # Save file temporarily
    # We cannot use NamedTemporaryFile because Windows/some environments block opening it twice
    # and we need to pass path to analyzer.
    fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1])
    os.close(fd)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        os.remove(temp_path)
        return JSONResponse(content={"error": f"Failed to save upload: {str(e)}"}, status_code=500)

    # Define the generator
    def stream_generator():
        try:
            # analyzer.analyze_video is a generator.
            # StreamingResponse will iterate this in a threadpool (because it's not async gen).
            for update in analyzer.analyze_video(temp_path):
                yield json.dumps(update) + "\n"
        except Exception as e:
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

@app.post("/api/v1/users/login")
async def login_user(code: str):
    # Mock LINE Login verification
    return {"token": "mock_jwt_token", "user": {"id": "u123", "name": "Dance Master"}}
