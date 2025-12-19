from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
import json
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
status_stream = {} # Simple in-memory status store

@app.get("/")
def read_root():
    return {"message": "Traditional Dance Analyzer API"}

@app.post("/api/v1/analyze")
async def analyze_video(file: UploadFile = File(...)):
    # We create a simple status tracking mechanism
    task_id = file.filename # Simplification for demo
    status_stream[task_id] = 0.0

    async def update_status(progress):
        status_stream[task_id] = progress

    try:
        results = await analyzer.analyze_video(file, status_callback=update_status)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/v1/status/{task_id}")
async def get_status(task_id: str):
    # This endpoint supports Server-Sent Events (SSE) conceptually
    # but for now just returns current status
    async def event_generator():
        while True:
            if task_id in status_stream:
                progress = status_stream[task_id]
                yield f"data: {json.dumps({'progress': progress})}\n\n"
                if progress >= 1.0:
                    break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/v1/users/login")
async def login_user(code: str):
    # Mock LINE Login verification
    # in production, verify 'code' with LINE API
    return {"token": "mock_jwt_token", "user": {"id": "u123", "name": "Dance Master"}}
