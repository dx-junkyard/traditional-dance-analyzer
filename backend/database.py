from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
import os
import uuid
import datetime

# Use environment variables or default for dev
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "mysql+mysqlconnector://user:password@db/dance_db")

# Add some connection arguments for stability if needed, but defaults are usually fine for basic usage
engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class DanceAnalysis(Base):
    __tablename__ = "dance_analyses"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String(255), index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Metrics
    stability_score = Column(Float)
    rhythm_score = Column(Float)
    dynamism_score = Column(Float)

    # Audio Summary
    tempo = Column(Float)
    duration = Column(Float)

    # Detailed Data (Large JSON)
    # Using JSON type for MySQL compatibility. If using SQLite for local dev, generic JSON works too in recent SQLAlchemy versions.
    pose_data = Column(JSON)
    audio_data = Column(JSON) # Stores onset_env, etc.

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Helper to create tables if they don't exist."""
    Base.metadata.create_all(bind=engine)

def save_analysis_result(db: Session, video_id: str, result: dict):
    """
    Parses the analyzer result dictionary and saves it to the database.
    """
    metrics = result.get("metrics", {})
    audio = result.get("audio_data", {})

    # Safely extract audio summary
    tempo = audio.get("tempo")
    if isinstance(tempo, (list, tuple)):
        tempo = tempo[0] # librosa might return list

    # Ensure audio_data is JSON serializable (numpy arrays to list)
    # The analyzer seems to handle some of this, but we should be careful.
    # We will trust the analyzer output is mostly JSON-ready, but `onset_env` in `audio` might be a list already.

    analysis = DanceAnalysis(
        video_id=video_id,
        stability_score=metrics.get("stability_score"),
        rhythm_score=metrics.get("rhythm_score"),
        dynamism_score=metrics.get("dynamism_score"),
        tempo=float(tempo) if tempo else None,
        duration=float(audio.get("duration")) if audio.get("duration") else None,
        pose_data=result.get("pose_data"),
        audio_data=audio
    )

    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis
