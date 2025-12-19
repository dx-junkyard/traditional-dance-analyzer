import cv2
import mediapipe as mp
import numpy as np
import librosa
import json
import tempfile
import os
from fastapi import UploadFile

class DanceAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )

    async def analyze_video(self, file: UploadFile, status_callback=None):
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)

        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # 1. Pose Analysis with MediaPipe
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pose_data = []
        frames_processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)

            frame_data = {
                "frame": frames_processed,
                "timestamp": frames_processed / fps if fps else 0,
                "landmarks": []
            }

            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    frame_data["landmarks"].append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility
                    })

            pose_data.append(frame_data)
            frames_processed += 1

            if status_callback and frames_processed % 30 == 0:
                await status_callback(frames_processed / frame_count * 0.5) # 50% for visual

        cap.release()

        # 2. Audio Analysis with Librosa
        try:
            y, sr = librosa.load(temp_path)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

            # Simple rhythm consistency check (variance in onset strength)
            rhythm_consistency = float(np.std(onset_env))

            audio_data = {
                "tempo": float(tempo),
                "rhythm_consistency": rhythm_consistency,
                "duration": librosa.get_duration(y=y, sr=sr)
            }
        except Exception as e:
            print(f"Audio analysis failed: {e}")
            audio_data = {"error": str(e)}

        if status_callback:
            await status_callback(1.0)

        # Cleanup
        os.remove(temp_path)
        os.rmdir(temp_dir)

        return {
            "pose_data": pose_data,
            "audio_data": audio_data,
            "metrics": self.calculate_metrics(pose_data, audio_data)
        }

    def calculate_metrics(self, pose_data, audio_data):
        # Placeholder for complex metrics
        # "腰の据わり" (Stability of hips - landmarks 23, 24)
        # "リズム調和度" (Correlation between movement and beat)
        # "序破急" (Dynamic changes in velocity)

        return {
            "stability_score": 0.85, # Mock score
            "rhythm_score": 0.92,    # Mock score
            "dynamism_score": 0.78   # Mock score
        }
